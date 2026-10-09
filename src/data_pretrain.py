"""
预训练数据管线: FineWeb-Edu (sample-10BT, ~100 亿 token) -> 自训 BPE 分词 -> 二进制分片

输出布局 (nanoGPT 风格, 训练时可直接 np.memmap):
  {out_dir}/train/shard_{i:05d}.bin    每片 1 亿 token, uint32
  {out_dir}/val.bin                    末尾 1 万篇文档作为验证集

用法 (本地 parquet, 绕过 HF 数据集枚举):
  python data_pretrain.py --local_dir /root/Test/mini-r1/data/raw \
      --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json --out_dir /root/Test/mini-r1/data
"""
import argparse
import glob
import os
import struct
import time

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from tokenizers import Tokenizer

TOKENS_PER_SHARD = 100_000_000   # 每分片 1 亿 token (uint32, 400MB)
VAL_DOCS = 10_000                # 末尾文档数用作验证集
EOT = 2                          # </s>, 文档间分隔符
NUM_PROC = 16                    # tokenize 子进程数 (128 核机器, 负载低时可用 16)


_TOK_CACHE = {}

def tokenize_batch(batch, tokenizer_path):
    """对一批文档做 BPE 编码, 每篇末尾追加 </s>。datasets.map 的 batched 回调。

    每个 worker 进程缓存一份分词器, 避免每批重复加载 (10B tokens 有 30 万批)。
    """
    tok = _TOK_CACHE.get(tokenizer_path)
    if tok is None:
        tok = _TOK_CACHE[tokenizer_path] = Tokenizer.from_file(tokenizer_path)
    out = tok.encode_batch(batch["text"])
    return {"ids": [e.ids + [EOT] for e in out]}


class ShardWriter:
    """把 tokenize 后的文档流写入 1 亿 token/片的 uint32 分片, 状态跨文件延续。"""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.buf = np.empty(TOKENS_PER_SHARD, dtype=np.uint32)
        self.shard_idx, self.pos, self.total = 0, 0, 0
        self.t0 = time.time()

    def write(self, ids):
        n = len(ids)
        if self.pos + n > TOKENS_PER_SHARD:
            self.buf[self.pos:].tofile(f"{self.out_dir}/train/shard_{self.shard_idx:05d}.bin")
            self.shard_idx += 1
            self.pos = 0
        self.buf[self.pos:self.pos + n] = ids
        self.pos += n
        self.total += n
        if self.total % (TOKENS_PER_SHARD // 4) < 256:
            rate = self.total / (time.time() - self.t0)
            print(f"  {self.total:,} tokens, {rate:,.0f} tok/s, shard={self.shard_idx}", flush=True)

    def close(self):
        if self.pos:
            self.buf[:self.pos].tofile(f"{self.out_dir}/train/shard_{self.shard_idx:05d}.bin")
        print(f"训练集完成: {self.total:,} tokens, {self.shard_idx + 1} 个分片")
        return self.total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--local_dir", default=None,
                    help="从本地 parquet 目录读取 (绕过 HF 数据集枚举)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_docs", type=int, default=VAL_DOCS)
    args = ap.parse_args()

    os.makedirs(f"{args.out_dir}/train", exist_ok=True)
    if args.local_dir:
        files = sorted(glob.glob(f"{args.local_dir}/*.parquet"))
        assert files, f"{args.local_dir} 下没有 parquet 文件"
        # 行数从 parquet 元数据直接读, 不用遍历 (每个文件只需读 footer)
        n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"本地 {len(files)} 个 parquet, 共 {n:,} 篇文档 (元数据统计)")
        print(f"验证集取末尾 {args.val_docs} 篇 (最后一个文件)")

        # streaming 的 map 不支持 num_proc, 改逐文件非流式加载:
        # 内存峰值 = 单文件文本 (~8GB) + ids, 而不是全量 100GB+
        map_kwargs = dict(batched=True, batch_size=64, num_proc=NUM_PROC,
                          fn_kwargs={"tokenizer_path": args.tokenizer},
                          remove_columns=["text"])
        writer = ShardWriter(args.out_dir)
        for idx, f in enumerate(files):
            print(f"[{idx + 1}/{len(files)}] 处理 {os.path.basename(f)} ...")
            ds = load_dataset("parquet", data_files=f, split="train")
            if idx == len(files) - 1 and args.val_docs > 0:
                val = ds.select(range(len(ds) - args.val_docs, len(ds)))
                val = val.map(tokenize_batch, **map_kwargs)
                total = 0
                with open(f"{args.out_dir}/val.bin", "wb") as fv:
                    for batch in val:
                        for ids in batch["ids"]:
                            fv.write(struct.pack(f"{len(ids)}I", *ids))
                            total += len(ids)
                print(f"验证集完成: {total:,} tokens -> {args.out_dir}/val.bin")
                del val
                ds = ds.select(range(len(ds) - args.val_docs))
            ds = ds.map(tokenize_batch, **map_kwargs)
            for batch in ds:
                for ids in batch["ids"]:
                    writer.write(ids)
            del ds
        writer.close()
    else:
        ds = load_dataset(args.dataset, args.name, split="train", streaming=True)
        n = ds.info.splits["train"].num_examples
        print(f"数据集共 {n:,} 篇文档")
        # hub 流式回退: 单进程 map (慢, 仅作备用路径)
        ds = ds.map(tokenize_batch, batched=True, batch_size=64,
                    fn_kwargs={"tokenizer_path": args.tokenizer},
                    remove_columns=list(ds.features.keys()) if ds.features else None)
        writer = ShardWriter(args.out_dir)
        for batch in ds:
            for ids in batch["ids"]:
                writer.write(ids)
        writer.close()


if __name__ == "__main__":
    main()
