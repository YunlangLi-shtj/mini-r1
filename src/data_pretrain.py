"""
预训练数据管线: FineWeb-Edu (sample-10BT, ~100 亿 token) -> 自训 BPE 分词 -> 二进制分片

输出布局 (nanoGPT 风格, 训练时可直接 np.memmap):
  {out_dir}/train/shard_{i:05d}.bin    每片 1 亿 token, uint32
  {out_dir}/val.bin                    末尾 1 万篇文档作为验证集

实现说明: 不用 datasets.map — 它对 ragged list-of-lists / bytes 返回值的
行拆解行为不可靠 (会把 token 展平成多行)。改为 pyarrow 按 RecordBatch 流式读
parquet, tokenizers 的 Rust 多线程 encode_batch 编码, 内存峰值 = 单个批次。

用法 (本地 parquet, 绕过 HF 数据集枚举):
  python data_pretrain.py --local_dir /root/Test/mini-r1/data/raw \
      --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json --out_dir /root/Test/mini-r1/data
"""
import argparse
import glob
import os
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

TOKENS_PER_SHARD = 100_000_000   # 每分片 1 亿 token (uint32, 400MB)
VAL_DOCS = 10_000                # 末尾文档数用作验证集
EOT = 2                          # </s>, 文档间分隔符
READ_BATCH = 2000                # 每次从 parquet 读取的行数


def tok_pack(tok, texts):
    """编码一批文本, 每篇末尾追加 </s>, 返回 uint32 数组列表 (空文本跳过)。"""
    texts = [t for t in texts if t]
    if not texts:
        return []
    return [np.asarray(e.ids + [EOT], dtype=np.uint32) for e in tok.encode_batch(texts)]


class ShardWriter:
    """把 tokenize 后的文档流写入 shard_tokens/片的 uint32 分片, 状态跨文件延续。"""

    def __init__(self, out_dir: str, shard_tokens: int = TOKENS_PER_SHARD, start_idx: int = 0):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint32)
        self.shard_idx, self.pos, self.total = start_idx, 0, 0
        self.t0 = time.time()

    def write(self, ids):
        """ids: uint32 numpy 数组。"""
        n = len(ids)
        if self.pos + n > self.shard_tokens:
            # 写满的部分 [:pos], 不是空尾巴 [pos:] (原版照抄 nanoGPT 时的 bug)
            self.buf[:self.pos].tofile(f"{self.out_dir}/train/shard_{self.shard_idx:05d}.bin")
            self.shard_idx += 1
            self.pos = 0
        self.buf[self.pos:self.pos + n] = ids
        self.pos += n
        self.total += n
        if self.total % (self.shard_tokens // 4) < READ_BATCH * 8:
            rate = self.total / (time.time() - self.t0)
            print(f"  {self.total:,} tokens, {rate:,.0f} tok/s, shard={self.shard_idx}",
                  flush=True)

    def close(self):
        if self.pos:
            self.buf[:self.pos].tofile(f"{self.out_dir}/train/shard_{self.shard_idx:05d}.bin")
        print(f"训练集完成: {self.total:,} tokens, {self.shard_idx + 1} 个分片")
        return self.total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True,
                    help="本地 parquet 目录 (绕过 HF 数据集枚举)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_docs", type=int, default=VAL_DOCS)
    ap.add_argument("--max_files", type=int, default=0, help="只处理前 N 个文件 (调试用)")
    ap.add_argument("--shard_tokens", type=int, default=TOKENS_PER_SHARD,
                    help="每分片 token 数 (调试用小值)")
    ap.add_argument("--skip_files", type=int, default=0,
                    help="跳过前 N 个文件 (断点续跑用, 已产出的分片保留)")
    args = ap.parse_args()

    os.makedirs(f"{args.out_dir}/train", exist_ok=True)
    all_files = sorted(glob.glob(f"{args.local_dir}/*.parquet"))
    assert all_files, f"{args.local_dir} 下没有 parquet 文件"
    files = all_files[args.skip_files:]
    if args.max_files:
        files = files[:args.max_files]
    # 行数从 parquet footer 直接读, 不遍历数据
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"共 {len(all_files)} 个 parquet, 本次处理 {len(files)} 个 ({n:,} 篇文档)")
    print(f"验证集取最后一个文件的末尾 {args.val_docs} 篇")

    # 断点续跑: 新分片编号从已存在的分片数接着排
    start_idx = len(glob.glob(f"{args.out_dir}/train/shard_*.bin"))
    if start_idx:
        print(f"已有 {start_idx} 个分片, 从 shard_{start_idx:05d} 继续")

    tok = Tokenizer.from_file(args.tokenizer)
    writer = ShardWriter(args.out_dir, shard_tokens=args.shard_tokens, start_idx=start_idx)
    for idx, f in enumerate(files):
        pf = pq.ParquetFile(f)
        n_rows = pf.metadata.num_rows
        val_start = n_rows - args.val_docs if (idx == len(files) - 1 and args.val_docs > 0) else None
        val_f, val_total = None, 0
        if val_start is not None:
            val_f = open(f"{args.out_dir}/val.bin", "wb")
        print(f"[{args.skip_files + idx + 1}/{len(all_files)}] {os.path.basename(f)}: {n_rows:,} 行"
              + (" (含验证集)" if val_start is not None else ""), flush=True)

        seen = 0
        for batch in pf.iter_batches(batch_size=READ_BATCH, columns=["text"]):
            texts = batch.column("text").to_pylist()
            b = len(texts)
            if val_start is not None and seen + b > val_start:
                cut = max(val_start - seen, 0)
                train_texts, val_texts = texts[:cut], texts[cut:]
            else:
                train_texts, val_texts = texts, []
            for arr in tok_pack(tok, train_texts):
                writer.write(arr)
            if val_texts:
                for arr in tok_pack(tok, val_texts):
                    val_f.write(arr.tobytes())
                    val_total += len(arr)
            seen += b
        if val_f is not None:
            val_f.close()
            print(f"验证集完成: {val_total:,} tokens -> {args.out_dir}/val.bin")
    writer.close()


if __name__ == "__main__":
    main()
