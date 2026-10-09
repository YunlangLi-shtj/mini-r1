"""
预训练数据管线: FineWeb-Edu (sample-10BT, ~100 亿 token) -> 自训 BPE 分词 -> 二进制分片

输出布局 (nanoGPT 风格, 训练时可直接 np.memmap):
  {out_dir}/train/shard_{i:05d}.bin    每片 1 亿 token, uint32
  {out_dir}/val.bin                    末尾 1 万篇文档作为验证集

用法:
  python data_pretrain.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
      --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json --out_dir /root/Test/mini-r1/data
"""
import argparse
import os
import struct
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

TOKENS_PER_SHARD = 100_000_000   # 每分片 1 亿 token (uint32, 400MB)
VAL_DOCS = 10_000                # 末尾文档数用作验证集
EOT = 2                          # </s>, 文档间分隔符
NUM_PROC = 8                     # tokenize 子进程数 (共享机器, 克制使用)


def tokenize_batch(batch, tokenizer_path):
    """对一批文档做 BPE 编码, 每篇末尾追加 </s>。datasets.map 的 batched 回调。"""
    tok = Tokenizer.from_file(tokenizer_path)
    out = tok.encode_batch(batch["text"])
    return {"ids": [e.ids + [EOT] for e in out]}


def consume(tok_ds, out_dir: str, is_val: bool, log_every: int = 5):
    """把 tokenize 后的批次流写入二进制分片, 返回写入的总 token 数。"""
    if is_val:
        path = f"{out_dir}/val.bin"
        f = open(path, "wb")
        total = 0
        for batch in tok_ds:
            for ids in batch["ids"]:
                f.write(struct.pack(f"{len(ids)}I", *ids))
                total += len(ids)
        f.close()
        print(f"验证集完成: {total:,} tokens -> {path}")
        return total

    buf = np.empty(TOKENS_PER_SHARD, dtype=np.uint32)
    shard_idx, pos, total = 0, 0, 0
    t0 = time.time()
    for batch in tok_ds:
        for ids in batch["ids"]:
            n = len(ids)
            if pos + n > TOKENS_PER_SHARD:
                buf[pos:].tofile(f"{out_dir}/train/shard_{shard_idx:05d}.bin")
                shard_idx += 1
                pos = 0
            buf[pos:pos + n] = ids
            pos += n
            total += n
            if total % (TOKENS_PER_SHARD // 4) < 256:
                rate = total / (time.time() - t0)
                print(f"  {total:,} tokens, {rate:,.0f} tok/s, shard={shard_idx}", flush=True)
    if pos:
        buf[:pos].tofile(f"{out_dir}/train/shard_{shard_idx:05d}.bin")
    print(f"训练集完成: {total:,} tokens, {shard_idx + 1} 个分片")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_docs", type=int, default=VAL_DOCS)
    args = ap.parse_args()

    os.makedirs(f"{args.out_dir}/train", exist_ok=True)
    ds = load_dataset(args.dataset, args.name, split="train", streaming=True)
    n = ds.info.splits["train"].num_examples
    print(f"数据集共 {n:,} 篇文档, 验证集取末尾 {args.val_docs} 篇")

    map_kwargs = dict(batched=True, batch_size=64, num_proc=NUM_PROC,
                      fn_kwargs={"tokenizer_path": args.tokenizer},
                      remove_columns=list(ds.features.keys()))

    # 末尾 val_docs 篇 -> 验证集, 其余 -> 训练分片
    consume(ds.skip(n - args.val_docs).map(tokenize_batch, **map_kwargs),
            args.out_dir, is_val=True)
    consume(ds.map(tokenize_batch, **map_kwargs), args.out_dir, is_val=False)


if __name__ == "__main__":
    main()
