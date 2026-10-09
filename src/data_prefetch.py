"""
预取 FineWeb-Edu sample-10BT 到 HF 缓存 (流式遍历 = 逐分片下载落盘)

与分词器训练并行跑 (BPE 用 CPU, 下载用网络, 互不干扰)。
data_pretrain.py 之后再流式加载时直接命中缓存, 无需重新下载。

用法:
  python data_prefetch.py [--name sample-10BT]
"""
import argparse
import time

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.name, split="train", streaming=True)
    n = ds.info.splits["train"].num_examples
    print(f"数据集共 {n:,} 篇文档, 开始流式遍历下载...", flush=True)
    t0, done = time.time(), 0
    for i, _ in enumerate(ds):
        done = i + 1
        if done % 50000 == 0:
            rate = done / (time.time() - t0)
            print(f"  已遍历 {done:,} / {n:,} 篇 ({rate:,.0f} 篇/s)", flush=True)
    print(f"预取完成: {done:,} 篇, 用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
