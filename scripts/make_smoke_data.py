"""生成冒烟测试用小数据: 随机 token 分片 (验证训练代码机制, 不追求数据质量)

用法: python make_smoke_data.py [out_dir]
输出: {out_dir}/train/shard_00000.bin (500 万随机 token), {out_dir}/val.bin (2 万 token)
"""
import os
import sys

import numpy as np

VOCAB = 152064
TRAIN_TOKENS = 5_000_000
VAL_TOKENS = 20_000


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/root/Test/mini-r1/data_smoke"
    os.makedirs(f"{out}/train", exist_ok=True)
    rng = np.random.default_rng(0)
    rng.integers(0, VOCAB, TRAIN_TOKENS, dtype=np.uint32).tofile(f"{out}/train/shard_00000.bin")
    rng.integers(0, VOCAB, VAL_TOKENS, dtype=np.uint32).tofile(f"{out}/val.bin")
    print(f"冒烟数据完成: {out} (train {TRAIN_TOKENS:,} tokens + val {VAL_TOKENS:,})")


if __name__ == "__main__":
    main()
