#!/usr/bin/env python3
"""贪心 (temperature=0) pass@1 —— 和 train_grpo.eval_pass1 走**完全同一条**路径。

存在的理由: 训练日志里的 pass@1 和 run_eval.sh (eval_gsm8k.py) 里的 pass@1
**不是同一个指标** —— 前者贪心、1 条; 后者 temperature=1.0、8 条里取第 1 条。
名字一样、口径不同, 摆在一起对比会得出相反的结论。这个脚本用来把口径对齐。

用法: python scripts/eval_greedy.py <ckpt> [n_eval]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from tokenizers import Tokenizer

from model import MiniR1Model, ModelConfig
from train_grpo import eval_pass1, load_gsm8k, math_reward, format_reward, extract_answer

IM_END = 5

ckpt_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
PROJ = "/root/Test/mini-r1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test, tok = load_gsm8k(f"{PROJ}/tokenizer/tokenizer.json", n_eval=n,
                          gsm8k_dir=f"{PROJ}/data/raw/gsm8k")

model = MiniR1Model(ModelConfig()).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
model.eval()

acc = eval_pass1(model, test, tok, max_new=384)

# 顺带统计: 多少条输出了字面量 \boxed{...} (奖励被 hack 的标志)
n_ph = n_box = 0
with torch.no_grad():
    for ex in test:
        ids = torch.tensor(ex["ids"], device=device).unsqueeze(0)
        out = model.generate(ids, 384, temperature=0.0, eos_id=IM_END)
        t = tok.decode(out[0, len(ex["ids"]):].tolist())
        if "\\boxed{" in t:
            n_box += 1
        if "\\boxed{...}" in t:
            n_ph += 1

print(f"\n=== {ckpt_path} (贪心, {n} 题) ===")
print(f"  pass@1 (贪心)          = {acc:.4f}  ({round(acc * n)}/{n})")
print(f"  输出里有 \\boxed{{      = {n_box}/{n}")
print(f"  字面量 \\boxed{{...}}   = {n_ph}/{n}   ← 把 prompt 里的占位符照抄出来")
