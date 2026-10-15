#!/usr/bin/env python3
"""打印检查点的**原始生成文本** (不经过 extract_answer)。

排查「评测口径对不上」时, 只看抽出来的答案是不够的 —— 得看模型到底写了什么。
用法: python scripts/show_completions.py <ckpt> [n_questions] [n_samples]
"""
import os
import sys

# 脚本在 scripts/ 下, 但 model.py / train_grpo.py 在 src/ —— sys.path[0] 是脚本所在目录,
# 不是 cwd, 所以得手动把 src/ 加进来 (和 run_eval.sh 从 PROJ 根目录跑 src/xxx.py 不同)。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from tokenizers import Tokenizer

from model import MiniR1Model, ModelConfig
from train_grpo import load_gsm8k, decode_completion

IM_END = 5

ckpt_path = sys.argv[1]
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer.from_file("/root/Test/mini-r1/tokenizer/tokenizer.json")
_, test, _ = load_gsm8k("/root/Test/mini-r1/tokenizer/tokenizer.json", n_eval=256,
                        gsm8k_dir="/root/Test/mini-r1/data/raw/gsm8k")

model = MiniR1Model(ModelConfig()).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
model.eval()

print(f"=== {ckpt_path} ===")
n_box_placeholder = n_box = n_tot = 0
for ex in test[:nq]:
    ids = torch.tensor(ex["ids"], device=device).unsqueeze(0).repeat(ns, 1)
    outs = model.generate(ids, 384, temperature=1.0, eos_id=IM_END)
    print(f"\n--- gold={ex['gold'][-40:]!r} ---")
    for j in range(ns):
        t = decode_completion(tok, outs[j, len(ex["ids"]):])
        n_tot += 1
        if "\\boxed{...}" in t:
            n_box_placeholder += 1
        if "\\boxed{" in t:
            n_box += 1
        print(f"[{j}] {t[:300]!r}")

print(f"\n统计: {n_tot} 条里, 含 \\boxed{{ 的 {n_box} 条, "
      f"其中字面量 \\boxed{{...}} 的 {n_box_placeholder} 条")
