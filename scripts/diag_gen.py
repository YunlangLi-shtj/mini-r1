#!/usr/bin/env python3
"""生成退化诊断: 模型写完答案之后, 是不是在自说自话地把正确答案埋掉?

为什么要单独量这个 (2026-10-14):
  GSM8K-CoT 的训练目标长度实测中位 **83 token** (p90 146, p99 226), 而评测和 GRPO
  rollout 都给 **max_new 384** —— 78% 的预算在训练分布之外, 那段没有老师, 全靠编。
  更麻烦的是 `extract_answer` 取的是**最后一个**非空盒子 (这是对的: 训练数据的最后
  一个盒子就是答案), 于是"先写出正确答案、再自己往下编"的采样会被判错。

  这个脚本量两件事:
    ① 自然停下来的比例 (出了 <|im_end|>) —— 低就说明模型没学会收尾;
    ② 第一个盒子对 vs 最后一个盒子对 —— 差值就是"被自己复读埋掉"的准确率。

用法:
  CUDA_VISIBLE_DEVICES=2 python -u scripts/diag_gen.py <ckpt> [n_questions] [n_samples]
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
from tokenizers import Tokenizer

from model import MiniR1Model, ModelConfig
from train_grpo import load_gsm8k, extract_answer, normalize, is_placeholder

IM_END = 5
PROJ = os.environ.get("MINIR1_PROJ", "/root/Test/mini-r1")

ckpt = sys.argv[1]
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer.from_file(f"{PROJ}/tokenizer/tokenizer.json")
_, test, _ = load_gsm8k(f"{PROJ}/tokenizer/tokenizer.json", n_eval=256,
                        gsm8k_dir=f"{PROJ}/data/raw/gsm8k")
test = test[:nq]

model = MiniR1Model(ModelConfig()).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
model.eval()

# 只取盒子内容, 保留**出现顺序** (extract_answer 内部取 m[-1], 这里要拿到全部)
BOX = re.compile(r"\\boxed\{([^{}]*)\}")


def boxes(t):
    return [x.strip() for x in BOX.findall(t) if not is_placeholder(x)]


n_tot = n_eos = n_hit_cap = 0
lens = []
first_ok = last_ok = 0
fmt = 0
n_multi_box = 0          # 写了不止一个盒子的采样数
for ex in test:
    ids = torch.tensor(ex["ids"], device=device).unsqueeze(0).repeat(ns, 1)
    outs = model.generate(ids, 384, temperature=1.0, eos_id=IM_END)
    gen = outs[:, len(ex["ids"]):]
    for j in range(ns):
        toks = gen[j].tolist()
        # 截到 EOS 为止 (generate 是整批都出 EOS 才停, 单条可能后面还有内容)
        eos_pos = toks.index(IM_END) if IM_END in toks else None
        hit = eos_pos is not None
        eff = toks[:eos_pos] if hit else toks
        n_tot += 1
        n_eos += hit
        n_hit_cap += (not hit)
        lens.append(len(eff))
        t = tok.decode(eff)
        bs = boxes(t)
        fmt += ("\\boxed{" in t)
        if len(bs) > 1:
            n_multi_box += 1
        g = normalize(ex["gold"])
        if bs:
            first_ok += (normalize(bs[0]) == g)
            last_ok += (normalize(bs[-1]) == g)
        else:
            # 没盒子时 extract_answer 会退回最后一个数字 —— 那也算一种"最后答案"
            last_ok += (normalize(extract_answer(t)) == g)

lens.sort()
print(f"\n=== 生成退化诊断: {ckpt} ===")
print(f"采样 {n_tot} 条 ({nq} 题 x {ns})")
print(f"自然停 (出 <|im_end|>): {n_eos}/{n_tot} = {n_eos/n_tot:.1%}")
print(f"跑满 384 上限        : {n_hit_cap}/{n_tot} = {n_hit_cap/n_tot:.1%}")
print(f"有效生成长度: 中位 {lens[len(lens)//2]}, p90 {lens[int(len(lens)*0.9)]}, "
      f"p99 {lens[int(len(lens)*0.99)]}, max {lens[-1]}")
print(f"格式合规 (含盒子)    : {fmt}/{n_tot} = {fmt/n_tot:.1%}")
print(f"写了 >1 个盒子的采样 : {n_multi_box}/{n_tot} = {n_multi_box/n_tot:.1%}")
print(f"★ 第一个盒子对       : {first_ok}/{n_tot} = {first_ok/n_tot:.2%}")
print(f"★ 最后一个盒子对     : {last_ok}/{n_tot} = {last_ok/n_tot:.2%}")
if first_ok > last_ok:
    print(f"  ⇒ 有 {(first_ok-last_ok)} 条是**先答对、后被自己的复读埋掉**的 "
          f"({(first_ok-last_ok)/n_tot:.1%} 的绝对差)")
else:
    print("  ⇒ 取最后一个盒子的口径没有损失准确率")
