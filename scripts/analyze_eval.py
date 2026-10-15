#!/usr/bin/env python3
"""看一眼评测报告里模型到底答了什么 —— 排查「两个评测口径对不上」用。

用法: python scripts/analyze_eval.py reports/gsm8k_sft.json [N]

★ 最重要的一个数: **pass@8** (8 条里至少一条对) = 题目的比例。
  它就是 GRPO 的**信号密度** —— 组内 8 条奖励有差异 ⇔ 至少一条对 (格式分饱和时)。
  而 eval_gsm8k.py 不报这个数, 得从这里反算。

  ⚠️ 比对必须走 `normalize`, 不能裸 `str(a) == str(b)`: 模型写 "1,234"、gold 是
     "1234" 时裸比会判错, pass@8 会被系统性低估 (2026-10-14 改掉)。
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_grpo import normalize

path = sys.argv[1]
show = int(sys.argv[2]) if len(sys.argv) > 2 else 3
d = json.load(open(path))
res, det = d["result"], d["details"]

print(f"=== {path} ===")
print(f"  {res}")
ncorr = sum(1 for x in det if x["correct"])
print(f"  pass@1 (第 1 条采样) 正确 {ncorr}/{len(det)}")

tot = sum(len(x["answers"]) for x in det)
none_cnt = sum(1 for x in det for a in x["answers"] if a is None)
print(f"  抽不出答案(extract_answer 返回 None)的样本: {none_cnt}/{tot}")

# 唯一答案数: 8 条采样里抽出几种不同答案 —— 全一样说明模型很确定
uniq = [len(set(x["answers"])) for x in det]
print(f"  每题 8 条采样里不同答案的种数: 均值 {sum(uniq)/len(uniq):.2f}, "
      f"只有 1 种的有 {sum(1 for u in uniq if u == 1)}/{len(uniq)} 题")

# ★ pass@k (k = 每题采样数): 8 条里至少一条对
def hit(x):
    g = normalize(x["gold"])
    return any(normalize(a) == g for a in x["answers"])

n_any = sum(1 for x in det if hit(x))
k = len(det[0]["answers"]) if det else 0
print(f"  ★ pass@{k} (至少一条对) = {n_any}/{len(det)} = {n_any/len(det):.2%}"
      f"   ← GRPO 的信号密度看这个")
print(f"  逐样本正确率 = {sum(1 for x in det for a in x['answers'] if normalize(a) == normalize(x['gold']))}"
      f"/{tot} = {sum(1 for x in det for a in x['answers'] if normalize(a) == normalize(x['gold']))/tot:.2%}")
print(f"  第 1 条错但 8 条里有对的: {n_any - ncorr} 题")

print()
for x in det[:show]:
    print(f"  Q: {x['q']}")
    print(f"  gold={x['gold']!r}")
    for i, a in enumerate(x["answers"]):
        print(f"    [{i}] {a!r}")
    print()

# 最常见的答案长啥样
common = Counter(a for x in det for a in x["answers"])
print("最常见的抽取结果 top10:", common.most_common(10))
