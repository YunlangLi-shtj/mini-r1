#!/bin/bash
# 一组检查点的消融评测, 依次跑 (简历上的核心结果)
#
# 用法: bash run_eval_pair.sh [n_eval] [tag:ckpt ...]
#       n_eval: 题数, 默认 256; 传 0 = 全量 1319 (跑得久, 但 σ 小 2.3 倍)
#       不给 tag:ckpt 就用下面那组默认 (2026-10-14 这一轮的四条)
#   例: bash run_eval_pair.sh 256 grpo:/root/Test/mini-r1/ckpts/grpo/grpo.pt
#
# 结果: reports/gsm8k_<tag>.json + logs/eval_<tag>.log
#
# ★ 为什么每条都要跑 (而不是只跑 GRPO 那条): run_eval.sh 同时给 pass@1 和 maj@8。
#   pass@1 是 temperature=1.0 的第 1 条采样, maj@8 是 8 条投票 ——
#   才能区分「真的会做了」和「只是变确定了」。
#
# ★ 但 256 题的 σ ≈ 1.4 题, 0.0156 和 0.0078 之间什么都说明不了 (实测踩过, 见 §8 第 1 条)。
#   真要下结论要么跑全量 1319, 要么用 scripts/compare_evals.py 做**配对**检验 ——
#   同一批题、同一个 seed, 配对之后灵敏度高得多。
set -uo pipefail

PROJ=/root/Test/mini-r1
N="${1:-256}"
shift || true

mkdir -p $PROJ/reports $PROJ/logs
cd $PROJ

if [ $# -gt 0 ]; then
  PAIRS=("$@")
else
  # 默认四条: 原始基座 / 热身后的基座 / 主臂 / 消融臂。
  # 热身基座 (sft_math6) 才是 GRPO 真正的"之前" —— 拿 sft.pt 去比会把热身的效果
  # 也算进 GRPO 头上, 那是这一轮最容易写错的一处结论。
  PAIRS=(
    "sft:$PROJ/ckpts/sft/sft.pt"
    "sft_math6:$PROJ/ckpts/sft_math6/sft.pt"
    "grpo:$PROJ/ckpts/grpo/grpo.pt"
    "grpo_lowfmt:$PROJ/ckpts/grpo_lowfmt/grpo.pt"
  )
fi

echo "===== [$(date '+%m-%d %H:%M')] 消融评测开始 (n_eval=$N, ${#PAIRS[@]} 条) ====="
# 不用 set -e: 一条失败不影响其余 —— 全都拿到才能对比
for pair in "${PAIRS[@]}"; do
  TAG="${pair%%:*}"
  CKPT="${pair#*:}"
  if [ ! -f "$CKPT" ]; then
    echo "--- [$(date '+%H:%M')] $TAG 跳过: 找不到 $CKPT ---"
    continue
  fi
  echo "--- [$(date '+%H:%M')] $TAG: $CKPT ---"
  bash scripts/run_eval.sh "$CKPT" "$TAG" "$N"
  echo "--- [$(date '+%H:%M')] $TAG 结束 (rc=$?) ---"
done
echo "===== [$(date '+%m-%d %H:%M')] 消融评测完成 ====="
