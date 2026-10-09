#!/bin/bash
# GRPO 启动: 自动挑空闲 GPU, 从 SFT 检查点做强化学习 (单卡, 策略+参考模型各一份)
# 用法: bash run_grpo.sh [sft_ckpt] [max_steps]
set -euo pipefail

PROJ=/root/Test/mini-r1
BASE="${1:-$PROJ/ckpts/sft/sft.pt}"
STEPS="${2:-300}"
[ -f "$BASE" ] || { echo "找不到 SFT 检查点: $BASE"; exit 1; }

FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 16000 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有空闲 GPU (GRPO 需 >16GB):"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${FREE%%,*}"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p $PROJ/ckpts/grpo $PROJ/logs
cd $PROJ
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/python -u src/train_grpo.py \
  --base_ckpt "$BASE" \
  --tokenizer $PROJ/tokenizer/tokenizer.json \
  --gsm8k_dir $PROJ/data/raw/gsm8k \
  --out_dir $PROJ/ckpts/grpo \
  --G 8 --rollout_batch 24 --max_new 384 \
  --beta 0.04 --clip_eps 0.2 --lr 5e-6 --max_steps $STEPS \
  > $PROJ/logs/grpo.log 2>&1 < /dev/null &
echo "GRPO 已启动, 日志: $PROJ/logs/grpo.log  检查点: $PROJ/ckpts/grpo/"
