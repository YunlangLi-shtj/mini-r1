#!/bin/bash
# SFT 启动: 自动挑空闲 GPU, 在预训练检查点上做指令微调 (单卡)
# 用法: bash run_sft.sh [base_ckpt]   默认用 ckpts/pretrain/latest.pt
set -euo pipefail

PROJ=/root/Test/mini-r1
BASE="${1:-$PROJ/ckpts/pretrain/latest.pt}"
[ -f "$BASE" ] || { echo "找不到预训练检查点: $BASE"; exit 1; }

FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 10240 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有空闲 GPU:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${FREE%%,*}"     # 只用第一张空闲卡
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p $PROJ/ckpts/sft $PROJ/logs
cd $PROJ
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/python -u src/train_sft.py \
  --base_ckpt "$BASE" \
  --tokenizer $PROJ/tokenizer/tokenizer.json \
  --data_file $PROJ/data/raw/openhermes/openhermes2_5.json \
  --out_dir $PROJ/ckpts/sft \
  --max_samples 200000 --max_len 2048 \
  --micro_batch 8 --grad_accum 4 \
  --lr 2e-5 --epochs 1 --warmup_steps 100 \
  > $PROJ/logs/sft.log 2>&1 < /dev/null &
echo "SFT 已启动, 日志: $PROJ/logs/sft.log  检查点: $PROJ/ckpts/sft/sft.pt"
