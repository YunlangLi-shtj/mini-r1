#!/bin/bash
# GSM8K 评测: pass@1 + maj@8, 对比任意检查点 (SFT vs GRPO 的核心消融)
# 用法: bash run_eval.sh <ckpt> <tag> [n_eval]
#   例: bash run_eval.sh /root/Test/mini-r1/ckpts/sft/sft.pt sft 256
set -euo pipefail

PROJ=/root/Test/mini-r1
CKPT="$1"
TAG="$2"
N="${3:-256}"
[ -f "$CKPT" ] || { echo "找不到检查点: $CKPT"; exit 1; }

FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 10240 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有空闲 GPU:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${FREE%%,*}"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"

mkdir -p $PROJ/reports $PROJ/logs
cd $PROJ
$PROJ/miniforge/envs/mini-r1/bin/python -u src/eval_gsm8k.py \
  --ckpt "$CKPT" \
  --tokenizer $PROJ/tokenizer/tokenizer.json \
  --gsm8k_dir $PROJ/data/raw/gsm8k \
  --n_eval $N --n_samples 8 --temperature 1.0 --max_new 384 \
  --out $PROJ/reports/gsm8k_$TAG.json \
  2>&1 | tee $PROJ/logs/eval_$TAG.log
