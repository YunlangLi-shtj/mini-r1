#!/bin/bash
# 预训练启动脚本: 自动挑选空闲 GPU (显存占用 < 6GB), 支持恢复训练
# 用法: bash run_pretrain.sh [--resume] [n_gpus]
# 需在服务器上运行, 依赖 conda 环境 mini-r1
set -euo pipefail

PROJ=/root/Test/mini-r1
PY=$PROJ/miniforge/envs/mini-r1/bin/python
N_GPUS="${2:-3}"
RESUME=""
[ "${1:-}" = "--resume" ] && RESUME="--resume"

# ---- 挑选空闲 GPU: 空闲显存 > 14GB 视为可用 (训练约需 8-10GB) ----
FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 14336 {printf "%d,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有空闲 GPU, 当前状态:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
IFS=',' read -ra GPUS <<< "$FREE"
# 取前 N 张 (逗号分隔传给 CUDA_VISIBLE_DEVICES)
USE=$(printf '%s\n' "${GPUS[@]:0:$N_GPUS}" | paste -sd, -)
export CUDA_VISIBLE_DEVICES=$USE
echo "使用 GPU: $USE ($(echo $USE | tr ',' '\n' | wc -l) 张)"

mkdir -p $PROJ/ckpts/pretrain $PROJ/logs
cd $PROJ
nohup $PROJ/miniforge/envs/mini-r1/bin/torchrun --nproc_per_node=$(echo $USE | tr ',' '\n' | wc -l) \
  src/train_pretrain.py \
  --data_dir $PROJ/data --ckpt_dir $PROJ/ckpts/pretrain \
  --micro_batch 8 --grad_accum 8 --seq_len 2048 \
  --peak_lr 1e-3 --min_lr 1e-4 --warmup_steps 500 \
  --val_every 500 --ckpt_every 2000 $RESUME \
  > $PROJ/logs/pretrain.log 2>&1 &
echo "预训练已启动 (PID $!), 日志: $PROJ/logs/pretrain.log"
