#!/bin/bash
# 预训练启动脚本: 自动挑选空闲 GPU, 支持恢复训练
# 用法: bash run_pretrain.sh [--resume] [n_gpus]   (n_gpus 默认 3)
# 需在服务器上运行, 依赖 conda 环境 mini-r1
set -euo pipefail

PROJ=/root/Test/mini-r1
N_GPUS="${2:-3}"
RESUME=""
[ "${1:-}" = "--resume" ] && RESUME="--resume"
# 已有检查点就自动续训 (共享机器上任务会被反复打断, 误从零开始会白跑几十小时)
if [ -z "$RESUME" ] && [ -f "$PROJ/ckpts/pretrain/latest.pt" ]; then
  echo "检测到已有 latest.pt, 自动改用 --resume (要重训请先手动删除该文件)"
  RESUME="--resume"
fi

# ---- 挑选空闲 GPU: 空闲显存 > 10GB 且利用率 < 80% (微批次 4 约需 7-9GB) ----
FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 10240 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有空闲 GPU, 当前状态:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
IFS=',' read -ra GPUS <<< "$FREE"
# 取前 N 张 (逗号分隔传给 CUDA_VISIBLE_DEVICES)
USE=$(printf '%s\n' "${GPUS[@]:0:$N_GPUS}" | paste -sd, -)
NGPU=$(echo $USE | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$USE
echo "使用 GPU: $USE ($NGPU 张)"

mkdir -p $PROJ/ckpts/pretrain $PROJ/logs
cd $PROJ
# micro_batch 4 + grad_accum 16 = 每步 131k tokens (共享机器上显存更稳)
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/torchrun --nproc_per_node=$NGPU \
  src/train_pretrain.py \
  --data_dir $PROJ/data --ckpt_dir $PROJ/ckpts/pretrain \
  --micro_batch 4 --grad_accum 16 --seq_len 2048 \
  --peak_lr 1e-3 --min_lr 1e-4 --warmup_steps 500 \
  --val_every 500 --ckpt_every 500 $RESUME \
  > $PROJ/logs/pretrain.log 2>&1 < /dev/null &
echo "预训练已启动, 日志: $PROJ/logs/pretrain.log"
