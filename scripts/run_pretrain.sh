#!/bin/bash
# 预训练启动脚本: 自动挑选空闲 GPU, 有检查点就自动续训
# 用法: bash run_pretrain.sh [n_gpus]   (n_gpus 默认 3, 上限见空闲卡数)
# 需在服务器上运行, 依赖 conda 环境 mini-r1
#
# 想提前收尾 (不用跑满 5B): 把检查点已有的 token 数 + 一段退火尾巴 (约 10%)
# 传给 --max_tokens 重启即可, 例如已经跑了 3.0B 就改成 --max_tokens 3300000000,
# WSD 调度会从当前进度立刻开始退火 —— 这样停下来的是个收敛好的模型, 而不是半成品。
# 想加练: 直接改成 --max_tokens 0 (跑满 9.9B) 继续跑, LR 会回到恒定段, 不用重新热身。
set -euo pipefail

PROJ=/root/Test/mini-r1
N_GPUS="${1:-3}"
CKPT=$PROJ/ckpts/pretrain/latest.pt
RESUME=""
if [ -f "$CKPT" ]; then
  # 有检查点一律续训: 共享机器上任务会被反复打断, 误从零开始会白跑几十小时。
  # (要重训请先手动删除 latest.pt)
  RESUME="--resume"
else
  echo "无检查点, 从头开始训练"
fi

# ---- 挑选空闲 GPU: 空闲显存 > 13GB 且利用率 < 80% ----
# 门槛以「nvidia-smi 口径」为准, 不能信 PyTorch 的自报数:
#   probe_mem.py 报 max_memory_reserved 8.19 GiB, 驱动器实记 ~10.8 GiB;
#   2026-10-09 再实测一次, 训练自己的 max_memory_allocated 是 11.13 GiB,
#   而 nvidia-smi --query-compute-apps 里我们那个进程记的是 12,046 MiB。
# 旧的 11800 门槛就卡在这条线下面 —— 会选中 11.8~12.0G 空闲的卡然后一启动就 OOM,
# 正是 §7 第 1 条那个坑的翻版 (当时就是这么崩了一晚上)。故提到 13000。
# 门槛不能设太高: 本机常年饱和, 设狠了就永远选不出卡; 但"选不出卡"只是等 60 秒重试,
# 比 OOM 崩溃循环便宜得多, 所以宁可偏高。
FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 13000 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有空闲 GPU, 当前状态:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
IFS=',' read -ra GPUS <<< "$FREE"
USE=$(printf '%s\n' "${GPUS[@]:0:$N_GPUS}" | paste -sd, -)
NGPU=$(echo $USE | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$USE
# 共享卡上显存碎片会让本可跑下的批次 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "使用 GPU: $USE ($NGPU 张)"

# 固定「全局批大小」≈131k tokens/步: 卡多了就把梯度累积降下来,
# 这样换卡数不改变学习率调度和数据消耗节奏, 只改变墙钟速度。
GA=$(( (64 + NGPU / 2) / NGPU ))
echo "grad_accum=$GA -> 每步 $((2048 * 1 * GA * NGPU)) tokens"

mkdir -p $PROJ/ckpts/pretrain $PROJ/logs
cd $PROJ
# 日志用追加模式: 每次重启的完整历史都留着 (守护脚本只在自己启动的那段里找「训练完成」)
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/torchrun --nproc_per_node=$NGPU \
  src/train_pretrain.py \
  --data_dir $PROJ/data --ckpt_dir $PROJ/ckpts/pretrain \
  --micro_batch 1 --grad_accum $GA --seq_len 2048 \
  --peak_lr 3e-4 --min_lr 3e-5 --warmup_steps 500 \
  --max_tokens 5000000000 \
  --val_every 200 --ckpt_every 200 $RESUME \
  >> $PROJ/logs/pretrain.log 2>&1 < /dev/null &
echo "预训练已启动, 日志: $PROJ/logs/pretrain.log"
