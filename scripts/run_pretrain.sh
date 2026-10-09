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
N_GPUS="${1:-1}"
CKPT=$PROJ/ckpts/pretrain/latest.pt
RESUME=""
if [ -f "$CKPT" ]; then
  # 有检查点一律续训: 共享机器上任务会被反复打断, 误从零开始会白跑几十小时。
  # (要重训请先手动删除 latest.pt)
  RESUME="--resume"
else
  echo "无检查点, 从头开始训练"
fi

# ---- 挑选空闲 GPU: 两级门槛, 优先多卡, 凑不齐就退回单卡 ----
# 门槛以「nvidia-smi 口径」为准, 不能信 PyTorch 的自报数 (probe_mem.py 报 8.19 GiB,
# 训练自报 max_memory_allocated 11.13 G, 而驱动器实记 12,046 MiB)。
#
# ★ 单卡和多卡的门槛不一样, 必须分开:
#     单卡每卡 12,046 MiB
#     多卡每卡 ~13.2~13.5 GiB —— DDP 还要额外持有一份梯度桶, 多 1.5 GiB
#   2026-10-10 就是拿单卡的 13000 门槛去选 3 卡的卡, 一启动就 OOM。
# 利用率也分两档: 单卡能和别人共卡 (2026-10-09 实测被别人打到 83% 利用率时我们
# 仍有 15.1k tok/s, 基本不掉速), 但多卡的前提是"几乎没人用", 否则 all-reduce
# 会和别人的算力互相拖。
SINGLE_MB=13000            # 单卡门槛
SINGLE_UTIL=80
MULTI_MB=15000             # 多卡每卡门槛 (24,564 总量 - 15,000 = 9.5G, 即近乎全空)
MULTI_UTIL=20

scan() {                   # $1=显存门槛 $2=利用率上限 -> "0,3" 形式
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits \
    | awk -F', ' -v m="$1" -v u="$2" '$3 - $2 > m && $4 < u {printf "%d,", $1}' \
    | sed 's/,$//'
}
# 数 "0,3" 里有几张卡。必须用 grep -c 而不是 wc -l: printf 不带结尾换行,
# wc -l 数的是换行符个数, 单个卡号会数出 0 —— 2026-10-10 就是这么把 "5" 算成
# 0 张卡, 再除以 NGPU 得到 division by 0, 训练直接没起来。
n_of() { printf '%s' "$1" | tr ',' '\n' | grep -c '[0-9]'; }

USE=""
if [ "$N_GPUS" -gt 1 ]; then
  CLEAN=$(scan "$MULTI_MB" "$MULTI_UTIL")
  if [ "$(n_of "$CLEAN")" -ge "$N_GPUS" ]; then
    USE=$(printf '%s' "$CLEAN" | cut -d, -f1-"$N_GPUS")
    echo "多卡: 有 $N_GPUS 张全空的卡 (>${MULTI_MB} MiB)"
  fi
fi
if [ -z "$USE" ]; then
  ANY=$(scan "$SINGLE_MB" "$SINGLE_UTIL")
  if [ -z "$ANY" ]; then
    echo "没有空闲 GPU, 当前状态:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 1
  fi
  USE=$(printf '%s' "$ANY" | cut -d, -f1)
  if [ "$N_GPUS" -gt 1 ]; then
    echo "凑不齐 $N_GPUS 张全空的卡 (只找到 $(n_of "$CLEAN") 张), 退回单卡"
  fi
fi
NGPU=$(n_of "$USE")
if [ "$NGPU" -lt 1 ]; then          # 兜底: 选卡逻辑出问题时宁可退出让守护重试,
  echo "选卡结果异常 (USE='$USE'), 放弃本次启动"   # 也不要带着 NGPU=0 往下走
  exit 1
fi
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
