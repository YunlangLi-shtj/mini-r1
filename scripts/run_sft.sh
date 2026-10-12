#!/bin/bash
# SFT 启动: 自动挑空闲 GPU, 在预训练检查点上做指令微调 (单卡)
# 用法: bash run_sft.sh [base_ckpt]   默认用 ckpts/pretrain/latest.pt
set -euo pipefail

PROJ=/root/Test/mini-r1
BASE="${1:-$PROJ/ckpts/pretrain/latest.pt}"
[ -f "$BASE" ] || { echo "找不到预训练检查点: $BASE"; exit 1; }

# 门槛 12000 MiB —— 这是 SFT 自己的实测值, 不要照搬预训练的 13000:
#   SFT (mb1+grad_ckpt) PyTorch 自报 9.81 GB, 按驱动器口径约 10.6G ≈ 10,850 MiB。
#   预训练那边是 11.13 G / 12,046 MiB, 比 SFT 重, 所以它的 13000 对 SFT 偏严 ——
#   会把 12.4G 空闲的卡白白挡在门外 (2026-10-12 GPU 4 就是这样被我干等了半小时)。
#   原来写的 10240 又比实测需求低, 两头都不对。
# 选卡改成「等」而不是「退出」: 2026-10-12 服务器要装盘关机, 我们必须在自己占的卡
# 释放出来的那一瞬间抢回去 —— 退出去等人工重跑, 这几分钟的空窗就会被别人占了。
WAIT_MAX=1800   # 最多等 30 分钟; 超过就退出, 由人决定
waited=0
FREE=""
# 写成 if 而不是 `[ -n "$FREE" ] && break`: 后者其实也能跑 —— 实测 bash 的 set -e
# 对 && 列表中非最后一条命令是豁免的 (bash -c 'set -e; x=""; [ -n "$x" ] && echo; echo 活' 会打印"活"),
# 所以 watch_pretrain.sh 里同样的写法一直没问题。这里用 if 只是不想让读代码的人
# 还要在脑子里跑一遍这条豁免规则 —— 这段逻辑承担着"关机前抢回自己被释放的卡"的任务,
# 写清楚比写短重要。
while [ -z "$FREE" ]; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$3 - $2 > 12000 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
  if [ -z "$FREE" ]; then
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      echo "等了 ${WAIT_MAX}s 仍没有空闲 GPU:"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
      exit 1
    fi
    if [ "$waited" -eq 0 ]; then echo "暂无空闲卡, 每 60s 重试 (最多 ${WAIT_MAX}s)..."; fi
    sleep 60; waited=$(( waited + 60 ))
  fi
done
export CUDA_VISIBLE_DEVICES="${FREE%%,*}"     # 只用第一张空闲卡
echo "使用 GPU: $CUDA_VISIBLE_DEVICES (等待 ${waited}s)"

mkdir -p $PROJ/ckpts/sft $PROJ/logs
cd $PROJ
# 日志用追加: 每次重启留一条分隔线, 「哪次从第几步续起」在日志里查得到。
# 预训练那边用追加是为了守护脚本按字节偏移判断是否完成; 这边是为了留下续训的证据。
echo "===== [$(date '+%m-%d %H:%M')] 启动 SFT (GPU ${FREE%%,*}) =====" >> $PROJ/logs/sft.log
# micro_batch 用实测过的档位, 别凭感觉调 (2026-10-12 连炸三次换来的):
#   mb1 + grad_ckpt -> 峰值 9.81 GB, 12.4G 空闲的共享卡就能跑  ← 当前选择
#   mb2 + grad_ckpt -> 要 ~12.2 GB, 得等 ≥13G 的干净卡, 但快一倍 (3.4h vs 6.7h)
#   mb8 (原默认)    -> 首次前向就 OOM, 已废弃
# 全局批大小固定 32 条/优化器步 (mb1 × ga32 或 mb2 × ga16), 换档位不改训练动力学。
# ckpt_every=100: 100 步 ≈ 7 分钟, 关机/被抢占最多丢 7 分钟。
# 4.9G 的检查点写盘约半分钟, 约占 7%, 换这个粒度划算 (默认的 200 会丢 14 分钟)。
# ★ 续训只认 sft_latest.pt, 且依赖 seed+micro_batch+grad_accum 完全一致:
#   换 mb/ga 会让样本顺序对不上, 「跳过前 N 个 micro-batch」就跳错了数据。
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/python -u src/train_sft.py \
  --base_ckpt "$BASE" \
  --tokenizer $PROJ/tokenizer/tokenizer.json \
  --data_file $PROJ/data/raw/openhermes/openhermes2_5.json \
  --out_dir $PROJ/ckpts/sft \
  --max_samples 200000 --max_len 2048 \
  --micro_batch 1 --grad_accum 32 --grad_ckpt 1 \
  --lr 2e-5 --epochs 1 --warmup_steps 100 \
  --ckpt_every 100 \
  >> $PROJ/logs/sft.log 2>&1 < /dev/null &
echo "SFT 已启动, 日志: $PROJ/logs/sft.log"
echo "  续训检查点: $PROJ/ckpts/sft/sft_latest.pt  最终产物: $PROJ/ckpts/sft/sft.pt"
