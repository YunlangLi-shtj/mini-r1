#!/bin/bash
# SFT 探针: 等到一张够空的卡, 跑 512 条样本, 把显存和 loss 留在日志里
#
# 为什么要有它: 2026-10-12 SFT 连续三次 OOM —— 但三次原因各不相同, 靠拍脑袋估显存
# 每次都估错, 所以改成让脚本自己等卡、自己量:
#   1) micro_batch 8 无 grad ckpt  -> 首次前向就 OOM (已分配 16.03 GiB)
#   2) micro_batch 4 + grad ckpt   -> 峰值爬到 24.35 GB, 反向时 OOM (比 1) 还费!)
#   3) micro_batch 1 + grad ckpt   -> 自己只用 10.72 GiB, 但那张卡被别人占了 11.21 GiB 挤爆
# 结论: 显存的主项不是 micro_batch, 估算模型不可靠, 只能实测。当前结论是 micro_batch 1 +
# grad_ckpt (≈12.4 GiB, 和预训练 12,046 MiB 同量级) 才是在共享机器上跑得起来的档位。
#
# 顺带验证两个 bug 的修复 (loss 必须从 2~4 起步, 不再是 12.9):
#   - labels 没左移一位 -> 模型被要求预测自己
#   - step 数 micro-batch 而 total_steps 数优化器步 -> LR 调度错乱
#
# 用法: setsid nohup bash scripts/probe_sft.sh [micro_batch] >/dev/null 2>&1 &
#       结果看 logs/probe_sft.log
#   micro_batch 默认 1 (实测 9.81 GB, 0.12s/micro ≈ 3.0k tok/s)
#   调大能显著提速 —— 每条样本平均才 ~358 token, 小批次的固定开销占大头:
#     mb4 那次是 0.22s/4条 ≈ 6.5k tok/s (但显存 >16GB, 共享卡上跑不了)
#   所以 mb2 值得一测: 预估 ~11GB (仍低于 13G 线), 可能省一半时间。
cd /root/Test/mini-r1
MB="${1:-1}"

LOG=logs/probe_sft.log
NEED_MB=12000   # 推算依据: 修复前实测「PyTorch 已分配 10.72 GiB + 还差 594 MiB」≈ 11.3 GiB,
                # 减去检查点副本那 1.6G (改为落 CPU 并 del) ≈ 9.7 GiB, 按驱动器口径加 ~1G 开销
                # ≈ 10.8 GiB。取 12000 MiB 留一点余量。
                # ★ 别设成 13500 —— 那会让 12.4G 空闲的卡白白空着干等 (2026-10-12 就等了一轮)。
                #   这里量的是「我们需要多少」, 门槛要贴着实测值, 不是越大越安全。
IDLE_MAX=20     # 显存空但正在被别人算的卡会拖慢我们, 也要躲

log() { echo "[probe $(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

log "等一张显存空闲 > ${NEED_MB} MiB 且利用率 < ${IDLE_MAX}% 的卡..."
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader,nounits \
        | awk -F', ' -v m="$NEED_MB" -v u="$IDLE_MAX" \
            '$3 - $2 > m && $4 < u {print $1; exit}')
  [ -n "$GPU" ] && break
  sleep 60
done
log "拿到 GPU $GPU, micro_batch=$MB"

# 固定「全局批大小」32 条/优化器步, 和预训练固定 131k tokens/步是一个道理:
# 改 micro_batch 只改墙钟速度, 不改训练动力学 (LR 调度、数据消耗节奏都不变)。
GA=$(( 32 / MB ))
rm -rf /tmp/sft_probe; mkdir -p /tmp/sft_probe
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  miniforge/envs/mini-r1/bin/python -u src/train_sft.py \
  --base_ckpt ckpts/pretrain/latest.pt --tokenizer tokenizer/tokenizer.json \
  --data_file data/raw/openhermes/openhermes2_5.json --out_dir /tmp/sft_probe \
  --max_samples 512 --max_len 2048 --micro_batch "$MB" --grad_accum "$GA" --grad_ckpt 1 \
  --log_every 4 > /tmp/sft_probe.log 2>&1
rc=$?

# 先存进变量再写日志: 边 grep 边往同一个文件 append 可能读到写了一半的内容
if grep -q "OutOfMemoryError" /tmp/sft_probe.log; then
  # 多取更长的尾部: 是我们自己撑爆的, 还是被别人挤掉的, 要从 "Process ... has ..." 看
  log "★ 结果: OOM (rc=$rc)"
else
  log "★ 结果: 跑完 (rc=$rc)"
fi
grep -E "^step |SFT 样本数|assistant token|Tried to allocate|Process [0-9]+ has" \
  /tmp/sft_probe.log | tail -14 | sed 's/^/    /' >> "$LOG"
log "结束 (micro_batch=$MB)。判定: loss 应从 2~4 起步 (原 bug 是 12.9); 峰值应 < 13 GB。"
