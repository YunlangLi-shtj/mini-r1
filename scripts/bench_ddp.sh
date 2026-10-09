#!/bin/bash
# 多卡 DDP 吞吐实测: 等到 N 张「干净卡」自动跑一次短测, 结果留在日志里
#
# 为什么需要它:
#   2026-10-09 实测多卡比单卡**慢 4 倍** (单卡 16,000 tok/s vs 双卡 3,990 tok/s),
#   根因是梯度累积期间漏了 model.no_sync() —— DDP 每积累一步就 all-reduce 一遍,
#   grad_accum=64 时通信量乘 64。修复已提交, 但**没验证过**。切多卡之前必须先量吞吐。
#   而本机常年饱和, 3 张干净卡很难同时凑齐, 所以让脚本自己等。
#
# 判定标准 (单卡基线 ~16.6k tok/s):
#   N 卡接近 16.6k × N  →  同步问题确实修好了, 可以切多卡
#   N 卡远低于它         →  还有别的瓶颈, 老老实实单卡
#
# 用法: setsid nohup bash scripts/bench_ddp.sh 3 >/dev/null 2>&1 &
#       结果看 logs/bench_ddp.log
cd /root/Test/mini-r1

N="${1:-3}"
LOG=logs/bench_ddp.log
PORT=29511      # 默认 29500 被正在跑的训练占着, 复用会 EADDRINUSE
NEED_MB=15000   # 多卡每卡实测要 ~13.2~13.5 GiB, 留 1.5G 余量。
                # ★ 不能沿用单卡的 13000: 那次 OOM 就是拿单卡门槛去选多卡的卡。
                #   (24,564 总量 - 15,000 = 9.5G, 即"基本没人占"的卡才算数)
IDLE_MAX=20     # 利用率也要低: 显存空但正在被别人算的卡会拖慢我们

log() { echo "[bench $(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

free_clean() {  # 输出够干净的空闲卡, "2,3,4" 形式
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits \
    | awk -F', ' -v m="$NEED_MB" -v u="$IDLE_MAX" \
        '$3 - $2 > m && $4 < u {printf "%d,", $1}' | sed 's/,$//'
}

log "等 $N 张空闲显存 > ${NEED_MB} MiB 且利用率 < ${IDLE_MAX}% 的卡..."
while true; do
  F=$(free_clean)
  C=$(printf '%s' "$F" | tr ',' '\n' | grep -c '[0-9]')
  [ "$C" -ge "$N" ] && break
  sleep 60
done
USE=$(printf '%s' "$F" | cut -d, -f1-"$N")
log "拿到卡 $USE, 开跑 $N 卡 12 步"

GA=$(( (64 + N / 2) / N ))      # 与 run_pretrain.sh 同一公式: 固定全局批 ≈131k tokens/步
log "grad_accum=$GA -> 每步 $((2048 * 1 * GA * N)) tokens"

rm -rf "ckpts/bench_$N"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$USE timeout 900 \
  /root/Test/mini-r1/miniforge/envs/mini-r1/bin/torchrun \
    --nproc_per_node="$N" --master_port="$PORT" \
    src/train_pretrain.py --data_dir data --ckpt_dir "ckpts/bench_$N" \
    --micro_batch 1 --grad_accum "$GA" --seq_len 2048 \
    --max_steps 12 --log_every 4 --val_every 100000 --ckpt_every 100000 \
    >> "$LOG" 2>&1
rc=$?

if grep -q "OutOfMemoryError" "$LOG"; then
  log "★ 结果: 又 OOM 了 —— 说明门槛还得再提, 或者要开 gradient_as_bucket_view 省 DDP 的桶"
elif [ $rc -eq 124 ]; then
  log "★ 结果: 900 秒超时 (慢得离谱), 多卡不可用"
else
  # 先存进变量再写日志: 边 grep 边往同一个文件 append 可能读到写了一半的内容
  readings=$(grep "^step" "$LOG" | tail -3)
  log "★ 结果: 完成 (rc=$rc), 读数如下 (step 5/9 是稳态):"
  printf '%s\n' "$readings" | sed 's/^/    /' >> "$LOG"
fi
rm -rf "ckpts/bench_$N"
log "结束。单卡基线 ~16,630 tok/s, 拿它乘以 $N 对照上面的读数。"
