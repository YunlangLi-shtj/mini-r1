#!/bin/bash
# SFT 守护: 进程被杀 (共享机器上别人抢卡吃显存) 后自动从检查点续训。
#
# 和 watch_pretrain.sh 的区别: 这里不做「换更多卡」—— SFT 是单卡任务
# (mb1+grad_ckpt 峰值 9.81 GB, 换多卡要重调 grad_accum, 而续训依赖
#  seed+micro_batch+grad_accum 完全一致, 换了样本顺序就对不上)。
#
# 用法: setsid nohup bash scripts/watch_sft.sh >/dev/null 2>&1 &
# 停:   pkill -f "[w]atch_sft.sh"   (只杀守护, 训练进程不动)
cd /root/Test/mini-r1

# 单实例保护 (pid 文件; 不用 flock 的原因见 watch_pretrain.sh 的注释)
PIDF=logs/.watch_sft.pid
if [ -f "$PIDF" ]; then
  OLD=$(cat "$PIDF" 2>/dev/null)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q "watch_sft.sh"; then
    echo "已有 SFT 守护在跑 (pid $OLD), 退出"; exit 1
  fi
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

WLOG=logs/sft_watch.log
LOG=logs/sft.log
log() { echo "[watch $(date '+%m-%d %H:%M')] $*" >> $WLOG; }
running() { pgrep -f "[t]rain_sft.py" > /dev/null; }

# 已经跑完了就别守着: sft.pt 只在最后一步写, 见到它就说明 SFT 结束过
if [ -f ckpts/sft/sft.pt ]; then log "sft.pt 已存在, SFT 早就完成了, 守护不启动"; exit 0; fi

# 偏移取「守护启动那一刻」的日志长度, 不是 -1:
# 被守护的那个训练可能是守护启动之前就跑着的, 它完成时写下的 "SFT 完成"
# 也在偏移之后, 所以照样能认出来。取 -1 反而会把历史记录里的完成误判成本次。
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
log "SFT 守护启动 (pid $$), 日志偏移 $off"
while true; do
  if ! running; then
    # 完成了就别再拉起来 (SFT 日志是追加模式, 所以要按偏移区分是哪一次的"完成")
    if [ "$off" -ge 0 ] && tail -c "+$((off + 1))" "$LOG" 2>/dev/null | grep -q "SFT 完成"; then
      log "SFT 已正常完成, 守护退出"; exit 0
    fi
    off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    out=$(bash scripts/run_sft.sh 2>&1) && rc=0 || rc=$?
    echo "$out" >> $WLOG
    if [ $rc -ne 0 ]; then
      # 没卡 / 没检查点: 空出来的卡很快会被别人抢走, 盯紧一点
      log "启动失败 (rc=$rc), 1 分钟后重试"
      sleep 60; continue
    fi
    log "已拉起 SFT: $(echo "$out" | grep '使用 GPU' | tail -1)"
  fi
  # 在跑: 60 秒巡一次, 进程一死马上回到循环顶部重启
  for _ in $(seq 1 5); do
    sleep 60
    running || break
  done
done
