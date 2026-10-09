#!/bin/bash
# 预训练守护: 1) 进程被杀(别人抢卡 OOM)后自动重选卡续训
#             2) 空闲卡变多时自动换更多卡跑 (检查点按绝对 token 记账, 换卡数是安全的)
# 用法: setsid nohup bash scripts/watch_pretrain.sh >/dev/null 2>&1 &
cd /root/Test/mini-r1
# 单实例: 重复启动守护脚本会互相抢卡/互杀 (flock 持有到进程退出)
exec 9>logs/.watch_pretrain.lock
flock -n 9 || { echo "已有守护脚本在跑, 退出"; exit 1; }

WLOG=logs/pretrain_watch.log
LOG=logs/pretrain.log
CKPT=ckpts/pretrain/latest.pt
MAX_GPUS=4                 # 最多用 4 张, 给同机器的其他人留 2 张
MIN_FREE_MB=10240          # 单卡空闲显存门槛, 必须与 run_pretrain.sh 里的一致
FRESH_SEC=600              # 换卡前等到一个 10 分钟内的检查点, 少丢重跑

free_gpus() {              # 输出空闲卡的 "0,3" 形式
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' -v m="$MIN_FREE_MB" '$3 - $2 > m && $4 < 80 {printf "%d,", $1}' | sed 's/,$//'
}
n_of() { [ -z "$1" ] && echo 0 || echo "$1" | tr ',' '\n' | wc -l; }
running() { pgrep -f "[t]rain_pretrain.py" > /dev/null; }
log() { echo "[watch $(date '+%m-%d %H:%M')] $*" >> $WLOG; }

cur=0
off=-1                     # 本次启动时日志的字节偏移 (-1 = 还没启动过): 只看这之后的内容判断是否训完
while true; do
  # ---- 没在跑: 要么已训完, 要么重启 ----
  if ! running; then
    if [ "$off" -ge 0 ] && tail -c "+$((off + 1))" "$LOG" 2>/dev/null | grep -q "训练完成"; then
      log "训练正常完成, 守护退出"; exit 0
    fi
    # 选卡交给 run_pretrain.sh 一家做 (两边各查一次会和别人抢卡的时机错开, 白折腾)
    off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    out=$(bash scripts/run_pretrain.sh $MAX_GPUS 2>&1) && rc=0 || rc=$?
    echo "$out" >> $WLOG
    if [ $rc -ne 0 ]; then
      # 空出来的卡很快会被别人抢走, 所以要盯紧一点
      log "当前没有可用的空闲卡, 1 分钟后重试"
      sleep 60; continue
    fi
    cur=$(echo "$out" | sed -n 's/.*(\([0-9]*\) 张).*/\1/p' | tail -1)
    cur=${cur:-1}
    log "启动训练: 用 $cur 张卡"
  fi

  # ---- 在跑: 60 秒巡一次; 进程一死就马上回到循环顶部重启 ----
  for _ in $(seq 1 5); do
    sleep 60
    running || break
  done

  # ---- 在跑: 空闲卡明显变多就换更多卡 (cur+N >= 1.5*cur, 小提升不折腾) ----
  if running && [ "$cur" -lt "$MAX_GPUS" ]; then
    F=$(free_gpus); N=$(n_of "$F")
    if [ "$N" -gt 0 ] && [ $(( (cur + N) * 2 )) -ge $(( cur * 3 )) ]; then
      use=$(( cur + N > MAX_GPUS ? MAX_GPUS : cur + N ))
      log "空闲卡变多 ($cur -> 计划 $use), 等一个较新的检查点后换卡"
      for _ in $(seq 1 60); do       # 最多等 30 分钟
        age=$(( $(date +%s) - $(stat -c %Y "$CKPT" 2>/dev/null || echo 0) ))
        [ "$age" -lt "$FRESH_SEC" ] && break
        running || break
        sleep 30
      done
      if running; then
        log "重启训练以换用 $use 张卡 (当前进度已在检查点里)"
        pkill -f "[s]rc/train_pretrain.py"      # 只杀我们自己的训练进程
        sleep 20
        cur=0
      fi
    fi
  fi
done
