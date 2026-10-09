#!/bin/bash
# 预训练守护: 进程死亡(被他人抢卡 OOM 等)且未正常完成时, 自动重新选卡 + 断点续训
# 用法: bash scripts/watch_pretrain.sh   (本身用 setsid nohup 后台跑)
cd /root/Test/mini-r1
LOG=logs/pretrain.log
WLOG=logs/pretrain_watch.log
MAX_ROUNDS=30

for round in $(seq 1 $MAX_ROUNDS); do
  echo "[watch $(date '+%m-%d %H:%M')] 第 $round 次启动" >> $WLOG
  if [ "$round" -eq 1 ]; then
    bash scripts/run_pretrain.sh >> $WLOG 2>&1              # 首次全新开始
  else
    bash scripts/run_pretrain.sh --resume >> $WLOG 2>&1     # 之后断点续训
  fi
  if [ $? -ne 0 ]; then
    echo "[watch] 没有空闲 GPU, 5 分钟后重试" >> $WLOG
    sleep 300
    continue
  fi
  # 等训练进程结束 (最多 12 小时)
  for i in $(seq 1 720); do
    pgrep -f "[t]rain_pretrain.py" > /dev/null || break
    sleep 60
  done
  if grep -q "训练完成" "$LOG" 2>/dev/null; then
    echo "[watch] 训练正常完成, 守护退出" >> $WLOG
    exit 0
  fi
  echo "[watch $(date '+%m-%d %H:%M')] 训练进程退出(未完成), 60 秒后重启" >> $WLOG
  sleep 60
done
echo "[watch] 重启次数用尽, 退出" >> $WLOG
exit 1
