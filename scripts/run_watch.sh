#!/bin/bash
# 后台启动预训练守护
cd /root/Test/mini-r1
mkdir -p logs
setsid nohup bash scripts/watch_pretrain.sh > /dev/null 2>&1 < /dev/null &
echo "WATCHER_LAUNCHED (日志: logs/pretrain_watch.log)"
