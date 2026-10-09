#!/bin/bash
# 下载启动包装: ssh 远程执行本脚本, 立即返回 (内联 & 链会被 ssh 挂住)
cd /root/Test/mini-r1
mkdir -p logs
setsid nohup bash scripts/download_data.sh > logs/download.log 2>&1 < /dev/null &
echo "DOWNLOAD_LAUNCHED (日志: logs/download.log)"
