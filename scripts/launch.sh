#!/bin/bash
# 统一后台任务启动器: 设置 HF 镜像环境变量 + setsid 脱离会话 + 日志落盘
# 用法 (在服务器上): bash scripts/launch.sh <日志名> <python脚本> [参数...]
PROJ=/root/Test/mini-r1
export HF_HOME=$PROJ/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com        # huggingface.co 国内不稳定, 走镜像
export HF_HUB_DOWNLOAD_TIMEOUT=60

NAME="$1"; shift
mkdir -p $PROJ/logs
cd $PROJ
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/python -u "$@" \
  > $PROJ/logs/$NAME.log 2>&1 < /dev/null &
echo "LAUNCHED: $NAME (日志: $PROJ/logs/$NAME.log)"
