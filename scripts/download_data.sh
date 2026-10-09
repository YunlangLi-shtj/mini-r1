#!/bin/bash
# 用 curl 从 hf-mirror 下载 sample-10BT 的 parquet 文件 (绕过 datasets 的 tree API 重定向问题)
# 支持断点续传 (-C -), 可重复执行; 用法: bash download_data.sh [起始索引] [数量]
set -uo pipefail

# 防并发: 两个实例同时 -C - 续传同一 .part 会交错写入导致文件损坏
exec 9>/tmp/.mini_r1_download.lock
flock -n 9 || { echo "已有下载实例在跑, 退出"; exit 1; }

PROJ=/root/Test/mini-r1
RAW=$PROJ/data/raw
REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
BASE="https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/$REV/sample/10BT"
N_FILES=14
START="${1:-0}"
COUNT="${2:-$N_FILES}"

mkdir -p "$RAW"
cd "$RAW"

# 4 路并行下载 (共享带宽, 过多并发反而慢)
for i in $(seq $START $((START + COUNT - 1))); do
  f=$(printf "%03d_00000.parquet" $i)
  [ $i -ge $N_FILES ] && break
  (
    if [ -f "$f" ] && [ $(stat -c%s "$f" 2>/dev/null || echo 0) -gt 100000 ]; then
      echo "跳过 (已存在): $f"
    else
      echo "下载 $f ..."
      # --max-time: 单次尝试上限 (卡死自动重来); --speed-time: 60s 均速 <1KB/s 即断
      curl -sL -C - --retry 10 --retry-delay 3 --connect-timeout 20 \
        --max-time 600 --speed-time 60 --speed-limit 1024 \
        -o "$f.part" "$BASE/$f" && mv "$f.part" "$f" && echo "完成: $f"
    fi
  ) &
  # 控制并发为 4
  if [ $(( (i - START + 1) % 4 )) -eq 0 ]; then wait; fi
done
wait
echo "=== 全部完成, 当前目录: ==="
ls -lh "$RAW" | head -20
