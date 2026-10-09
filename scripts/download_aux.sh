#!/bin/bash
# 下载 SFT / GRPO 所需的小数据集到本地 (绕过 datasets 的 tree API 重定向问题)
#  - openai/gsm8k (train 2.3MB + test 419KB, parquet)
#  - teknium/OpenHermes-2.5 (1.9GB json)
# 用法: bash download_aux.sh  (可重复执行, 已存在则跳过)
set -uo pipefail

PROJ=/root/Test/mini-r1
RAW=$PROJ/data/raw
mkdir -p $RAW/gsm8k $RAW/openhermes

GSM8K="https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main"
OH="https://hf-mirror.com/datasets/teknium/OpenHermes-2.5/resolve/main"

dl() {  # dl <url> <dest>
  if [ -f "$2" ] && [ $(stat -c%s "$2" 2>/dev/null || echo 0) -gt 1000 ]; then
    echo "跳过 (已存在): $2"
  else
    echo "下载: $2"
    curl -sL -C - --retry 5 --retry-delay 3 --connect-timeout 20 \
      -o "$2.part" "$1" && mv "$2.part" "$2" && echo "完成: $2"
  fi
}

dl "$GSM8K/train-00000-of-00001.parquet" $RAW/gsm8k/train.parquet
dl "$GSM8K/test-00000-of-00001.parquet"  $RAW/gsm8k/test.parquet
dl "$OH/openhermes2_5.json"              $RAW/openhermes/openhermes2_5.json

echo "=== 完成 ==="
ls -lh $RAW/gsm8k $RAW/openhermes
