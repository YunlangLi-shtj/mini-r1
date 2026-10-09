#!/bin/bash
# 等全部 14 个 parquet 下载完成 -> 启动 data_pretrain 数据管线
# ssh 远程执行本脚本即可 (等待在服务器侧进行)
cd /root/Test/mini-r1
for i in $(seq 1 120); do
  n=$(ls data/raw/*.parquet 2>/dev/null | wc -l)
  [ "$n" -ge 14 ] && break
  sleep 10
done
echo "parquet 数量: $(ls data/raw/*.parquet 2>/dev/null | wc -l)/14"
bash scripts/launch.sh data_pretrain src/data_pretrain.py \
  --local_dir /root/Test/mini-r1/data/raw \
  --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json \
  --out_dir /root/Test/mini-r1/data
echo "PIPELINE_LAUNCHED"
