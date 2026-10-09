#!/bin/bash
# 数据管线断点续跑: 跳过已处理的前 5 个文件 (000-004, 已产出 41 个分片)
cd /root/Test/mini-r1
mkdir -p logs
setsid nohup ./miniforge/envs/mini-r1/bin/python -u src/data_pretrain.py \
  --local_dir /root/Test/mini-r1/data/raw \
  --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json \
  --out_dir /root/Test/mini-r1/data \
  --skip_files 5 \
  > logs/data_pretrain.log 2>&1 < /dev/null &
echo "PIPELINE_RESUME_LAUNCHED (日志: logs/data_pretrain.log)"
