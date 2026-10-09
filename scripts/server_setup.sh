#!/bin/bash
# Mini-R1 服务器环境搭建脚本（幂等，可重复执行）
# 安装位置: /root/Test/mini-r1  (可写盘, 203G 可用)
set -euo pipefail

BASE=/root/Test/mini-r1
MF=$BASE/miniforge
ENV_NAME=mini-r1

export HF_HOME=$BASE/.cache/huggingface
export PIP_CACHE_DIR=$BASE/.cache/pip
mkdir -p "$BASE" "$HF_HOME" "$PIP_CACHE_DIR"

echo "=== [1/4] 安装 miniforge ==="
if [ ! -x "$MF/bin/conda" ]; then
  curl -sL --connect-timeout 20 -o /tmp/mf.sh \
    https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh \
    || curl -sL -o /tmp/mf.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/mf.sh -b -p "$MF"
  rm -f /tmp/mf.sh
  # 清华 conda 镜像加速
  "$MF/bin/conda" config --set show_channel_urls yes
  "$MF/bin/conda" config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
fi
"$MF/bin/conda" --version

echo "=== [2/4] 创建 conda 环境 ==="
PIP="$MF/envs/$ENV_NAME/bin/pip"
PY="$MF/envs/$ENV_NAME/bin/python"
if [ ! -x "$PY" ]; then
  # 显式 -p 前缀: 系统 .condarc 的 envs_dirs 指向只读盘, -n 会落到错误位置
  CONDA_ENVS_PATH="$MF/envs" "$MF/bin/conda" create -p "$MF/envs/$ENV_NAME" -y python=3.10
fi
"$PY" --version

echo "=== [3/4] 安装 PyTorch (CUDA 12.4) ==="
# 先试清华镜像, 失败则用官方源 (driver 550 最高支持 cu124)
"$PIP" install --quiet torch==2.6.0 torchvision==0.21.0 \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124 \
  || "$PIP" install --quiet torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo "=== [4/4] 安装依赖 ==="
"$PIP" install --quiet \
  numpy tqdm datasets tokenizers wandb sentencepiece \
  transformers==4.51.3 accelerate

# flash-attn 预编译轮子 (可选加速, 失败不阻塞: 用 SDPA 兜底)
"$PIP" install --quiet flash-attn==2.7.4.post1 --no-build-isolation || \
  echo "WARN: flash-attn 安装失败, 将使用 PyTorch 原生 SDPA"

echo "=== 验证 ==="
"$PY" -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU{i}: {p.name} {p.total_memory/1e9:.1f}GB')
import transformers, datasets
print('transformers:', transformers.__version__, '| datasets:', datasets.__version__)
"
echo "=== SETUP DONE ==="
