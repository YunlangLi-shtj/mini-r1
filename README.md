# Mini-R1: 从零训练一个会推理的小模型

**「预训练 → SFT → GRPO 强化学习」全流程从零实现** —— 对标 DeepSeek-R1 的推理模型训练路线，
在 3×RTX 4090 上训练一个 ~0.4B 的数学推理模型。

## ✨ 项目亮点

| 模块 | 内容 | 面试考点 |
|------|------|----------|
| 模型 | 从零手写 Llama 风格 Transformer（RMSNorm / RoPE / SwiGLU / GQA / Tied Embeddings），仅依赖 PyTorch | 注意力机制、位置编码、GQA 显存分析 |
| 分词器 | 从零训练 byte-level BPE（vocab 152K，与 Qwen2 同规模） | BPE 合并算法、subword 原理 |
| 预训练 | ~10B tokens (FineWeb-Edu)，DDP 分布式 + bf16 混合精度 + 梯度检查点 + 余弦退火 | 分布式训练、缩放定律、训练稳定性 |
| SFT | ChatML 指令微调，仅对回复部分计算 loss（label masking） | 数据格式、loss 设计 |
| **GRPO** | **从零实现 GRPO 算法**：组内相对优势、clip 概率比、k3 KL 估计，GSM8K 规则奖励（答案正确性 + 格式） | RL 数学推导、advantage、KL 正则 |
| 评测 | pass@1 / maj@k 多数投票，SFT vs GRPO 对比消融 | 实验设计 |

## 🏗 架构 (~407M 参数)

```
d_model=1024 | 16 层 | 16 查询头 / 8 KV 头 (GQA) | head_dim=64
FFN 4096 (SwiGLU) | vocab 152,064 | max_seq 4096 | RoPE θ=10,000 | 无偏置无 dropout
```

参数量: 嵌入 155.7M + 16×15.7M 层 ≈ **407M** (bf16 存储 ~0.8GB)

## 🔄 训练流程

```
FineWeb-Edu (10B tok) ──BPE──> 二进制分片 ──预训练──> base 模型
                                                        │ SFT (OpenHermes)
                                                        ▼
                                                   SFT 模型 ──GRPO──> RL 模型
                                                  (π_ref 冻结)   (GSM8K, 规则奖励)
```

### GRPO 算法（从零实现, [train_grpo.py](src/train_grpo.py)）

对每个 prompt q 采样 G 条回答 o₁...o_G，组内归一化优势（**无需价值网络**）：

```
Aᵢ = (rᵢ − mean(r)) / std(r)

L = −E[ min(ρᵢ·Aᵢ, clip(ρᵢ, 1−ε, 1+ε)·Aᵢ) ] + β · KL(π_ref ‖ π_θ)
ρᵢ = π_θ(oᵢ|q) / π_θ_old(oᵢ|q)          # 逐 token 概率比

KL 用 k3 无偏估计: exp(logπ_ref − logπ_θ) − (logπ_ref − logπ_θ) − 1
```

奖励函数：规则化（无奖励模型）—— 数学答案对错（`\boxed{}` / 末位数字提取，与 gold 归一化比对）+ 格式奖励。

## 📁 目录结构

```
src/
  model.py            # 从零 Transformer + 采样
  tokenizer_train.py  # BPE 分词器训练
  data_pretrain.py    # 数据下载与分词分片
  train_pretrain.py   # DDP 预训练
  train_sft.py        # 指令微调
  train_grpo.py       # GRPO 强化学习 (含奖励函数)
  eval_gsm8k.py       # pass@1 / maj@k 评测
  chat.py             # 交互式 demo
scripts/
  server_setup.sh     # 服务器环境一键搭建
```

## 🚀 复现

```bash
# 1. 环境
bash scripts/server_setup.sh

# 2. 分词器 (2 亿字符语料, ~20min)
python src/tokenizer_train.py --out_dir /root/Test/mini-r1/tokenizer

# 3. 数据 (sample-10BT, ~1-2h 视网速)
python src/data_pretrain.py --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json \
    --out_dir /root/Test/mini-r1/data

# 4. 预训练 (3 卡, ~10B tokens, 约 1 天 GPU 时)
torchrun --nproc_per_node=3 src/train_pretrain.py \
    --data_dir /root/Test/mini-r1/data --ckpt_dir /root/Test/mini-r1/ckpts/pretrain

# 5. SFT
python src/train_sft.py --base_ckpt /root/Test/mini-r1/ckpts/pretrain/latest.pt \
    --out_dir /root/Test/mini-r1/ckpts/sft

# 6. GRPO
python src/train_grpo.py --base_ckpt /root/Test/mini-r1/ckpts/sft/sft.pt \
    --out_dir /root/Test/mini-r1/ckpts/grpo

# 7. 评测 (SFT vs GRPO 对比)
python src/eval_gsm8k.py --ckpt /root/Test/mini-r1/ckpts/sft/sft.pt --n_samples 8
python src/eval_gsm8k.py --ckpt /root/Test/mini-r1/ckpts/grpo/grpo.pt --n_samples 8
```

## 📊 实验结果

> 训练完成后填入：预训练 loss 曲线、SFT vs GRPO 的 GSM8K pass@1 / maj@8、消融实验（G 大小、β、奖励设计）。

## 🔭 未来工作

- [ ] KV-Cache 推理加速 / vLLM 部署 rollout（GRPO 采样效率 10x）
- [ ] 奖励模型 (PRM) 替代规则奖励
- [ ] 多轮 GRPO 迭代 (iterative RL)
- [ ] 中英混合数据
- [ ] 消融: β、G、温度、update epochs 对收敛的影响
