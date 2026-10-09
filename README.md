# Mini-R1: 从零训练一个会推理的小模型

**「预训练 → SFT → GRPO 强化学习」全流程从零实现** —— 对标 DeepSeek-R1 的推理模型训练路线，
用**一张 RTX 4090** 训练一个 ~0.4B 的数学推理模型（全程单卡可复现）。

## ✨ 项目亮点

| 模块 | 内容 | 面试考点 |
|------|------|----------|
| 模型 | 从零手写 Llama 风格 Transformer（RMSNorm / RoPE / SwiGLU / GQA / Tied Embeddings），仅依赖 PyTorch | 注意力机制、位置编码、GQA 显存分析 |
| 分词器 | 从零训练 byte-level BPE（vocab 152K，与 Qwen2 同规模） | BPE 合并算法、subword 原理 |
| 预训练 | 5B tokens (FineWeb-Edu)，bf16 混合精度 + 梯度检查点 + **WSD 学习率调度** + 断点续训 | 训练稳定性、学习率调度、显存优化 |
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
FineWeb-Edu (10B tok 语料, 取 5B 训练)
        │ BPE (vocab 152K)
        ▼
   二进制分片 ──预训练──> base 模型 ──SFT (OpenHermes)──> SFT 模型 ──GRPO──> RL 模型
                                                                    (GSM8K, 规则奖励)
                                                          π_ref = SFT 权重, 冻结
```

### GRPO 算法（从零实现, [train_grpo.py](src/train_grpo.py)）

对每个 prompt q 采样 G 条回答 o₁...o_G，组内归一化优势（**无需价值网络**）：

```
Aᵢ = (rᵢ − mean(r)) / std(r)

L = −E[ min(ρᵢ·Aᵢ, clip(ρᵢ, 1−ε, 1+ε)·Aᵢ) ] + β · KL(π_θ ‖ π_ref)
ρᵢ = π_θ(oᵢ|q) / π_θ_old(oᵢ|q)          # 逐 token 概率比

KL 用 k3 无偏估计: exp(logπ_ref − logπ_θ) − (logπ_ref − logπ_θ) − 1
```

> k3 的方向：样本 x ~ p 时 `(q/p) − log(q/p) − 1` 是 `KL(p‖q)` 的无偏估计。这里 p=π_θ、q=π_ref，
> 估出来的是 **KL(π_θ‖π_ref)**（把当前策略拉回参考策略），别写成反的。

奖励函数：规则化（无奖励模型）—— 数学答案对错（`\boxed{}` / 末位数字提取，与 gold 归一化比对）+ 格式奖励。

## 📁 目录结构

```
src/
  model.py             # 从零 Transformer (RMSNorm/RoPE/SwiGLU/GQA) + 采样 + 分块交叉熵
  tokenizer_train.py   # BPE 分词器训练
  data_pretrain.py     # 数据下载与分词分片
  train_pretrain.py    # 预训练 (bf16 + 梯度检查点 + WSD + 断点续训)
  train_sft.py         # 指令微调 (ChatML, 只对 assistant 段算 loss)
  train_grpo.py        # GRPO 强化学习 (含规则奖励函数)
  eval_gsm8k.py        # pass@1 / maj@k 评测
  chat.py              # 交互式 demo
scripts/
  server_setup.sh      # 环境一键搭建
  run_pretrain.sh      # 选空闲卡 + 拼参数 + 启动 (有检查点自动续训)
  watch_pretrain.sh    # 守护: 训练被杀自动重选卡续训
  status.sh            # 状态速查 (进度/吞吐/剩余时间/显存)
  run_sft.sh  run_grpo.sh  run_eval.sh
  test_resume_accounting.py   # 断点续训记账的 CPU 冒烟测试
```

## 🚀 复现

```bash
# 1. 环境
bash scripts/server_setup.sh

# 2. 分词器 (2 亿字符语料, ~20min)
python src/tokenizer_train.py --out_dir $PROJ/tokenizer

# 3. 数据 (FineWeb-Edu sample-10BT, ~1-2h 视网速)
python src/data_pretrain.py --tokenizer $PROJ/tokenizer/tokenizer.json --out_dir $PROJ/data

# 4. 预训练 (5B tokens; 单卡约 3.5 天, 会按空闲显存自动选卡)
bash scripts/run_pretrain.sh 1
bash scripts/run_watch.sh          # 守护, 被杀自动续训 (共享机器上建议开)
bash scripts/status.sh             # 随时查进度

# 5. SFT
bash scripts/run_sft.sh

# 6. GRPO
bash scripts/run_grpo.sh

# 7. 评测 (SFT vs GRPO 对比)
bash scripts/run_eval.sh
```

> 详细的踩坑记录、设计取舍、以及接手项目要看的东西，都在 [AGENTS.md](AGENTS.md)。

## 📊 实验结果

> 训练完成后填入：预训练 loss 曲线、SFT vs GRPO 的 GSM8K pass@1 / maj@8、消融实验（G 大小、β、奖励设计）。

## 🔭 未来工作

- [ ] KV-Cache 推理加速 / vLLM 部署 rollout（GRPO 采样效率 10x）
- [ ] 奖励模型 (PRM) 替代规则奖励
- [ ] 多轮 GRPO 迭代 (iterative RL)
- [ ] 中英混合数据
- [ ] 消融: β、G、温度、update epochs 对收敛的影响
