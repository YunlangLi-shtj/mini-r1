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
| **KV-Cache** | **手写增量解码缓存**：只存未复制的 KV（GQA 省一半显存）、右下角对齐的因果掩码、RoPE 位置偏移。实测 rollout **4.78×**（49.9h → 10.4h） | 推理优化、SDPA 语义、数值等价性测试 |
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
  model.py             # 从零 Transformer (RMSNorm/RoPE/SwiGLU/GQA) + KV-Cache + 分块交叉熵
  tokenizer_train.py   # BPE 分词器训练
  data_pretrain.py     # 数据下载与分词分片
  train_pretrain.py    # 预训练 (bf16 + 梯度检查点 + WSD + 断点续训)
  train_sft.py         # 指令微调 (ChatML, 只对 assistant 段算 loss)
  train_grpo.py        # GRPO 强化学习 (含规则奖励函数 + 裁判自检)
  eval_gsm8k.py        # pass@1 / maj@k 评测
  chat.py              # 交互式 demo
scripts/
  server_setup.sh      # 环境一键搭建
  download_data.sh     # FineWeb-Edu 下载 (curl + 断点续传 + flock 防并发)
  run_pretrain.sh      # 选空闲卡 + 拼参数 + 启动 (有检查点自动续训)
  watch_pretrain.sh    # 守护: 训练被杀自动重选卡续训
  watch_grpo.sh        # 守护: GRPO 被杀自动续训 (每 5 步一个检查点)
  status.sh            # 状态速查 (按进程自动识别阶段, 进度/吞吐/剩余时间/显存)
  run_sft.sh  run_grpo.sh  run_eval.sh
  probe_sft.sh         # SFT 显存实测探针 (找安全档位用)
  bench_kv_cache.py    # KV-Cache 加速比实测 (自带正确性自检 + 显存守门)
  test_resume_accounting.py   # 断点续训记账的 CPU 冒烟测试
  test_sft_resume.sh   # SFT 续训回归测试 (自动判定通过/不通过)
  test_kv_cache.py     # KV-Cache 数值等价性回归测试 (8 项, 纯 CPU)
docs/                  # ★ 从零开始的教程, 见下
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

# 6. GRPO (有检查点会自动续训; 守护脚本在被抢占后自动拉起)
bash scripts/run_grpo.sh
setsid nohup bash scripts/watch_grpo.sh >/dev/null 2>&1 &

# 7. 评测 (SFT vs GRPO 对比)
bash scripts/run_eval.sh
```

### 推理加速：手写 KV-Cache

`generate()` 每生成一个 token 都要重算整条序列的注意力（O(L³) 总量）。
加上增量解码缓存后降到 O(L²)，**实测 4.78×**（同一张 4090、真实 407M 模型、fp32）：

| | 每组 rollout (G=8 × 512 tok) | 吞吐 | 300 步 GRPO |
|---|---|---|---|
| 朴素自回归 | 24.96 s | 123 tok/s | 49.9 h |
| **+ KV-Cache** | **5.22 s** | **589 tok/s** | **10.4 h** |

三个实现要点（其中两个**不报错但算错**）：
① 缓存只存 **未复制**的 KV（GQA 的 8 个 KV 头而非 16 个头），显存省一半；
② SDPA 的 `is_causal=True` 是**左上角对齐**的 —— 增量步（q_len=1）会只看见第 0 个 key，
   必须改用右下角对齐的显式掩码；
③ RoPE 的位置要接着缓存长度排（`cos[past_len:total]`），写成 `cos[:T]` 形状照样对得上。

正确性由 [`scripts/test_kv_cache.py`](scripts/test_kv_cache.py) 的 **8 项数值等价性检查**保证
（纯 CPU、10 秒、不抢卡），改任何掩码/RoPE 代码后必跑。

## 📖 从零读懂这个项目（初学者教程）

[`docs/`](docs/) 是一套**面向初学者的完整教程** —— 假设你没接触过大模型训练、没用过 GPU 服务器，
从「这个词是什么意思」讲到「每一行代码为什么这么写」。

**[👉 从这里开始：docs/README.md](docs/README.md)**（先选一条阅读路线，不必按顺序读）

| 文档 | 讲什么 |
|---|---|
| [01-项目全貌与术语](docs/01-项目全貌与术语.md) | 项目在干什么 + 完整名词表（token / loss / BPE / GQA / pass@1…） |
| [02-服务器与SSH](docs/02-服务器与SSH.md) | 从零 SSH 上远程机器、传文件、看日志、让任务断线后继续跑 |
| [03-环境搭建](docs/03-环境搭建.md) | 裸机 → 能跑训练，逐段拆解 `server_setup.sh` 的每个决定 |
| [04-代码精读](docs/04-代码精读.md) | **逐文件、按依赖顺序**拆解 `src/` 与 `scripts/`，含设计取舍与踩坑 |
| [05-数据管线](docs/05-数据管线.md) | 数据从哪来、`.bin` 分片格式、怎么自己解码检查 |
| [06-训练与评测](docs/06-训练与评测.md) | 三阶段怎么跑、日志每个字段什么意思、检查点里存了什么 |
| [07-GitHub与工程实践](docs/07-GitHub与工程实践.md) | 什么被 gitignore 了为什么、怎么不把密码传上 GitHub |
| [08-排错FAQ](docs/08-排错FAQ.md) | 常见故障「现象 → 原因 → 处理」 |

> 详细的踩坑记录、设计取舍、以及接手项目要看的东西，都在 [AGENTS.md](AGENTS.md)。

## 📊 实验结果

### 三段式训练

| 阶段 | 数据 | 规模 | 最终指标 | 墙钟 |
|---|---|---|---|---|
| 预训练 | FineWeb-Edu | 4.999B tokens / 38,141 步 | val_loss **3.1296** | 46 h（零重启） |
| SFT | OpenHermes-2.5（20 万条） | 6,240 步 / 1 epoch | loss 2.24 → **1.4328** | 6.6 h |
| 数学热身 SFT | GSM8K-CoT 7,473 条 | 6 epoch | GSM8K pass@8 **10.9%**（见下） | 1.5 h |
| GRPO | GSM8K 训练集 | 300 步 / G=8 | 见下方消融 | ~12 h |

预训练 loss 曲线：`ckpts/pretrain/train_log.jsonl`（每 10 步一行 JSON，含 `val_loss`）。

### GSM8K 评测（全量 1319 题 × 8 采样，`temperature=1.0`）

> ⚠️ 同一个词在本项目里有**两种口径**，对比时必须对齐，否则会得出相反的排序：
> `pass@1`（评测脚本）= 第 1 条采样的正确率；`pass@1`（训练日志）= 贪心解码。
> 下面的 `pass@1` 一律指前者。

| 检查点 | 说明 | pass@1 | pass@8 | 逐样本正确率 |
|---|---|---|---|---|
| `ckpts/sft/sft.pt` | OpenHermes SFT | 0.78% | 6.25% | 0.93% |
| `ckpts/sft_math6/sft.pt` | + GSM8K-CoT 热身 6 epoch | **2.34%** | **15.6%** | 2.25% |

8 条采样里至少一条答对：**6.25% → 15.6%**（配对 McNemar p=0.065，256 题子集上未达 0.05；
点估计约 2.5 倍）。这条提升里**混着两个改动**：数学热身，和「奖励读到了 EOS 之后的幻觉 token」
的修复（见故障复盘）。两者没有分开测量 —— 见下面「已知未测」。

**GRPO 消融（全量 1319 题 × 8 采样，配对检验）**：两条臂都从热过身的 `sft_math6` 出发、
各 300 步、各占一张 4090，唯一差别是格式奖励权重（1.0 vs 0）。

| 检查点 | pass@1 | maj@8 | pass@8 | 逐样本正确率 | 组内格式一致性 |
|---|---|---|---|---|---|
| `sft_math6`（基线，无 GRPO） | 2.00% | 1.92% | 12.81% | 1.86% | 0.525 |
| `grpo`（主臂，格式分 1.0） | **2.77%** | **2.85%** | **14.63%** | **2.11%** | **0.202** |
| `grpo_lowfmt`（消融臂，格式分 0） | 1.77% | 2.15% | 11.30% | 1.63% | 0.467 |

三条配对检验（McNemar 精确检验 + 按题 bootstrap，同一批题）：

| 对比 | pass@8 | McNemar p | 逐样本 Δ (95% CI) |
|---|---|---|---|
| 主臂 vs 基线 | 12.81% → 14.63% | 0.131（不显著） | +0.26% [−0.09, +0.60] |
| 消融臂 vs 基线 | 12.81% → 11.30% | 0.196（不显著） | −0.23% [−0.56, +0.10] |
| **主臂 vs 消融臂** | **14.63% → 11.30%** | **0.0054** ✅ | **−0.48% [−0.83, −0.14]** |

**结论分三句，别混着说：**

1. **格式奖励有可测的作用**（臂对臂 p=0.005，净 44 题）—— 但作用**不是**"教会模型写 `\boxed{}`"
   （基座本来就会 87%），而是**降低组内方差**：主臂把"这道题的 8 条采样写不写盒子"统一掉了
   （组内格式不一致度 0.525 → 0.202），消融臂没有（0.467）。GRPO 的优势 = 奖励的组内差异，
   格式在组内随机摆动时会往优势里灌进大量与数学无关的方差，把本就稀疏的数学信号淹掉。
2. **GRPO 本身在准确率上仍无可测提升**（主臂 +1.82% pass@8，p=0.131）—— 方向为正、
   机制清楚，但 1319 题只能排除 ±2.3% 以上的效应。**不能写「GRPO 带来提升」。**
3. **也没有变差**（消融臂 −1.5%，p=0.196）。**同样不能写「GRPO 让模型变差」。**

复现：`bash scripts/run_eval_pair.sh 0` → `python scripts/compare_evals.py A.json B.json`。

### 🔬 故障复盘：两次「指标全绿，模型在退化」

这个项目最值钱的部分可能不是最终数字，而是**两次不报错、不崩、但结论全错的坑**。

**① 奖励恒为 0 —— 裁判坏了，不是模型弱。** 数学奖励拿模型抽出的 `"18"` 去和 GSM8K 的
`answer` 字段（整段解题过程 `"...#### 18"`）比对，**永远不相等**：优势全 0 → loss 恒 0 →
`opt.step()` 什么都不做，跑完 4 步权重与基座的最大差是 **0.000e+00**。日志上只显示
`r 0.0000`，极容易被读成「模型太弱」。修法是加 `gold_answer()` 取 `####` 之后的部分，
并加一条自检：**把标准答案原文当模型输出喂进去必须得满分**，不过自检不许开跑。

**② 格式奖励被 hack —— 模型把提示词里的占位符抄了出来。** 提示词写的是
`put your final answer within \boxed{...}`，而格式奖励只检查「输出里有没有 `\boxed{` 这个
字符串」。于是最省力的拿分方式就是**把提示词里那个 `\boxed{...}` 原样抄一遍**：

| | 采样里最后一个盒子内容是字面量 `...` |
|---|---|
| SFT 基座 | 96 / 2048 = 4.7% |
| GRPO 300 步后 | **1344 / 2048 = 66%**（放大 14 倍） |

训练日志上一切都在"变好"（`r` 逼近格式分上限 0.1、format_ratio → 1.00），而**准确率没有
任何可测变化**（贪心 0→4 题、采样 2→0 题，256 题下 σ≈1.4 题，两个方向都说明不了）。
这条负对照完整保留在 `ckpts/grpo_hacked_v1/`。修法两处：格式奖励要求盒子里**真的有内容
且不是 `...`**，提示词也不再给现成占位符。**规则奖励里用到的字符串，绝不能同时出现在
提示词里。**

**③ 配对检验的基线被代码版本污染（方法论）。** 曾用两份评测报告做 McNemar 配对检验，
得出「EOS 截断把 pass@8 从 10.94% 抬到 15.62%」。但复现实验发现：**同一检查点、同一
seed、同一命令跑三次，输出逐字节相同**（sha256 完全一致）；而那次对比的两份报告
**是两版不同代码生成的** —— 配对检验衡量的其实是"两版代码"，不是那个修复。
**结论：做配对检验前先确认两次评测之间 `src/` 没动过。**

### 已知未测 / 诚实的边界

- **EOS 修复的单独效应量未知**：它的机制是明确的（`generate()` 是整批都出 EOS 才返回，
  先停下的序列会被继续推到 `max_new`，奖励从幻觉里抠答案），但干净的前后对照已经拿不到
  （旧代码版本没留）。所以上表那条 6.25% → 15.6% 只能报**合计**，不拆。
- 256 题子集上三个配对检验（热身、1→6 epoch、EOS）**全都 p>0.05**，方向一致但样本不够；
  要下硬结论得跑全量 1319（不一致对从 ~36 涨到 ~185，同样比例下 p 掉到 1e-5 量级）。
- **本项目的 GRPO 是冷启动极限的实例**：热身把 pass@8 抬到 10~15%，也就是每 24 组 rollout
  里只有 3~4 组带数学信号 —— 这正是「RLVR 不创造新能力、只锐化已有能力」的量化体现。

> 完整的踩坑记录（22 条，含显存/守护脚本/断点续训/容器 PID 命名空间等运维坑）见
> [AGENTS.md](AGENTS.md) §7；实验的逐步决策与待办见 §8。

## 🔭 未来工作

- [x] ~~KV-Cache 推理加速~~ —— 已完成，实测 **4.78×**（[bench_kv_cache.py](scripts/bench_kv_cache.py)）
- [ ] 批量推理 / vLLM 部署 rollout（进一步压缩采样时间）
- [ ] 奖励模型 (PRM) 替代规则奖励
- [ ] 多轮 GRPO 迭代 (iterative RL)
- [ ] 中英混合数据
- [ ] 消融: β、G、温度、update epochs 对收敛的影响
