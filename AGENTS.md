# AGENTS.md —— Mini-R1 项目交接说明

> 面向「接手这个项目的人」和「AI 编程助手」：**现在做到哪了 → 怎么查状态 → 怎么跑下一步 → 为什么这么写**。
> 项目本身的介绍（亮点、架构、GRPO 公式推导）在 [README.md](README.md)。

---

## 0. 一句话

从零手写一个 ~407M 的 Llama 风格模型，完整走完 **预训练(FineWeb-Edu) → SFT(OpenHermes) → GRPO(GSM8K)** 三段式，
目标是复现 DeepSeek-R1 那条「用强化学习把推理能力逼出来」的路线，并用消融证明 GRPO 带来提升。

---

## 1. 现在到哪一步了

_最后更新: 2026-10-09 23:00_

| 阶段 | 状态 | 产物 / 说明 |
|---|---|---|
| 环境搭建 | ✅ 完成 | conda 环境 + torch 2.6.0/cu124，见 §3 |
| 代码编写 | ✅ 完成 | `src/` 全部模块，见 §4 |
| 分词器 | ✅ 完成 | byte-level BPE，vocab 152,064（与 Qwen2 同规模） |
| 数据管线 | ✅ 完成 | 100 个分片，**9.903B train + 0.010B val tokens**，66 GB |
| **预训练** | 🔄 **进行中** | 跑到 step 1601 / 38146（0.210B / 5B tokens），loss 12.15 → 4.29 |
| SFT | ⏸ 未开始 | 数据已就绪，脚本已就绪 |
| GRPO | ⏸ 未开始 | 数据已就绪，**但需先加 KV-Cache**（见 §8） |
| 评测 / 消融 | ⏸ 未开始 | — |
| README 实验数据 | ⏸ 未开始 | 等评测结果 |

**当前训练的实时快照**（用 §2 的命令可以自己看）：

```
step   1601/38146 | 0.210B tok | loss 4.3017 | lr 3.00e-04 | 15,409 tok/s | 11.13G
val_loss @step 1601 = 4.35     检查点 latest.pt 4.6G     剩余 ≈ 3.2 天 (单卡)
```

> 多卡已修复并实测（见 §7 第 10 条）：**2 卡 1.77×，3 卡 2.00×**，上限设为 2 卡。
> `run_pretrain.sh` 会优先凑 2 张全空的卡，凑不齐自动退回单卡；守护脚本在有空卡时
> 会自动切过去。**不需要手动干预，也不要手动加卡。**

---

## 2. 怎么查看状态

### 2.1 最快的方式：一条命令看全部

本地终端里执行（`~/.server_cmd.sh` 是本地助手脚本，见 §3.2）：

```bash
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'
```

输出包含：进程在不在 → 训练进度与预计剩余时间 → 最新 loss / 验证集 loss → 检查点时间 → 6 张卡的占用情况。

### 2.2 想看细节时，各文件的用途

所有路径都在服务器的 `/root/Test/mini-r1/` 下。

| 想看什么 | 文件 / 命令 | 说明 |
|---|---|---|
| 训练输出（人看） | `logs/pretrain.log` | **追加模式**，历次重启的记录都在里面，往下翻能看到每次是从哪续起的 |
| 守护脚本干了啥 | `logs/pretrain_watch.log` | 每次「启动训练 / 没卡了 1 分钟后重试 / 换卡重启」都有时间戳 |
| 结构化指标（画图用） | `ckpts/pretrain/train_log.jsonl` | 每 10 步一行 JSON：`step / tokens / loss / lr / tok_per_s`，验证集是 `val_loss` |
| 检查点 | `ckpts/pretrain/latest.pt` | 4.6 GB，含 model + optimizer + **绝对 token 数**，每 200 步原子写一次 |
| 数据管线日志 | `logs/data_pretrain.log` | 分词分片过程的记录 |

常用命令：

```bash
# 只看最近几行训练输出
~/.server_cmd.sh 'tail -5 /root/Test/mini-r1/logs/pretrain.log'

# 画 loss 曲线（本地跑，用 jsonl）
~/.server_cmd.sh 'cat /root/Test/mini-r1/ckpts/pretrain/train_log.jsonl' > train_log.jsonl
python -c "
import json,matplotlib.pyplot as plt
r=[json.loads(l) for l in open('train_log.jsonl')]
t=[(x['tokens']/1e9,x['loss']) for x in r if 'loss' in x]
v=[(x['tokens']/1e9,x['val_loss']) for x in r if 'val_loss' in x]
plt.plot(*zip(*t),label='train'); plt.plot(*zip(*v),'o-',label='val')
plt.xlabel('tokens (B)'); plt.ylabel('loss'); plt.legend(); plt.savefig('loss.png',dpi=150)
"
```

### 2.3 训练停了怎么办

守护脚本（`scripts/watch_pretrain.sh`）会自动处理：进程被杀 → 重新选卡 → 从检查点续训。
**只有守护脚本自己也不在了才需要人工介入**，`status.sh` 会明确提示这种情况，并给出拉起命令。

---

## 3. 环境与服务器

### 3.1 服务器

| 项 | 值 |
|---|---|
| 地址 | `10.176.54.32`，端口 `13811`，用户 `root` |
| 环境 | Docker 容器，Ubuntu 22.04，128 核 / 503 GB 内存 |
| GPU | **6 × RTX 4090 (24 GB)，多人共享，常年饱和** |
| 项目路径 | `/root/Test/mini-r1`（`/root/Test` 是唯一可写的大盘，203 GB） |
| conda | `/root/Test/mini-r1/miniforge/envs/mini-r1`（Python 3.10） |

### 3.2 密码与本地助手脚本

> ⚠️ **密码不写进仓库**。它在本地两个脚本里，这两个文件在仓库之外、永远不提交：

- `~/.server_cmd.sh '<远程命令>'` —— 在服务器上执行一条命令
- `~/.server_sync.sh <本地目录> <远程目录>` —— 把本地代码同步上去（**必须带两个参数**，漏掉会静默失败）

```bash
# 同步代码到服务器（改完本地代码后必做，否则服务器跑的还是旧代码）
bash ~/.server_sync.sh /Users/liyunlang/mini-r1 /root/Test/mini-r1
```

同步方式是 tar+ssh 覆盖写：把本地目录整个打包解到远程目录，**不会删除**远程多出来的文件。
所以 `data/`、`ckpts/`、`miniforge/` 这些只存在于服务器的大目录不受影响。

---

## 4. 代码结构

```
src/
  model.py             # 从零手写 Transformer：RMSNorm / RoPE / SwiGLU / GQA / tied embedding
                       #   含 chunked_cross_entropy（152k 词表的显存关键优化）
  tokenizer_train.py   # 训练 byte-level BPE（vocab 152064）
  data_pretrain.py     # FineWeb-Edu → 二进制分片（uint32 memmap）
  train_pretrain.py    # 预训练：DDP + bf16 + 梯度检查点 + WSD 调度 + 断点续训
  train_sft.py         # 指令微调：ChatML 格式，只对 assistant 部分算 loss
  train_grpo.py        # GRPO：组相对优势 + clip + k3 KL 估计 + 规则奖励（含奖励函数）
  eval_gsm8k.py        # pass@1 / maj@k 评测（奖励函数从 train_grpo 导入，避免两份实现漂移）
  chat.py              # 交互式 demo
scripts/
  status.sh            # ★ 状态速查（见 §2.1）
  run_pretrain.sh      # 选空闲卡 + 拼参数 + 启动预训练（有检查点自动 --resume）
  watch_pretrain.sh    # ★ 守护：进程被杀自动续训（现锁死单卡，见 §7 第 10 条）
  run_watch.sh         # 把守护脚本拉起来（setsid nohup，单实例保护）
  run_sft.sh / run_grpo.sh / run_eval.sh   # 后续阶段的启动脚本（同样自动选卡）
  bench_ddp.sh         # 多卡吞吐实测: 等够 N 张干净卡自动跑一次短测（见 §7 第 10 条）
  probe_mem.py         # 显存实测探针（跑一步完整训练，报峰值）
  test_resume_accounting.py   # 断点续训记账的 CPU 冒烟测试（改记账逻辑后必跑）
```

**改代码后的固定动作**：`bash ~/.server_sync.sh <本地> <远程>` → 跑一次 `test_resume_accounting.py` → 提交。

---

## 5. 怎么跑下一步

### 5.1 预训练（正在跑，一般不用动）

```bash
bash scripts/run_pretrain.sh 1           # 有检查点会自动续训；★ 卡数固定 1（见 §7 第 10 条）
bash scripts/run_watch.sh                # 拉起守护（单实例保护，重复启动会自己退出）
```

平时**不用手动跑**——守护脚本会自己拉起训练。只有 `status.sh` 提示「守护进程不在跑」时才需要 `run_watch.sh`。

目标是 **5B tokens / 38146 步**。想改预算只改 `run_pretrain.sh` 里的 `--max_tokens`：

- **想加练**：改成 `--max_tokens 0`（跑满 9.9B），LR 会回到恒定段，不用重新热身
- **想提前收尾**：改成「当前已训 token 数 + 约 10% 退火尾巴」，例如跑到 3.0B 时改成 `--max_tokens 3300000000`，
  WSD 调度会立刻从当前进度开始退火——**停下来的是个收敛好的模型，而不是半成品**

### 5.2 SFT

```bash
bash scripts/run_sft.sh
```

数据：`data/raw/openhermes/openhermes2_5.json`（OpenHermes-2.5，1.9 GB，取 20 万条）。
产物：`ckpts/sft/`。**依赖预训练检查点**，所以要等预训练跑完（或至少跑出可用的 base 模型）。

### 5.3 GRPO

```bash
bash scripts/run_grpo.sh
```

配置：G=8（每 prompt 采样 8 条）、β=0.04、clip ε=0.2、300 步、规则奖励（`\boxed{}` 答案匹配 + 格式分）。
数据：`data/raw/gsm8k/{train,test}.parquet`。产物：`ckpts/grpo/`。

> ⚠️ **跑之前先看 §8 第 1 条**：当前 `generate()` 没有 KV-Cache，GRPO 的 rollout 会非常慢。

### 5.4 评测（SFT vs GRPO 消融，这是简历上的核心结果）

```bash
bash scripts/run_eval.sh
```

指标：GSM8K 上的 pass@1 与 maj@8，两个模型各跑一次做对比。

---

## 6. 关键设计决策（面试会问「为什么」的部分）

| 决策 | 为什么 |
|---|---|
| **WSD 调度**（warmup → 恒定 LR → 末尾 15% 退火），而非一路余弦 | 余弦要求开跑前把总步数定死。共享机器上能跑多久、要不要加练都是未知数；WSD 下任意时刻停下都能补一段退火尾巴收敛，想延长也不用重新热身（MiniCPM / DeepSeek 同款） |
| **固定全局批 ≈131k tokens/步**，卡多了就降 grad_accum | 换卡数只改变墙钟速度，不改变 LR 调度节奏和数据消耗速度，训练结果可复现 |
| **断点续训按绝对 `tokens_seen` 记账**，不用 `step × 每步token` | 每步 token 数含卡数因子。守护脚本每次重启用几张卡是随机的，按 step 推算会让续训跳到错误的数据位置（漏读/重读几亿 token） |
| **检查点原子写**（`.tmp` + `os.replace`） | 被抢占时正好在覆盖 `latest.pt` 会留下损坏文件，`--resume` 再也起不来 |
| **只有 world>1 才上 DDP** | torchrun 单进程也会设 `RANK`，不能只看它是否存在；单卡上 DDP 的梯度桶白占 ~2.5 GB |
| **分块交叉熵（512 行/块）** | 152k 词表下 `log_softmax` 的 fp32 临时张量才是显存瓶颈（2048 行时要 1.2 GB），比模型主体还大。分块后显存降下来，数值完全等价 |
| **LR = 3e-4**（对齐 Pythia-410M） | 同为 0.4B 级别公开可查的配方。我们的批更小(131k vs 2M)，小批噪声小能吃更大 LR，但要连跑几天没人盯着，取稳不取快 |
| **GPT-2 式初始化 N(0, 0.02)** | 初始 loss 直接 = ln(152064) ≈ 11.93，避免前几百步的 loss 平台期 |

---

## 7. 踩过的坑（改数据/训练代码前请先读）

1. **显存门槛必须按 `nvidia-smi` 口径设，不能信 PyTorch 的自报数**
   `probe_mem.py` 报的 8.19 GiB 是 `max_memory_reserved`；`train_log` 里打印的 11.13 G 是 `max_memory_allocated`；
   **驱动器真正记的是 12,046 MiB**（`nvidia-smi --query-compute-apps=used_memory` 实测）。
   按 8.19 或 11.13 设门槛 → 进程被抢占后在"看起来够"的卡上一启动就 OOM，然后**永远起不来**。
   本机常年饱和，门槛也不能设太高，但"选不出卡"只是等 60 秒重试，比 OOM 崩溃循环便宜得多，所以**宁可偏高**。
   现为 **13000 MiB**，`run_pretrain.sh` 和 `watch_pretrain.sh` 里必须保持一致（status.sh 的标注也用它）。
   （11800 是 2026-10-09 改掉的旧值：它卡在 12,046 这条线下面，会选中装不下的卡。）

   顺带一提：`max_memory_allocated` 在配置完全相同的情况下历史上报过 8.19 / 9.61 / 11.13 三个值，原因未查明，
   所以**不要**拿它当门槛依据。真要看占用，用 `nvidia-smi --query-compute-apps=pid,used_memory`。

2. **步数口径要除以 `seq×mb×GA×world`**
   `grad_accum` 随卡数调整，所以 `seq×mb×GA` 本身带卡数因子。拿它当"全局步数"的除数会让 world=2 的日志步数变成 world=1 的两倍。
   （LR 调度两种口径下都自洽，属可读性问题；但排查 2 卡启动失败时确实被误导过。）

3. **`datasets.map` 处理 ragged list-of-lists / bytes 返回值不可靠**——会把 token 展平成多行。
   分词管线最终弃用 `datasets`，改 `pyarrow.iter_batches` + `tokenizers.encode_batch`（Rust 并行，2.4M tok/s）。

4. **HF 下载必须 curl 到本地再用 `load_dataset(data_files=本地路径)`**
   hf-mirror 的部分 tree API 会 302 回 huggingface.co（被墙），`datasets` 跟随重定向即崩。
   另外：两个下载实例同时 `-C -` 续传同一个 `.part` 会交错写入导致 parquet 损坏，必须 flock 防并发。

5. **守护脚本单实例保护不能用 flock**——flock 的 fd 会被 `sleep` 子进程继承，杀掉守护后残留的 `sleep` 仍占着锁，新守护再也起不来。
   现在用 pid 文件 + `/proc/PID/cmdline` 校验。

6. **日志必须用追加模式**：守护脚本靠"本次启动时的字节偏移"判断「训练完成」是本次还是上次，截断模式分不清。

7. **`pkill -f` 的括号技巧保护不了自己所在的 shell**：同一条命令里既要 pkill 又要启动同名脚本时，
   自己的命令行里就含那个字符串，会被自己匹配到。用变量拼路径绕开。

8. **容器 PID 命名空间**：`nvidia-smi --query-compute-apps=pid` 报的是宿主 pid，容器里 `/proc/<宿主pid>` 不存在，别以为进程死了。
   用 `pgrep` 找自己的进程。

9. **共享机器上不得杀别人的进程**——`pkill` 只能针对自己脚本的特征，且必须用 `[x]` 括号技巧避免自匹配。

10. **梯度累积时忘了 `model.no_sync()` → 多卡比单卡慢 4 倍**（当前最重要的一条）
    DDP 默认每次 `backward()` 就 all-reduce 一遍梯度。梯度累积 `grad_accum=64` 意味着
    **64 次 backward 里有 63 次是白同步的**，通信量直接乘 64。2026-10-09 实测：
    单卡 16,000 tok/s（9.6G）vs 双卡 **3,990 tok/s**（12.6G）——加了一张卡反而慢 4 倍。
    修法是累积期间用 `model.no_sync()` 包住 forward/backward，只在最后一次同步。
    更要命的是它的**次生灾害**：双卡每卡要 12.6G，这台常年饱和的机器上很难两张卡同时空出 11.8G，
    守护脚本的换卡逻辑于是反复"抢到卡 → OOM → 60 秒后重试"，20:36~22:12 一个半小时崩了 103 次，
    进度几乎没动。所以现在 `watch_pretrain.sh` 里 `MAX_GPUS=1` **锁死单卡**。
    想重新启用多卡：先确认 no_sync 生效（`train_pretrain.py` 里已有），**再实测吞吐**，
    确认真的变快才改回 2；同时把门槛提到 ~15G，否则照样 OOM。

    **2026-10-10 实测结果**（`scripts/bench_ddp.sh`，用的是全空的卡）：

    | 配置 | 吞吐 | 相对单卡 | 每卡效率 | 消耗 GPU·天 |
    |---|---|---|---|---|
    | 单卡 | 16,630 tok/s | 1.00× | 100% | 3.37 |
    | 2 卡 | 29,389 tok/s | **1.77×** | 88% | 3.81 |
    | 3 卡 | 33,260 tok/s | 2.00× | 67% | 5.05 |

    结论：**上限设 2 卡**。第 3 张卡只多 13% 吞吐却多占一整张卡，在共享机器上很不划算；
    多卡的额外开销来自每个优化器步都要 all-reduce 一遍 fp32 梯度（约 1.6 GB），
    而全局批固定 131k tokens/步，所以这个开销摊不薄。

    显存上还有两个坑：
    - **多卡每卡要 ~13.2–13.5 GiB，比单卡的 12.0 GiB 多 1.5 GiB**（DDP 梯度桶）。
      所以单卡门槛 13000 **只适用于单卡**，拿去选多卡的卡必然 OOM。
      多卡门槛 **15000**（近乎全空）。`run_pretrain.sh` 现在是两级门槛 + 凑不齐退回单卡。
    - 可省 ~1.6 GiB/卡 的优化还没做：`DDP(..., gradient_as_bucket_view=True)`
      让梯度直接是桶的视图，不再多持有一份 fp32 梯度。真要压门槛时值得加，但要先测。

    另外两个坑：
    - **同机跑第二个 torchrun 必须换端口**，默认 29500 被正在跑的训练占着，会 `EADDRINUSE`
      （bench_ddp.sh 用 29511）。
    - **"空闲卡"的两种含义要分清**：`memory.used` 低才是真空闲。别人的进程常常占着
      10 GiB 但 `utilization` 是 0%——利用率看着空闲，显存根本不够，选卡一定要按显存算。

11. **`wc -l` 数的是换行符个数，不是行数**——`printf '%s' "$x" | tr ',' '\n' | wc -l`
    在 $x 不含逗号时会返回 **0**（因为 printf 不补结尾换行）。
    2026-10-10 选卡函数就是这么把 `"5"`（一张卡）算成 0 张，再 `除以 NGPU` 得到
    `division by 0`，`set -e` 直接让启动脚本退出、训练没起来。
    用 `grep -c '[0-9]'` 代替 `wc -l`（grep 会处理没有结尾换行的最后一行）。
    教训：**纯函数要单独测**，别只在"正常输入"上跑一次就算过。

---

## 8. 还要做什么

按优先级：

1. **给 `generate()` 加 KV-Cache**（GRPO 之前必须做）
   现在是朴素自回归：每生成一个 token 都要对整条序列重算一遍 attention，复杂度 O(L²) 且无缓存复用。
   GRPO 每步要生成 24 prompt × 8 条 rollout，按当前实现预计要几十小时。
   加 KV-Cache 是数量级的加速，本身也是个不错的简历技术点。
2. **跑完预训练**（≈3.6 天）→ SFT → GRPO → 评测
3. **填 README 的实验数据**：预训练 loss 曲线、SFT vs GRPO 的 pass@1 / maj@8
4. **消融实验**（现有脚本支持）：G 的大小、β、clip ε、奖励设计、update epochs
5. 可选：`chat.py` 录一段交互 demo；模型/数据传 HuggingFace

**已知遗留**：`/workspace/envs/mini-r1`（154 MB）是早期误建的 conda 环境，确认无用后可以删。

---

## 9. 给 AI 助手的注意事项

- **先跑 `bash scripts/status.sh` 看现状**，再决定做什么。预训练可能在几天前就跑完了。
- 改 `train_pretrain.py` 的记账/调度逻辑后，**必须跑 `scripts/test_resume_accounting.py`**（纯 CPU，不抢卡）。
- 同步代码用 `~/.server_sync.sh <本地> <远程>`，**两个参数都要给**。
- 任何情况下**不要把服务器密码写进仓库**；`~/.server_cmd.sh`、`~/.server_sync.sh` 在仓库之外，不要提交、不要在输出里回显密码。
- 服务器上**不要杀其他用户的进程**。训练被抢占是常态，交给守护脚本处理。
- 对话历史见 §10。

---

## 10. 附：AI 助手会话记录在哪里

Claude Code 按「**启动时的目录**」分目录保存会话，所以记录在本地：

```
~/.claude/projects/<启动目录转义>/
  <session-id>.jsonl      # 完整对话记录（JSONL，一行一条，可直接 grep / 阅读）
  memory/MEMORY.md        # 跨会话记忆索引，每次新会话自动加载
  memory/*.md             # 每条记忆一个文件
```

本项目这段对话是在 **`/Users/liyunlang`（家目录）** 启动的，所以记录在：
`~/.claude/projects/-Users-liyunlang/c8535776-716b-49f7-8c45-cdbb526e715c.jsonl`

**这就是"打开 mini-r1 目录后看不到历史"的原因**：换成在 `~/mini-r1` 启动时，助手会去找
`~/.claude/projects/-Users-liyunlang-mini-r1/`，那个目录不存在。

想接着这段对话，从**家目录**启动并恢复：

```bash
cd ~ && claude --resume      # 在列表里选这段会话
cd ~ && claude --continue    # 直接续上家目录下最近的一段
```

本文件（`AGENTS.md`）的作用正是弥补这一点：**不管从哪里启动，项目上下文都在仓库里**。
