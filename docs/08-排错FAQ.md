# 08 - 排错 FAQ

> 按「你看到了什么」组织。每个问题都给 **现象 → 原因 → 处理**。
> 前 5 条是出现频率最高的。

---

## 🔌 连接与网络

### Q1. `ssh: connect to host ... port 13811: Connection timed out`

**先判断是"服务器挂了"还是"你连不上"**：

```bash
# ① 公网正常吗
ping -c 2 baidu.com

# ② 服务器通吗
nc -z -v 10.176.54.32 13811

# ③ 你的 VPN 连着吗（macOS）
ifconfig | grep -A3 utun
```

| ① | ② | ③ | 结论 |
|---|---|---|---|
| ✅ | ❌ | 无 utun | **VPN 没连** —— 服务器是 `10.x` 内网地址，必须在校内网/VPN 里才能到 |
| ✅ | ❌ | 有 utun | VPN 连着但路由不对，重连 VPN |
| ❌ | ❌ | — | 你的网络有问题 |

> **⚠️ 最重要的一点：训练不受影响。**
> 训练是 `setsid nohup` 启动的，早就脱离了 SSH 会话。
> 你连不上只是"看不见"，它还在跑。

**怎么确认训练还在跑**：等网络恢复后看日志的修改时间：

```bash
~/.server_cmd.sh 'ls -l --time-style=full-iso /root/Test/mini-r1/logs/sft.log'
```

时间戳是几分钟前 → 还在跑。时间戳停在几小时前 → 真的停了。

---

### Q2. SSH 断开之后训练停了

**原因**：启动方式错了。必须让进程脱离终端会话：

```bash
# ❌ 会随 SSH 断开而死
ssh root@... 'python src/train_pretrain.py ...'

# ✅ 脱离会话
setsid nohup python -u src/train_pretrain.py ... > log.txt 2>&1 < /dev/null &
```

**处理**：从检查点重启即可（[run_pretrain.sh](../scripts/run_pretrain.sh)
会自动加 `--resume`，进度不会丢）。然后**把守护脚本拉起来**，
以后就不用手动管了：

```bash
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/run_watch.sh'
```

详见 [02-服务器与SSH](02-服务器与SSH.md#5-让任务在断线后继续跑关键)。

---

## 💾 显存与 OOM

### Q3. `torch.OutOfMemoryError: CUDA out of memory`

**这是本项目最常见的问题**，要先分清是哪种：

**第一步：看是自己撑爆的，还是被别人挤掉的。**

```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
  (GPU 0; 24.00 GiB total capacity; 21.31 GiB already allocated;
   ...
   Process 12345 has 18.42 GiB memory in use.     ← ★ 关键这一行
```

- **`Process NNN has XX GiB in use`**，且 NNN **不是我们的进程** →
  **被别人挤掉的**。这是共享机器的常态，等一等或换张卡。
  → 交给守护脚本处理，它会重选卡。
- **`already allocated` 很高，没有别的进程** → **是我们自己撑爆的**。
  → 说明档位选大了，见下。

**第二步：按阶段处理。**

| 阶段 | 当前安全档位 | 实测占用 | `run_*.sh` 里的门槛 |
|---|---|---|---|
| 预训练 | `mb1 × ga64`（单卡）/ `mb1 × ga32`（双卡） | 12,046 MiB | 13000 / 15000 MiB |
| SFT | `mb1 × ga32` + 梯度检查点 | ~10,850 MiB | 12000 MiB |
| GRPO | `G8 × 24 prompt × 384 token` | **14.0 GiB**（驱动器口径 17.3 GiB） | **19000 MiB** |

**第三步（GRPO 专属）：看 `峰值` 那一列。**
每步的日志末尾会打 `峰值 x.xG`（`torch.cuda.max_memory_allocated()`）：

```
step 25 | loss -0.0210 | r 0.0922 | 有信号组 19/24 | lr 4.91e-06 | 148s | 峰值 14.0G
```

**超过 21 G 就是要 OOM 了**，别等它崩 —— 见 Q3′，那次的代价是 78 次崩溃。
这个数字是**排障时唯一能看出趋势的东西**：只有一行 `OutOfMemoryError` 的话，
你分不清"本来就要满了"和"某一步突然涨上去"。

**不要凭感觉调 `micro_batch`** —— 本项目为了找 SFT 的安全档位，
连续 OOM 了三次，而且**原因每次都不一样**：

```
mb8 无 grad_ckpt   → 首次前向就 OOM（16.03 GiB）
mb4 + grad_ckpt    → 爬到 24.35 GB，反向时 OOM（比 mb8 还费！）
mb1 + grad_ckpt    → 自己只用 10.72 GiB，但那张卡被别人占了 11.21 GiB 挤爆
```

**结论：显存只能实测，估算不可靠。** 用探针：

```bash
~/.server_cmd.sh 'cd /root/Test/mini-r1 && setsid nohup bash scripts/probe_sft.sh 1 >/dev/null 2>&1 &'
# 结果看 logs/probe_sft.log
```

---

### Q3′. ★ 训练每 3 分钟崩一次，报错一模一样（连崩 78 次）

**现象**：守护脚本每隔几分钟把训练拉起来，跑一会儿又死，日志里全是同一条 OOM，
**连字节数都一个字不差**：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.52 GiB.
GPU 0 has a total capacity of 23.64 GiB of which 1.51 GiB is free.
Process 936198 has 22.12 GiB memory in use. Of the allocated memory
21.28 GiB is allocated by PyTorch, and 404.54 MiB is reserved by PyTorch but unallocated.
```

```bash
grep -c "OutOfMemoryError" logs/grpo.log     # 78 —— 先数一下崩了几次
```

**三个判断，按顺序做：**

**① 报错数字每次完全一样 → 确定性失败，别再等它自己好。**
被别人挤掉、抢卡、碎片，这些每次的字节数都不一样。**一字不差 = 代码的问题。**

**② `404.54 MiB reserved but unallocated` → 不是碎片。**
碎片长这样（Q13″ 那条）：`reserved 22.54 GiB / allocated 只有 2.24 GiB` ——
"占着坑但没放东西"。这里未分配的只有 404 MiB，说明那 21 GiB **全是有主的活张量**。
`expandable_segments:True` 早就开了，帮不上忙。

**③ 那 21 GiB 是谁 → 去代码里找没被释放的副本。**
这一例是两个 `torch.load(..., map_location=device)` 留下的
（`ckpt` 1.63 GB + `ck` 4.90 GB = 6.5 GB 纯垃圾）：
`load_state_dict` 对 `nn.Module` 是**复制**进参数的，原件只要还被变量引用着就不释放。
修法 `map_location="cpu"` + `del`；另一半（优化器状态）反而是"就地采用"，
所以别背哪个复制哪个采用，**记「加载完就 `del`」**。
代码细节见 [docs/04 ⑥(f)](04-代码精读.md)。

**④ 另一半是激活**：更新那一步要存 16 层激活（约 5 GB）。
给策略模型开梯度检查点，且必须带 `cache_passthrough=True`（原因见 docs/04 ①(c)）。

修完峰值 **21.28 GiB（还在涨）→ 14.0 GiB**（PyTorch 口径），一步 148 秒（原来 146 秒，几乎没变慢）。

**⑤ 最后回头量一遍门槛。** 修完实测驱动器口径是 **17,726 MiB = 17.3 GiB**，
比 PyTorch 自报的 14.0 GiB 多 3.7 GB —— 而 `run_grpo.sh` 原来的门槛只要 16000 MiB 空闲，
**最坏情况照样 OOM**。已提到 **19000**。
这次的教训有点扎心：**门槛不是被崩溃证明错的，是被「修好之后的实测数字」证明错的** ——
崩溃期间那句"16000 是保守值"听着很合理，其实从来没验证过。

**这次最贵的教训不在显存，在守护脚本：**
`watch_grpo.sh` 的职责是"进程没了就拉起来"，它**不问为什么死的**。
于是确定性崩溃被忠实地变成了崩溃**循环** —— 78 次、2.5 小时、一张卡全程被一个
必然失败的任务占着。**发现进程死了，先 `grep -c OutOfMemoryError` 再决定拉不拉。**

---

### Q4. 「明明有卡空着，为什么脚本说没有空闲 GPU」

**原因**：门槛设得不对。有三种可能：

**① 你用错了显存口径。** 见 [06-训练与评测 §5](06-训练与评测.md#51-三个数字别搞混)：

```bash
# PyTorch 自报（会低估）
max_memory_allocated = 9.81 G
# 驱动器真实记账（选卡要用这个）
nvidia-smi → 10,850 MiB
```

**②「空闲」有两种含义**：

```
GPU 4   10124 MiB / 24564 MiB   利用率 0%     ← 利用率是 0，但显存被占着
GPU 5    1024 MiB / 24564 MiB   利用率 95%    ← 利用率高，但显存基本空着
```

**判断能不能用要看显存（`memory.used`），不是看利用率。**
别人的进程常常占着 10 GB 但利用率 0%（在等数据加载），你上去照样 OOM。

**③ 门槛设得比实际需要高。** 本项目踩过：SFT 实际只要约 10.8 GB，
但脚本一开始写的门槛是 13000 MiB（照搬预训练的），
于是「12.4 GB 空闲」的卡被白白挡在门外，**干等了半小时**。
现在 SFT 用 12000。

**调试**：

```bash
~/.server_cmd.sh 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
```

---

## 🏃 训练进程

### Q5. 训练进程不见了 / 日志不更新

```bash
# ① 进程在不在
~/.server_cmd.sh 'pgrep -af "[t]rain_sft.py"'

# ② 守护在不在
~/.server_cmd.sh 'pgrep -af "[w]atch_sft.sh"'

# ③ 一条命令看全部（推荐）
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'
```

**处理矩阵**：

| 训练 | 守护 | 怎么办 |
|---|---|---|
| ✅ | ✅ | 正常，不用管 |
| ❌ | ✅ | 守护会在 60 秒内自动拉起。等着，或看 `logs/sft_watch.log` |
| ❌ | ❌ | **需要人工介入**。看日志最后几行找原因，然后 `bash scripts/run_sft.sh` + `bash scripts/watch_sft.sh` |
| ✅ | ❌ | 训练在跑但没人保护。拉起守护：`bash scripts/watch_sft.sh` |

**守护脚本说"已有守护在跑"但实际没有** —— pid 文件残留：

```bash
rm -f /root/Test/mini-r1/logs/.watch_sft.pid
```

（正常情况下 `trap ... EXIT` 会清理，只有 `kill -9` 才会残留。）

---

### Q6. `tail -f` 看着日志一直不动，是卡住了吗

**可能只是 Python 的输出缓冲。** 启动时漏了 `-u`：

```bash
python -u script.py          # -u = 不缓冲 stdout
```

**也可能真的在加载数据。** SFT 每次启动要解析 1.9 GB 的 json + 分词 20 万条，
**需要 3~4 分钟**，这段时间日志是空的，进程 CPU 占用很高。

**怎么区分**：看 CPU 和磁盘活动：

```bash
top -bn1 | head -20
iostat -x 1 3                # 磁盘在读写 → 在加载
```

> ⚠️ **这个"加载要 4 分钟"直接害过一个测试脚本**：
> 它用 `timeout 30` 想「跑 30 秒就杀掉」，但 `timeout` 是**从进程启动就计时**的，
> 所以在加载阶段就被杀了 —— 一个检查点都没留下。
> **要按日志内容判断**（"出现了某个检查点行"），不能按秒数。

---

### Q7. `pkill` 把别人的进程杀了 / 把自己杀了

**两条铁律**：

```bash
# ❌ 绝对不要
pkill -f python

# ✅ 只针对自己的脚本，且用 [x] 括号技巧
pkill -f "[t]rain_sft.py"
pkill -f "[w]atch_sft.sh"
```

**为什么要加方括号**：`pgrep`/`pkill` 会把**自己的命令行**也算进匹配范围。
如果写成 `pkill -f "train_sft.py"`，而这条命令本身是
`ssh ... 'pkill -f "train_sft.py"'`，那么**它自己的命令行里就含这个字符串**，
在某些情况下会自匹配。

`[t]rain_sft.py` 这个正则匹配的是 `train_sft.py`，
但**命令行里的字面文本是 `[t]rain_sft.py`**，不匹配自己。

**更进一步**：如果同一条命令里既要 `pkill` 又要启动同名脚本，
连括号技巧都不够用 —— 用变量拼路径绕开：

```bash
W=$(ls scripts/ | grep watch_sft)
pkill -f "[t]rain_sft.py"; bash scripts/$W
```

---

## 🔁 断点续训

### Q8. ★ 续训之后 loss **比不中断更低**

**这是最危险的一种故障**，因为「loss 更低」看起来是好消息。

**现象**：续训后同一 step 的 loss 比不中断跑的低不少
（实测：step 40 时 A=2.8216 vs B=2.3211，**低 18%**）。

**原因**：跳过逻辑的 off-by-one。

```python
step, micro = start_step, start_micro     # ❌ 错：micro 一开始就等于 start_micro
for batch in dl:
    if micro < start_micro:                # 条件当场为假 → 一条都不跳过
        micro += 1; continue
```

**优化器从第 32 步的状态接着，数据却从头读起** ——
等于把前 32 步见过的东西又学了一遍，所以 loss 偏低。

**修法**：`micro` 从 **0** 起（`step` 仍从 `start_step` 起，LR 要用）。

**验证**：跑 [test_sft_resume.sh](../scripts/test_sft_resume.sh)，
它有三段对比 + 自动判定（阈值 5%）。

> 📌 **教训**：这是"不报错的 bug"。
> **判据要反过来看** —— 续训后 loss 明显偏离（无论高低）都是可疑的，
> 而不是"更低就是更好"。

---

### Q9. 续训后 loss 跳变 / 对不上

除了 Q8，还有几个可能：

| 原因 | 怎么确认 |
|---|---|
| `seed` 变了 | 检查启动参数是否和上次一致 |
| `micro_batch` / `grad_accum` 变了 | 同上 —— 这三个任一变了，样本顺序就对不上 |
| 数据分片被改过 | `ls -l data/train/` 看时间戳 |
| 检查点损坏 | 见 Q10 |

> ⚠️ **硬约束**：SFT 的续训**只认 `seed + micro_batch + grad_accum` 完全一致的组合**。
> 要改档位，**必须先删掉 `ckpts/sft/sft_latest.pt`** 从头跑。

---

### Q10. `torch.load` 报错 / 检查点加载失败

**情况 A：`_pickle.UnpicklingError: Weights only load failed`**

PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`（安全变更，
防止恶意检查点执行任意代码）。本项目的检查点符合要求，
但如果你自己往里面加了自定义对象就会失败。

```python
# 确认它到底是不是纯数据
import torch
c = torch.load("ckpts/sft/sft_latest.pt", map_location="cpu")   # 能加载就是没问题
print(list(c.keys()))     # ['step', 'micro', 'model', 'opt']
```

**情况 B：`unexpected EOF` / `PytorchStreamReader failed`**

**检查点文件损坏**。原因通常是：写检查点时进程被杀，
留下半个文件。

**这就是为什么 [train_pretrain.py](../src/train_pretrain.py#L238) 和
[train_sft.py](../src/train_sft.py#L182) 都用原子写**：

```python
torch.save({...}, tmp)
os.replace(tmp, path)     # 原子操作：要么旧文件完整，要么新文件完整
```

**处理**：损坏的 `latest.pt` 只能删掉，从更早的检查点重来。
如果连更早的都没有，那就只能从头训练 —— 所以
**检查点频率别设太稀疏**（预训练 200 步、SFT 100 步）。

---

## 📉 训练指标

### Q11. loss 是 NaN / inf

**原因**（按概率排序）：

1. **学习率太大** —— 最可能。
2. 数据里有异常样本（空序列、id 越界）。
3. 梯度爆炸（但本项目有 `clip_grad_norm_(1.0)` 兜底）。
4. bf16 溢出（bf16 范围比 fp16 大得多，不太可能）。

**处理**：

```bash
# 看 loss 是从哪一步开始 NaN 的
grep -n "nan\|NaN" logs/pretrain.log | head
```

从检查点重启，把 `--peak_lr` 降一半试试。
**重点是它发生的时间点** —— 如果是刚起步，是 LR 问题；
如果是几千步之后突然出现，更可能是数据问题。

---

### Q12. ★ 为什么预训练 loss 3.13，SFT loss 1.45，能说 SFT 更好吗？

**不能。这两个数字不可比。**

| | 预训练 | SFT |
|---|---|---|
| 数据 | 网页文本 | 对话 |
| 算 loss 的位置 | **所有** token | **只算 assistant 的回复** |
| 任务难度 | 预测任意下一个词 | 预测一个"好回答"的下一个词 |

「只算回答部分」这一条尤其关键 —— 回答文本比网页文本**好预测得多**
（格式固定、句子短、风格统一），所以 loss 天然更低。
**这不是模型变强了，是任务变简单了。**

**唯一能跨阶段比较的是 GSM8K 的 pass@1。**
这也是为什么这个项目要专门做评测（[eval_gsm8k.py](../src/eval_gsm8k.py)）。

> 同理：**不同模型之间的 loss 也不可比**（词表大小不同，ln(vocab) 就不同）。
> 看到别人报 loss，第一件事是问「什么配置下的」。

---

### Q13. loss 一直不降 / 降得特别慢

| 检查项 | 怎么查 |
|---|---|
| LR 是不是太小 | 日志里的 `lr` 字段 |
| 数据对不对 | `python src/data_pretrain.py --max_files 1` 后手动 decode 看看（[05](05-数据管线.md#61-分片里到底是什么文本)） |
| 模型有没有真的在更新 | 检查点的 `step` 有没有涨 |
| 是不是卡在 warmup | warmup 500 步，前 500 步 LR 从 0 线性上升，loss 降得慢是正常的 |
| 是不是 labels 错了 | loss 从 11.9 还是 12.9 起步？**12.9 = labels 没左移一位** |

---

### Q13′. ★ GRPO 的 loss 恒为 0 / 权重纹丝不动 / `有信号组 0/24`

**先说结论：这三个现象里，最可能的原因是「裁判坏了」，不是「模型太弱」。**

**别做的事**：调 LR、加步数、换 `beta` —— 裁判坏了的话这些全都白费。

**要做的第一件事**：拿标准答案原文喂给奖励函数（**10 秒钟**）：

```bash
cd /root/Test/mini-r1 && CUDA_VISIBLE_DEVICES="" miniforge/envs/mini-r1/bin/python -c "
import sys; sys.path.insert(0, 'src')
from train_grpo import math_reward, format_reward
from datasets import load_dataset
d = load_dataset('parquet', data_files='data/raw/gsm8k/train.parquet', split='train')
ex = d[0]
print('gold 字段的末尾:', repr(ex['answer'][-40:]))
print('拿原文当输出喂进去 ->', math_reward(ex['answer'], ex['answer']))  # 必须是 1.0
print('格式分 ->', format_reward(chr(92) + 'boxed{1}'))                  # 必须是 0.1
"
```

实测输出（**这两行就是判据**）：

```
gold 字段的末尾: 'ips altogether in April and May.\n#### 72'
拿原文当输出喂进去 -> 1.0
格式分 -> 0.1
```

> 💡 那个 `chr(92)` 是反斜杠 `\`。**别直接写 `\boxed`** —— 反斜杠穿过
> 双引号 + shell 两层转义后会被吃掉或变成字面反斜杠，实测踩过
> （`SyntaxError: f-string expression part cannot include a backslash`）。
> 用 `chr(92)` 拼出来最稳。

**`math_reward` 返回 0.0 就是裁判坏了。** 项目里真发生过：

| 现象 | 真实原因 |
|---|---|
| `r 0.0000`，`有信号组 0/24` | 奖励恒为 0，**一个梯度都拿不到** |
| loss 恒为 0，权重和基座逐位相同 | 同上（不是"GRPO 没效果"） |
| `pass@1` 恒为 0 | 同上（不是"SFT 基座不行"） |
| **不报错、不警告** | 没有任何东西告诉你在算错的奖励 |

**根因**：GSM8K 的 `answer` 字段是**整段解题过程**，答案在最后一行 `#### 72`。
原来的代码拿整段跟模型输出的 `"72"` 比，永远不相等。

**修法**：[train_grpo.py](../src/train_grpo.py) 里的 `gold_answer()`
（取 `####` 之后那段），并且 `math_reward` 内部也过一遍，
调用方传整段还是传答案都不会错。

**防线**：`load_gsm8k()` 结尾会跑 `check_reward_fn(train[:200])` ——
**自检不过就不让开始训练**，打印 `奖励函数自检: 200 条标准答案全部判对 ✓`。
没看到这行就是没跑过自检。

> 📌 **通用原则**：凡是「奖励 / 判分 / 评分」这类函数，
> **第一件事是用满分样本试它**，而不是等训练跑起来看曲线。
> 曲线不动的原因有几十种，但"裁判坏了"是唯一一种
> **再等 300 步也不会自己好**的。

**如果自检过了但 `有信号组` 仍然低** —— 那才是模型/数据问题，按顺序查：

| 检查项 | 怎么查 |
|---|---|
| `max_new` 太小 | 384 是合理的；早期拿 32 试跑时 8 条回答全被截断，奖励必然全一样 |
| 基座是不是根本不会做 | 先跑 `eval_gsm8k.py` 看 pass@1 和 `format_ratio`（见 §4.1′） |
| 格式分是不是唯一的信号 | `r` 恰好是 `0.1 × 用了 \boxed{} 的比例` → 说明算对的那些一个都没有。**这时先去看 Q13‴**，别急着调超参 |

---

### Q13″. ★ 加了 KV-Cache 之后生成的文本"通顺但答非所问"

**这是静默错误的典型症状**：不崩、不警告、loss 照常下降，只是模型变傻。

两个可能的原因，都在 `model.py` 里：

| 现象 | 原因 | 正确写法 |
|---|---|---|
| 完全胡说 / 只重复第一个 token | **SDPA 的 `is_causal=True` 是左上角对齐的** —— `T=1`（增量步）时那唯一一行只能看到第 0 个 key | `T==1` 时不加掩码、`is_causal=False`；`T>1` 时手写 `ones(T, total).tril(past_len)` |
| 位置像是乱的 / 长文本崩溃 | RoPE 用了 `cos[:T]` 而不是 `cos[past_len:total]` —— **形状完全对得上**，所以不报错 | `self.cos[past_len:total]` |

**怎么确认**：跑数值等价性测试（纯 CPU，10 秒，不抢卡）

```bash
CUDA_VISIBLE_DEVICES="" python scripts/test_kv_cache.py
```

它做 8 项检查，包括「多 token 续写 vs 整条前向」「缓存不改写历史」
「贪心解码逐 token 对齐」。**只要动过 `Attention.forward` 或掩码逻辑就跑它。**

> ⚠️ 这个测试自己也很容易写错下标（第一版就错了）：
> 「第 j 行的 logits」是喂完下标 j 的 token **之后**的分布，对应**预测下标 j+1**。
> 判断依据：**如果每一行都错，而且是错差不多的量，那是比较对象错了，不是模型错了。**

---

### Q13‴. ★★★ 训练日志每个指标都在变好，但模型其实在退化（奖励被 hack）

**这是本项目最贵的一课**（2026-10-14，300 步 GRPO 跑完之后才发现），
也是唯一一个「**没有任何报错、所有仪表盘都指向上、而结果是负的**」的故障。

**现象**（全部来自真实日志，一个数字都没编）：

| 指标 | 变化 | 看着像 |
|---|---|---|
| `r`（原始奖励） | 0.0549 → **0.0976** | 奖励快翻倍了 |
| 格式正确率 | 0.426 → **1.000** | 格式学得完美 |
| 贪心 `pass@1` | 0.0000 → **0.0156** | 准确率起来了 |
| `有信号组` | 17.2 → **7.3** | 只是任务变难了 |

**真相**：`math_reward` 全程恒为 0，`r` 的上限**就是那 0.1 的格式分**。
`r → 0.099` 不是「接近满分」，是「**把唯一能拿的分拿完了**」。
而 1.000 的格式正确率是这么来的：

```python
PROMPT_TMPL = "... put your final answer within \\boxed{{...}}."   # 提示词里有现成的占位符
def format_reward(c):  return 0.1 if re.search(r"\\boxed\{", c) else 0.0   # 只查这 4 个字符
```

**模型发现的最省力路径不是解题，是把提示词里的 `\boxed{...}` 抄一遍。**
实测贪心 256 题，42 条输出里有 `\boxed{`，其中 **42 条全是字面量占位符，一条真答案都没有**。
采样口径下含占位符的比例从 4.7% 涨到 **66%**。
原始输出长这样（**未经 `extract_answer` 处理**）：

```
4. cannot be true within \boxed{...}
5.
```

#### 怎么早发现它 —— 三个判据

| 判据 | 具体怎么查 |
|---|---|
| **奖励卡在某个常数上限附近** | `r` 逼近一个整数/半整数（这里是 0.1），而数学分那一项是 0 → 算一下 `0.1 × 格式合规率` 是不是恰好等于 `r` |
| **`r` 涨、`有信号组` 跌** | 这个组合只有一个解释：**奖励在趋同** —— 大家都拿到了同一个白送分。**两个数字必须一起看**，单看一个都是好消息 |
| **⚠️ 必须看原始输出** | `scripts/show_completions.py <ckpt> 3 3` —— 它打印**解码后未抽取**的原文，统计字面量占位符的条数 |

最后一条是**唯一能一锤定音的**。前两条是线索，不是证据 ——
`r` 涨也可能是真在学，`有信号组` 跌也可能是任务确实变简单了。

```bash
# 三条命令，10 分钟，能省掉一次 12 小时的错误训练
python scripts/show_completions.py ckpts/grpo/grpo.pt 3 3    # ① 看原文
python scripts/analyze_eval.py reports/gsm8k_grpo.json       # ② 看抽出来的答案分布
python scripts/eval_greedy.py ckpts/grpo/grpo.pt 256         # ③ 对齐两个 pass@1 口径 + 数占位符
```

#### 修法

**两道缺一不可**（只做①不做②，模型会换个更省力的抄法）：

```python
# ① 格式奖励: 要求括号里真的有东西, 且不是占位符
m = re.findall(r"\\boxed\{([^{}]*)\}", completion)
ok = any(normalize(x) not in (None, "", "...", "…") for x in m)
return 0.1 if ok else 0.0

# ② 提示词里不能再出现可被照抄的字面量
#    改成不含省略号的措辞, 或干脆不提 \boxed (评分里认它就够了)
```

> 📌 **通用原则**：**奖励函数里用到的字符串，绝不能同时出现在提示词里。**
> 规则奖励的漏洞不是"猜"出来的，是**读出来**的 —— 你的正则、关键词、占位符，
> 模型全都看得见。判据不是「模型应该看得懂 `...` 是占位符」，
> 而是「**这条最省力的路径存不存在**」。
>
> **还有一条更一般的**：**代理指标涨 ≠ 目标变好。**
> 这次 format_ratio、`r`、贪心 pass@1 三个代理指标全在涨，真实准确率是 0。
> **任何一次训练跑完，至少要有一个环节是去看原始输出**，
> 否则你优化的就是仪表盘，不是模型。

> ⚠️ **附带的坑：同名不同义的 `pass@1`。**
> 训练日志里的 `pass@1` 是**贪心**（`train_grpo.eval_pass1`，temperature=0，1 条）；
> `run_eval.sh` 里的 `pass@1` 是 **temperature=1.0 采样 8 条取第 1 条**。
> 这次两个口径给出**相反的排序**（0.0078 vs 0 / 0 vs 0.0156）——
> **在噪声里挑口径 = 自欺欺人**。要用 `scripts/eval_greedy.py` 对齐后再比。

---

## 📦 数据

### Q14. `datasets` 下载卡住 / `ConnectionError`

**原因**：你在直连 `huggingface.co`（国内不通）。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/Test/mini-r1/.cache/huggingface
```

**如果设了镜像还是崩** —— 那是 hf-mirror 的部分 API 会 302 跳回原站。
**本项目因此完全放弃了 `datasets` 的下载能力**，
改用 curl 直接下文件：[download_data.sh](../scripts/download_data.sh)。

---

### Q15. parquet 文件损坏 / 读出来是乱的

**原因**：两个下载进程同时对同一个文件 `-C -` 续传，
**交错写入**，产生一个"大小正常但内容损坏"的文件。

**预防**：[download_data.sh](../scripts/download_data.sh) 用了 flock：

```bash
exec 9>/tmp/.mini_r1_download.lock
flock -n 9 || { echo "已有下载实例在跑, 退出"; exit 1; }
```

**处理**：删掉重下。

```bash
rm data/raw/00X_00000.parquet
bash scripts/download_data.sh 3 1        # 只重下第 3 个
```

---

### Q16. 磁盘满了

```bash
~/.server_cmd.sh 'df -h /root/Test && du -sh /root/Test/mini-r1/*'
```

**能删的**（按性价比）：

| 目录 | 大小 | 说明 |
|---|---|---|
| `.cache/` | 几 GB | HF / pip 缓存，删了会自动重建 |
| `data/raw/` | 29 GB | **理论上可删**（分片已生成），但想改分词器重跑管线时又要重下 |
| `ckpts/*/` 里的旧检查点 | 各 5 GB | 确认不需要续训了才能删 |

**不能删的**：`data/train/`（重跑要 2 小时）、`tokenizer/`、当前在用的检查点。

---

## 🍎 macOS 本地

### Q17. `timeout: command not found`

**macOS 没有 GNU 的 `timeout`**（那是 Linux 的工具）。
装了 coreutils 的话有 `gtimeout`：

```bash
brew install coreutils
gtimeout 10 some_command
```

**但更重要的教训**：`timeout N` 是**从进程启动就计时**的。
如果一个任务光加载数据就要 4 分钟，`timeout 30` 会在加载阶段就把它杀掉。
**要按输出内容判断结束条件，不要按秒数。**

```bash
# ❌ 会在加载阶段被杀
timeout 30 python train.py

# ✅ 等日志里出现某个标志
until grep -q "\[ckpt\] step 32 已存" run.log; do sleep 2; done
```

---

### Q18. 解析 `~/.server_cmd.sh` 的输出老是对不上

**原因**：`~/.server_cmd.sh` 的输出**前面有回显的命令本身**，
还可能带 `\r`（回车符）。用 `=` 精确比较会莫名失败。

```bash
# ❌ 会失败
out=$(~/.server_cmd.sh 'test -f x && echo 1')
[ "$out" = "1" ] && echo "存在"

# ✅ 用通配匹配
case "$out" in *"1"*) echo "存在" ;; esac
```

**真实案例**：本项目有一个后台任务用 `[ "$out" = "1" ]` 判断检查点是否存在，
结果在**检查点早就写好、日志里明摆着**的情况下，跑了 40 分钟后报
「仍未见 sft_latest.pt」—— **假警报**。
差点让人去"修"一个根本不存在的问题。

> 📌 **假信号比没有信号更危险**：它会让你修改正确的东西。

---

### Q19. 服务器上多出一堆 `._` 开头的小文件

macOS 的 AppleDouble 格式，纯垃圾。

```bash
~/.server_cmd.sh 'find /root/Test/mini-r1 -name "._*" -delete'
```

**根治**：同步时排除（见 [02-服务器与SSH](02-服务器与SSH.md#74-macos-传上去的-_-文件)）。

---

## 🧪 代码与测试

### Q20. 改了预训练的记账逻辑，怎么验证没改坏

```bash
python scripts/test_resume_accounting.py
```

纯 CPU，几秒钟，验证 `plan_progress()` / `decay_start()` 在各种边界输入下的行为。
**改记账/调度逻辑后必跑。**

---

### Q20′. 改了 KV-Cache 相关的代码，怎么验证没改坏

```bash
CUDA_VISIBLE_DEVICES="" python scripts/test_kv_cache.py
```

纯 CPU，10 秒。8 项数值等价性检查：预填充 vs 整条前向、单 token 增量、
**多 token 续写**、不同长度分批、缓存不改写历史、贪心解码逐 token 对齐、
`use_cache=False` 要报错、eos 提前停止。

原理和那两个静默坑见上面 Q13″。

---

### Q20″. 改了梯度检查点 / 模型加载相关的代码，怎么验证没改坏

```bash
CUDA_VISIBLE_DEVICES="" python scripts/test_grad_ckpt.py
```

纯 CPU，秒级。3 项：

1. 带 KV-Cache 的 `generate()` 与不打补丁的模型**逐 token 一致**；
2. 不开缓存的反传梯度与不打补丁的**逐张量相同**（实测最大差 `0.00e+00`）；
3. 默认参数（预训练/SFT 用的那档）下传 `use_cache` **仍然要报错** ——
   别在给 GRPO 开口子的时候把护栏顺手拆了。

**`CUDA_VISIBLE_DEVICES=""` 别省** —— 共享机器上，纯 CPU 的测试去占一张卡是纯浪费。

背景见 [docs/04 ①(c)](04-代码精读.md) 和 Q3′。

---

### Q21. 想在本地（无 GPU）验证代码能不能跑

```bash
# ① 模型本体：CPU 就能跑完整的前向+反向+采样+KV-Cache 等价性
python src/model.py

# ② 用随机数据跑通训练流程
python scripts/make_smoke_data.py /tmp/smoke
python src/train_pretrain.py --data_dir /tmp/smoke --ckpt_dir /tmp/smoke_ckpt \
    --micro_batch 2 --grad_accum 2 --max_tokens 2000000

# ③ 检查点记账的冒烟测试
python scripts/test_resume_accounting.py

# ④ KV-Cache 等价性（8 项）
python scripts/test_kv_cache.py
```

**这四条不需要 GPU、不需要数据集**，是提交代码前的最低验证。

---

## 📋 一张速查表

```bash
# 状态
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'

# 进程
~/.server_cmd.sh 'pgrep -af "[t]rain_grpo.py"'
~/.server_cmd.sh 'pgrep -af "[w]atch_grpo.sh"'

# 日志
~/.server_cmd.sh 'tail -20 /root/Test/mini-r1/logs/grpo.log'
~/.server_cmd.sh 'tail -20 /root/Test/mini-r1/logs/grpo_watch.log'

# GPU
~/.server_cmd.sh 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'

# 磁盘
~/.server_cmd.sh 'df -h /root/Test; du -sh /root/Test/mini-r1/*'

# 网络
nc -z -v 10.176.54.32 13811; ifconfig | grep -A3 utun

# 启动 / 停止 （换阶段就把 grpo 换成 sft / pretrain）
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/run_grpo.sh'
~/.server_cmd.sh 'cd /root/Test/mini-r1 && setsid nohup bash scripts/watch_grpo.sh >/dev/null 2>&1 &'
~/.server_cmd.sh 'pkill -f "[w]atch_grpo.sh"'          # 只停守护，训练进程不动

# 纯 CPU 的回归测试（★ 记得先腾空 CUDA_VISIBLE_DEVICES）
~/.server_cmd.sh 'cd /root/Test/mini-r1 && CUDA_VISIBLE_DEVICES="" python scripts/test_kv_cache.py'
~/.server_cmd.sh 'cd /root/Test/mini-r1 && CUDA_VISIBLE_DEVICES="" python scripts/test_grad_ckpt.py'

# 崩溃循环：先数次数、再决定拉不拉（Q3′）
~/.server_cmd.sh 'cd /root/Test/mini-r1 && grep -c OutOfMemoryError logs/grpo.log; tail -3 logs/grpo_watch.log'

# ★ 指标全绿但要确认模型真的在变好（Q13‴）—— 看**原始输出**，别看仪表盘
~/.server_cmd.sh 'cd /root/Test/mini-r1 && python scripts/show_completions.py ckpts/grpo/grpo.pt 3 3'
~/.server_cmd.sh 'cd /root/Test/mini-r1 && python scripts/eval_greedy.py ckpts/grpo/grpo.pt 256'
~/.server_cmd.sh 'cd /root/Test/mini-r1 && python scripts/analyze_eval.py reports/gsm8k_grpo.json'
```

---

## 最后：四条通用排错原则

1. **先问「它应该是什么样」，再看「实际是什么样」。**
   比如「预训练显存应该是 12,046 MiB」——
   有了这个基准，才知道 10,850 是正常的、16,000 是异常的。

2. **区分「报错」和「静默错误」。**
   报错好办（有堆栈）。**这个项目真正贵的教训全是静默的**：
   labels 没左移、跳过循环 off-by-one、`.gitignore` 规则失效、
   `wc -l` 数错卡数、**奖励函数恒返回 0**、
   **KV-Cache 的掩码/RoPE 偏移写错** —— 它们不报错，只产生错误的结果。

3. **让测试自己判定通过/不通过。**
   打印两列 loss 数字让人眼看，等于没有测试 ——
   「续训后 loss 低了 0.5」会被脑补成「收敛更好」。

4. **能在启动前用文件内容判断的，别等启动后。**
   比如守护脚本用 `grep 日志` 判断「是不是已经跑完了」，
   而不是把训练拉起来、让 `torch.load` 读 4.9 GB 检查点之后再判断。

---

回到 [文档首页](README.md) ｜ 遇到没覆盖的问题，去 [AGENTS.md](../AGENTS.md) 的
「踩过的坑」一节找找，那里有 18 条更底层的记录。
