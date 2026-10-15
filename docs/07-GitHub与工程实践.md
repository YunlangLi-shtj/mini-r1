# 07 - GitHub 与工程实践

> 这篇讲三件事：**这个仓库现在是什么状态**、**哪些东西故意没进仓库**、
> **怎么改代码而不搞坏它**。最后有一节讲「怎么不把密码传上 GitHub」——
> 这是每个用服务器的人都该知道的事。

---

## 1. 仓库现状

| 项 | 值 |
|---|---|
| 地址 | https://github.com/YunlangLi-shtj/mini-r1 |
| 可见性 | **public** |
| 主分支 | `main` |
| 体积 | 很小（代码 + 文档，几百 KB） |

```bash
git clone https://github.com/YunlangLi-shtj/mini-r1.git
cd mini-r1
```

**clone 下来只有代码和文档** —— 没有数据、没有模型、没有日志。
这是故意的，见 §2。

### 1.1 仓库里的文件

```
mini-r1/
├── README.md          # 项目门面（给人看的第一眼）
├── AGENTS.md          # 交接文档（实际进度 + 踩坑记录 + 设计取舍）
├── CLAUDE.md          # 一行 @AGENTS.md，让 AI 助手自动读取上下文
├── .gitignore         # 排除大文件（★ 很重要，见 §2）
├── src/               # 9 个 Python 文件，约 1440 行
├── scripts/           # 21 个脚本（bash + 2 个 py），约 1120 行
└── docs/              # 你正在读的这套教程
```

### 1.2 两份文档的分工（值得学）

| | [README.md](../README.md) | [AGENTS.md](../AGENTS.md) |
|---|---|---|
| **读者** | 陌生人、面试官、招聘方 | 接手项目的人 / AI 助手（包括未来的自己） |
| **风格** | 门面：亮点、架构图、公式、实验结果 | 内部：现在到哪了、为什么这么写、踩过什么坑 |
| **长度** | 115 行，克制 | 386 行，详尽 |
| **更新频率** | 低（结论性的东西） | 高（每次状态变化都改） |

**为什么要分开**：README 太长没人看，AGENTS.md 太简略没法交接。
混在一起的结果是两边都做不好。

### 1.3 `CLAUDE.md` 是什么

```markdown
@AGENTS.md
```

就一行。这是 Claude Code（AI 编程助手）的约定：
启动时会自动读取项目根目录的 `CLAUDE.md`，`@文件名` 表示把那个文件的内容也读进来。

**为什么需要它**：AI 助手**不会自动知道**你上周做了什么决定、
哪个坑已经踩过。把这个上下文写在仓库里，
**无论从哪里启动助手，它都能立刻接上**。

> 📌 这条经验对人也适用：**项目的上下文要写在仓库里，不能只留在聊天记录里。**

---

## 2. 什么没进仓库，为什么

[.gitignore](../.gitignore) 的全部内容：

```gitignore
.DS_Store
__pycache__/

# ---- 大文件 / 运行时产物: 只存在于服务器, 永远不进仓库 ----
data/          # 66G: FineWeb-Edu 分片 + 原始 parquet/json + gsm8k
ckpts/         # 4.6G+: 模型检查点 + train_log.jsonl
logs/          # 训练/守护脚本日志
miniforge/     # 7G: conda 环境

*.pt           # 检查点
*.bin          # token 分片（data/ 已排除，这条是双保险）
*.parquet
*.part         # 下载中的临时文件
```

### 2.1 为什么排除它们

| 目录 | 大小 | 为什么排除 |
|---|---|---|
| `data/` | 66 GB | 公开数据集，**能用代码重新生成** |
| `ckpts/` | 15 GB | 训练产物，同上 |
| `miniforge/` | 7 GB | 环境，用 `server_setup.sh` 重建 |
| `logs/` | 小但很多 | 运行产物，没有版本价值 |

**三条原则**（适用于任何项目）：

1. **体积**：GitHub 单文件上限 100 MB，仓库建议 < 1 GB。
   一旦 commit 了一个大文件，**它会永远留在历史里**（除非重写历史）。
2. **可再生产物不进版本控制**。能由代码生成的东西，存"生成它的代码"就够了。
3. **环境不进仓库**。你的 conda 环境在别人机器上没用，给他一个 `setup.sh` 更有用。

### 2.2 `.gitignore` 的一个隐藏陷阱

```gitignore
data/  # 66G 数据
```

**这行是错的！** `.gitignore` 只在**行首**认 `#` 注释，
行尾的 `# 66G 数据` 会被当成模式的一部分，于是这行匹配的是
「一个叫 `data/  # 66G 数据` 的文件」—— **规则静默失效**。

正确的写法：

```gitignore
# 66G: FineWeb-Edu 分片 + 原始 parquet/json + gsm8k
data/
```

本项目的 [.gitignore](../.gitignore) 里专门写了注释提醒这一点：

```gitignore
# 注意: .gitignore 只在行首认 # 注释, 不能写 "data/  # 说明" —— 那样整行会被
# 当成一个模式, 规则静默失效。说明一律单独成行。
```

> 📌 **这是"静默失败"的典型**：不报错、不警告，只是规则没生效。
> 等你发现的时候，66 GB 已经在 `git status` 里了。

### 2.3 怎么确认规则真的生效了

```bash
# 看看还有没有该忽略却没忽略的
git status --porcelain

# 检查某个文件是不是被忽略了
git check-ignore -v data/train/shard_00000.bin
# 期望输出: .gitignore:20:data/   data/train/shard_00000.bin
#           ↑ 文件名 : 行号 : 匹配到的规则

# 已经 add 进暂存区的大文件？列出来
git diff --cached --stat | tail -5
```

---

## 3. ⚠️ 密码绝不能进仓库

这是每个用远程服务器的人**必须**建立的反射。

### 3.1 为什么

- git 历史是**永久**的。删掉文件、再提交一次，**旧版本还在历史里**，
  `git log -p` 或 GitHub 的搜索都能翻出来。
- 推到 GitHub 就是**公开**。机器人会在推送上线的几秒内扫描
  `password`、`ssh`、`api_key` 等关键词。
- 密码泄露 = 别人可以用你的身份登录服务器、删你的数据、
  甚至用这台机器做别的事。

### 3.2 本项目的做法

密码只存在于**仓库之外**的两个文件里：

```
~/.server_cmd.sh      # 执行远程命令
~/.server_sync.sh     # 同步代码
```

它们在**家目录**，而仓库在 `~/mini-r1`。`git` 只会跟踪仓库目录内的文件，
所以天然不会碰到它们。并且：

- 这两个文件**没有**被 `git add` 过
- [AGENTS.md](../AGENTS.md) §3.2 明确写着「密码不写进仓库」
- 文档（包括本目录）里一律用 `<你的密码>` 占位

### 3.3 更通用的做法（推荐）

**① 环境变量 + `.env` 文件**

```bash
# .env（加入 .gitignore！）
SSH_PASS=xxxxx

# 脚本里
source ~/.env
```

**② SSH 密钥（最推荐）**

```bash
ssh-keygen -t ed25519 -C "me@example.com"      # 生成一对密钥
ssh-copy-id -p 13811 root@10.176.54.32         # 把公钥装到服务器上
# 以后直接 ssh，不需要密码 —— 也就没有密码可以泄露
```

私钥（`~/.ssh/id_ed25519`）**永远不离开你的电脑**。
本项目没用这个方案（服务器是别人管的，加公钥要走流程），但你应该知道它更好。

**③ 提交前扫一遍**

```bash
# 看看这次要提交什么
git diff --cached

# 万一手滑，搜一下暂存区里有没有敏感词
git diff --cached | grep -iE 'password|passwd|secret|token|api[_-]?key'

# 搜整个历史（慢，但值得偶尔跑一次）
git log -p | grep -iE 'sshpass|password' | head
```

**④ 用工具兜底**：`git-secrets`、`gitleaks`、`trufflehog`
可以做成 pre-commit 钩子，自动阻止含密钥的提交。

### 3.4 如果已经推上去了怎么办

**光删文件再提交是没用的**，历史里还在。正确顺序：

1. **立刻改密码**（第一步永远是止损，不是清理）
2. 用 `git filter-repo` 或 BFG 重写历史，把那段内容从所有 commit 里抹掉
3. 强推 `git push --force`
4. 通知所有 clone 过这个仓库的人重新 clone

⚠️ 注意：**如果仓库已经被 fork 或缓存，你无法完全清除**。
所以「永远不要提交」比「提交后再清理」重要得多。

---

## 4. macOS 特有的两个坑

### 4.1 `.DS_Store`

Finder 浏览过的每个目录都会生成一个 `.DS_Store`（存图标位置之类的元数据）。
它已经在 [.gitignore](../.gitignore) 里了。

```bash
# 全仓库清理一次
find . -name '.DS_Store' -delete
```

### 4.2 `._` 文件（AppleDouble）

在 macOS 上用 `tar` 打包或 `scp` 传文件时，可能会多出一堆 `._xxx` 文件：

```
._AGENTS.md    ._src    ._README.md    ._.
```

这是 macOS 把「扩展属性」（比如"从哪下载的"、自定义图标）单独存成的小文件。
**在 Linux 上毫无意义，是纯垃圾。**

```bash
# 服务器上清理
find /root/Test/mini-r1 -name '._*' -delete

# 或者在同步时排除（推荐，见 docs/02 §3）
tar czf - --exclude='._*' --exclude='.DS_Store' --exclude='.git' .
```

本项目服务器上确实有一批这样的文件 —— 它们是早期用 tar 同步时带过去的。

---

## 5. 提交习惯

### 5.1 本项目的 commit message 风格

```bash
$ git log --oneline -6
5e8f130 SFT 断点续训 + 续训回归测试 (测试抓到两个静默 bug)
49674ee 多卡启用: 两级显存门槛 + 修 n_of 的 wc -l 计数 bug
686b18e 多卡实测脚本 bench_ddp.sh + 记录多卡显存门槛
38d1f8a README/GRPO 文档对齐实际实现
98e27de 加固 .gitignore: 排除服务器上的 data/ckpts/logs/miniforge
```

特点：

- **中文**（项目是中文的，读者也是中文的）
- **一句话说清"做了什么"**，不写 "update" / "fix bug" 这种没有信息量的
- **括号里补"为什么"或"结果"**：`(测试抓到两个静默 bug)`、
  `(测试抓到)` —— 这是最有价值的部分，半年后你自己就靠这个回忆
- **一个 commit 一件事**：能单独回滚

### 5.2 一个工作流

```bash
# 1. 改代码（本地改！服务器上的改动会被同步覆盖）
vim src/train_sft.py

# 2. 跑测试（见下）
python scripts/test_resume_accounting.py

# 3. 同步到服务器
bash ~/.server_sync.sh /Users/liyunlang/mini-r1 /root/Test/mini-r1

# 4. 看一眼要提交什么
git status
git diff

# 5. 提交
git add -A
git commit -m "SFT 断点续训 + 续训回归测试 (测试抓到两个静默 bug)"

# 6. 推送
git push
```

### 5.3 改代码之后必须做的事

**[AGENTS.md](../AGENTS.md) §4** 写着：

> 改代码后的固定动作：`server_sync.sh` → 跑一次 `test_resume_accounting.py` → 提交。

```bash
# 改了预训练的记账/调度逻辑，必须跑这个（纯 CPU，不抢 GPU）
python scripts/test_resume_accounting.py
```

**为什么这条规则重要**：断点续训的代码路径**只在"出事时"被执行**。
出事时你可能在睡觉，而这条路径从来没跑过。
所以要在没事的时候，主动把它跑一遍。

同理，改了 SFT 的续训逻辑要跑：

```bash
~/.server_cmd.sh 'cd /root/Test/mini-r1 && setsid nohup bash scripts/test_sft_resume.sh >/dev/null 2>&1 &'
# 约 10 分钟，结果看 logs/test_sft_resume.log
```

**这个测试真的抓到过 bug** —— 两个静默 bug（`torch.save` 不建父目录、
跳过循环的 off-by-one）。详见 [04-代码精读](04-代码精读.md#怎么验证它-1)。

---

## 6. 当前待办（仓库层面）

| 事项 | 说明 |
|---|---|
| **push 未推送的提交** | 本地领先 `origin/main` 若干提交，见下 |
| **LICENSE 缺失** | 仓库没有开源许可证。用了别人的数据集（ODC-By / Apache-2.0 / MIT），自己却没声明许可，别人想用会有顾虑 |
| **`reports/` 和 `configs/` 是空目录** | git 不跟踪空目录，所以 clone 下来看不到它们。要么删掉，要么放个 `.gitkeep` |
| **`data_prefetch.py` 已过时** | 它是早期方案，实际用的是 curl 下载。可以在文件头加个说明 |
| **`configs/` 里没有配置** | 所有超参都写在 `run_*.sh` 的命令行里。想做成可切换的配置文件可以放这里 |

### 6.1 检查有没有没推送的提交

```bash
git log --oneline origin/main..HEAD        # 列出未推送的
git status -sb                              # 首行会显示 [ahead N]
```

**为什么要在意**：所有工作都在本地 commit 但没 push，
硬盘坏了就全没了 —— 而且 `origin` 上那份还是旧的。

```bash
git push origin main
```

---

## 7. 关于开源许可证（补充说明）

这个项目用了三个数据集，各自的许可：

| 数据集 | 许可 | 要求 |
|---|---|---|
| FineWeb-Edu | ODC-By 1.0 | 署名 |
| OpenHermes-2.5 | Apache-2.0 | 保留声明 |
| GSM8K | MIT | 保留声明 |

**代码本身没有 LICENSE 文件**，这在公开仓库里是个明显的空缺。
常见选择：

- **MIT**：最宽松，随便用。适合「希望别人尽量用起来」的项目
- **Apache-2.0**：和 MIT 类似，但多一条**专利授权**条款。数据/模型项目常用
- **CC-BY-NC-4.0**：禁止商用。数据集常用（FineWeb-Edu 就是这个）

**加 LICENSE 的方法**（GitHub 上直接操作最省事）：
仓库页面 → Add file → Create new file → 文件名输入 `LICENSE` →
右侧会出现 "Choose a license template"。

> ⚠️ 这是**需要你自己决定**的事（涉及法律含义），
> 本文只列出选项和影响，不替你选。

---

下一篇 → [08-排错FAQ](08-排错FAQ.md)。
