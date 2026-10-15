# 02 - 服务器与 SSH

> 这篇假设你**从来没连过远程服务器**。读完你应该能：连上一台 Linux 机器、
> 在上面找文件看文件、把本地代码传上去、让一个要跑三天的任务在断网后继续跑。

---

## 1. 为什么需要一台服务器

你的笔记本没有 GPU（或者只有一块小显卡），而训练要连跑三天。
所以实际的计算发生在**别人机房里的另一台电脑**上，你这台机器只负责敲命令、看结果。

这台「另一台电脑」就叫**服务器**。它的特点是：

- 有 GPU（本项目用的是 6 张 RTX 4090）
- 不关机（你的笔记本合盖就睡了，训练就断了）
- 多人共用（**这一点很重要**，后面反复出现：显存会被别人抢）
- 你只能通过命令行访问，没有桌面

你和它之间通过网络通信，用的协议叫 **SSH**（Secure Shell）。

---

## 2. SSH 是什么，怎么用

SSH 做的事情很简单：**把你敲的一行命令发过去，把它打印的东西传回来**，
中间全程加密。就像你在那台机器上开了一个终端，只不过这个终端隔着网线。

### 2.1 基本命令

```bash
ssh <用户名>@<服务器地址> -p <端口>
```

- `用户名`：服务器上的账号，本项目是 `root`（最高权限账号）
- `服务器地址`：IP 或域名，例如 `10.176.54.32`（这是内网地址，见 §6）
- `端口`：SSH 服务监听的端口，默认 22。本项目用 `13811`（改端口是为了少被扫描）

回车后会提示：

```
The authenticity of host '[10.176.54.32]:13811' can't be established.
ED25519 key fingerprint is SHA256:xxxxx...
Are you sure you want to continue connecting (yes/no)?
```

这是**第一次**连接时的正常提示，问你「这台机器的指纹我记下来了，确认吗」。
输入 `yes`。以后再连就不会问了（指纹记在 `~/.ssh/known_hosts` 里）。

⚠️ 如果**以前连过、现在突然又弹这个警告**，那要小心 ——
可能是服务器重装了，也可能是有人在中间冒充它。核实清楚再输 yes。

然后输入密码（**输入时屏幕上不会有任何显示**，不会出现 `****`，这是正常的，
输完直接回车）。

### 2.2 密码 vs 密钥

| 方式 | 怎么工作 | 优点 | 缺点 |
|---|---|---|---|
| **密码认证** | 每次连接输入密码 | 简单，换电脑也能用 | 每次都要输；密码可能被脚本存下来泄露 |
| **密钥认证** | 本地生成一对公私钥，公钥放到服务器上，私钥留在本地 | 免密，更安全 | 需要一次性配置 |

本项目**用的是密码认证**（服务器是别人管的，加公钥要走流程）。
代价就是：**密码得存在某个地方，否则每次都手输** —— 见下一节。

> 🔐 无论用哪种，**密码/私钥绝对不能进 git 仓库**。
> 一旦推到 GitHub（哪怕是私有仓库再转公开），就等于公开了。
> 本项目的做法见 [07-GitHub与工程实践](07-GitHub与工程实践.md)。

### 2.3 登录进去之后是什么样

```
root@fdu-4090-2:~#        ← 提示符变了，说明你已经在服务器上了
```

此时敲的每条命令都在**服务器**上执行，不是在你的 Mac 上。这一点最容易搞混：

```bash
$ ls                      # 列出的是服务器当前目录的文件！
$ python xxx.py           # 跑的是服务器上的 python
```

退出：输入 `exit`，或者按 `Ctrl-D`。

---

## 3. 一条命令远程执行：`~/.server_cmd.sh`

每次都 `ssh` 进去、敲命令、再退出，很麻烦。更常用的方式是**一次执行一条命令就返回**：

```bash
ssh root@10.176.54.32 -p 13811 'cd /root/Test/mini-r1 && bash scripts/status.sh'
```

注意命令要用**单引号**包起来，否则里面的 `$`、`*` 会被你本地的 shell 先解释一遍。

但这样每敲一次都要输一遍密码。所以本项目在本地放了两个小脚本
（**在仓库之外，永远不提交**）：

**`~/.server_cmd.sh`** —— 在服务器上执行一条命令：

```bash
#!/bin/bash
# 用法: ~/.server_cmd.sh '<远程命令>'
HOST=10.176.54.32
PORT=13811
USER=root
PASS='<你的密码>'          # ← 只存在这个文件里，不要提交到任何仓库
TIMEOUT=600

export SSHPASS="$PASS"
exec sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    -p "$PORT" "$USER@$HOST" "$@"
```

> `sshpass -e` 表示「密码从环境变量 `SSHPASS` 读」，这样密码不会出现在
> `ps` 的命令行里（别人 `ps aux` 看不到）。
> 如果机器上没装 sshpass（macOS 默认没有，`brew install sshpass` 也常被墙），
> 用下面的 `expect` 版本，或者干脆手输密码。

**`~/.server_sync.sh`** —— 把本地目录同步到服务器：

```bash
#!/bin/bash
# 用法: ~/.server_sync.sh <本地目录> <远程目录>   ★ 两个参数都要给
set -euo pipefail
HOST=10.176.54.32; PORT=13811; USER=root
PASS='<你的密码>'
LOCAL="$1"; REMOTE="$2"
[ -n "$LOCAL" ] && [ -n "$REMOTE" ] || { echo "用法: $0 <本地目录> <远程目录>"; exit 1; }

cd "$LOCAL"
tar czf - --exclude='.git' --exclude='__pycache__' --exclude='.DS_Store' . \
  | sshpass -p "$PASS" ssh -p "$PORT" -o StrictHostKeyChecking=no \
      "$USER@$HOST" "mkdir -p '$REMOTE' && tar xzf - -C '$REMOTE'"
echo "已同步 $LOCAL -> $REMOTE"
```

用起来就是：

```bash
# 在服务器上看状态
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'

# 把本地改好的代码传上去
~/.server_sync.sh /Users/liyunlang/mini-r1 /root/Test/mini-r1
```

⚠️ **`server_sync.sh` 必须带两个参数**。少给一个的话，`$2` 是空字符串，
`mkdir -p ''` 会失败 —— 但如果脚本写得不好，可能既不报错也没同步，
你以为传上去了，其实服务器跑的还是旧代码。**这个坑本项目踩过。**

### 3.1 同步是「覆盖写」，不是「镜像」

上面这个 tar 方式是**覆盖**：本地有什么就写过去什么，
**不会删除**远程多出来的文件。所以：

- ✅ 服务器上的 `data/`、`ckpts/`、`logs/`、`miniforge/`（只存在于服务器的大目录）**不受影响**
- ⚠️ 你在服务器上直接改的代码，下次同步会被本地版本**覆盖掉**
  （所以：**改代码在本地改，别在服务器上改**）

**或者**，如果只是传一两个文件：

```bash
scp -P 13811 ./src/model.py root@10.176.54.32:/root/Test/mini-r1/src/
#      ↑ 注意 scp 的端口参数是大写 -P，ssh 是小写 -p，这是个经典坑
```

---

## 4. 在服务器上查看文件

服务器没有图形界面，全靠命令。这几个够用 90% 的场景：

| 命令 | 作用 | 例子 |
|---|---|---|
| `ls -lh` | 列文件，`-h` 让人看得懂大小（4.6G 而不是 4889051030） | `ls -lh ckpts/pretrain/` |
| `cd` | 切目录，`cd -` 回上一个 | `cd /root/Test/mini-r1` |
| `pwd` | 我现在在哪 | |
| `cat` | 打印整个文件（**大文件别用**，会刷屏） | `cat .gitignore` |
| `head -20 f` | 看开头 20 行 | `head -20 logs/sft.log` |
| **`tail -20 f`** | **看结尾 20 行（最常用，看训练进度）** | `tail -20 logs/sft.log` |
| `tail -f f` | 实时滚动跟踪（`Ctrl-C` 退出） | `tail -f logs/sft.log` |
| `less f` | 分页浏览，`/关键词` 搜索，`q` 退出 | `less AGENTS.md` |
| `grep 关键词 f` | 在文件里搜 | `grep val_loss ckpts/pretrain/train_log.jsonl` |
| `wc -l f` | 数行数 | `wc -l src/*.py` |
| `du -sh dir` | 目录多大 | `du -sh data/*` |
| `df -h` | 磁盘还剩多少 | `df -h` |
| `find . -name "*.pt"` | 按名字找文件 | |
| `nvidia-smi` | **看 GPU 占用（最常用）** | 见 §5 |

几个实用组合：

```bash
# 训练到哪一步了？—— 只看最后一行 step 记录
tail -1 logs/sft.log

# 只看 checkpoint 记录
grep '\[ckpt\]' logs/sft.log | tail -5

# 今天的日志（假设日志里有时间戳）
grep "$(date '+%m-%d')" logs/sft_watch.log

# 在代码里搜一个函数定义在哪
grep -rn "def chunked_cross_entropy" src/
```

> 💡 **`nvidia-smi` 怎么看**：
> ```
> GPU  Index  Memory-Used / Memory-Total   GPU-Util
>   0     0      12046MiB /  24564MiB         97%     ← 有人在训练，显存被吃了 12G
>   1     1       1024MiB /  24564MiB          0%     ← 基本空闲，可以用
> ```
> 判断「能不能用」**要看显存，不是看利用率**。
> 别人的进程常常占着 10 GB 但利用率 0%（在等数据），这时候你上去照样 OOM。

---

## 5. 让任务在断线后继续跑（关键）

**这是初学者最容易踩的坑**：

```bash
# ❌ 千万不要这样启动长任务
ssh root@... 'cd /root/Test/mini-r1 && python src/train_pretrain.py ...'
```

为什么？因为 SSH 会话结束时，它启动的进程**会被一起杀掉**
（收到 `SIGHUP` 信号）。你的笔记本合盖、Wi-Fi 抖一下、地铁进隧道 —— 训练就没了。
一个跑了 20 小时的任务可能因此归零。

正确做法：让进程**脱离终端会话**，成为一个独立的守护进程：

```bash
setsid nohup <命令> > <日志文件> 2>&1 < /dev/null &
#  │      │                     │        │          └─ 放到后台
#  │      │                     │        └─ 标准输入从 /dev/null 读（没有终端可读）
#  │      │                     └─ 把 stdout 和 stderr 都写进日志文件
#  │      └─ 忽略 SIGHUP（断开连接时发的"挂断"信号）
#  └─ 新建一个会话，彻底脱离当前终端
```

本项目的所有 `run_*.sh` 和 [scripts/launch.sh](../scripts/launch.sh) 都是这么写的：

```bash
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/python -u src/train_sft.py \
  --base_ckpt ... > $PROJ/logs/sft.log 2>&1 < /dev/null &
echo "已启动，日志: $PROJ/logs/sft.log"      # 立刻返回
```

注意 `-u`：让 Python **不缓冲输出**。不加的话日志会攒一大块才写一次，
你用 `tail -f` 看会以为卡住了。

于是你可以这样用：

```bash
# 启动（立刻返回，不断线）
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/run_sft.sh'

# 过两小时回来看
~/.server_cmd.sh 'tail -3 /root/Test/mini-r1/logs/sft.log'
```

### 5.1 那怎么知道它还在不在？

```bash
~/.server_cmd.sh 'pgrep -af "[t]rain_sft.py"'    # 有输出 = 还在跑
```

`[t]` 这个方括号是个小技巧：它让**这条命令自己的命令行文本**不匹配这个正则
（自己的命令行里是 `[t]rain_sft.py`，不等于 `train_sft.py`），
否则 `pgrep` 会把自己也算进去（`ssh` 那层会带上命令文本）。本项目踩过这个坑。

### 5.2 `tmux`：另一个选择

如果你想要「进去看看、敲点东西、又不断开」，用 `tmux`：

```bash
tmux new -s train     # 新建一个名为 train 的会话
# ...在里面跑训练...
# 按 Ctrl-B 然后按 D  → "detach"，会话留在后台
tmux ls               # 看有哪些会话
tmux attach -t train  # 重新接回去，屏幕和离开时一模一样
```

本项目**没有用 tmux**，因为 `setsid nohup` 已经够了，而且守护脚本可以在
训练被杀后自动重启 —— 这是 tmux 做不到的。但你自己玩的时候 tmux 很舒服。

---

## 6. 网络：内网地址和 VPN

本项目服务器地址是 `10.176.54.32` —— 这是一个**内网（私有）地址**。

- `10.x.x.x`、`192.168.x.x`、`172.16~31.x.x` 这三段是国际约定的私有地址，
  **公网上不存在**，只在校园网/公司内网里能到达。
- 所以你在家、在咖啡馆、用 4G 热点，**都连不上**，除非连上学校 VPN。

**症状**：昨天还好好的，今天突然 `Connection timed out`，
但打开百度/Google 都正常。

**怎么判断是 VPN 问题还是服务器挂了**：

```bash
# 1. 通不通
nc -z -v 10.176.54.32 13811          # macOS 自带的 nc，-z 只探测端口不传数据
#    通了 → Connection succeeded

# 2. 看本机有没有 VPN 隧道
ifconfig | grep -A3 utun             # macOS 的 VPN 接口叫 utun*
#    如果有 utun0 且带 inet 10.x 地址 → VPN 连着
```

**别慌**：训练是用 `setsid nohup` 脱离 SSH 启动的，
**你的网络断了不影响它**。你只是「看不见」了，它还在跑。

---

## 7. 共享机器的礼仪与陷阱

### 7.1 不要杀别人的进程 ⚠️

```bash
# ❌ 绝对不要
pkill -f python
killall python
```

这台机器上跑着**别人的实验**。`pkill python` 会把所有人的训练全杀掉。
本项目踩过这个原则，所以所有清理命令都写成**精确匹配自己的脚本特征**：

```bash
pkill -f "[t]rain_sft.py"       # ✅ 只杀自己的 SFT
pkill -f "[w]atch_sft.sh"       # ✅ 只杀自己的守护脚本
```

### 7.2 容器里的 PID 和外面不一样

本项目的服务器是一台 Docker 容器。在容器里跑 `nvidia-smi` 时，
它列出的进程 PID 是**宿主机**（外面那台真机器）的 PID：

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv
pid, used_gpu_memory [MiB]
3847291, 12046 MiB            ← 这个 PID 在容器里 /proc/3847291 是不存在的！
```

```bash
$ ls /proc/3847291            # No such file or directory
```

**别以为进程死了**。要找自己的进程，用 `pgrep`（它看的是容器内的进程表）：

```bash
pgrep -af "train_sft"         # ← 这个才准
```

### 7.3 磁盘：哪个目录能写

服务器上不是所有地方都能写、都有空间：

| 位置 | 大小 | 能写吗 | 用途 |
|---|---|---|---|
| `/`（系统盘，overlay） | 48 GB 空闲 | 能写，但**别放数据** | 系统、临时文件 |
| `/root/Test` | 203 GB | ✅ 能写 | **项目主盘，代码+数据+模型都在这** |
| `/root/Data` | — | ❌ 只读 | 别人挂载的公共数据 |
| `/workspace` | — | 视机器而定 | 有些环境会误建在这 |

所以本项目的路径全部写死成 `/root/Test/mini-r1/...`。
这也解释了为什么 [server_setup.sh](../scripts/server_setup.sh) 里装 conda 要显式
指定 `-p /root/Test/mini-r1/miniforge/envs/mini-r1` ——
系统的 conda 配置把环境目录指向了只读盘，用 `conda create -n xxx` 会失败。

### 7.4 macOS 传上去的 `._` 文件

如果你在 macOS 上用 `tar` 或 Finder 传过文件，服务器上会出现一堆 `._AGENTS.md`、
`._src` 这种怪文件：

```
-rw-r--r--  1 501 staff  163 Oct 12 14:28 ._.
-rw-r--r--  1 501 staff  163 Oct 12 14:28 ._AGENTS.md
```

这是 macOS 的 **AppleDouble** 格式 —— 它把「扩展属性」（图标位置、
「从哪里下载的」之类）单独存成一个小文件。在 Linux 上完全没用，是垃圾。

处理：同步时加 `--exclude='._*'`，或者在服务器上 `find . -name '._*' -delete`。
本项目 [AGENTS.md](../AGENTS.md) 里没有专门记这条，但服务器上确实有一批。

---

## 8. 命令速查

```bash
# ---------- 连接 ----------
ssh root@10.176.54.32 -p 13811              # 直接登录（交互式）
~/.server_cmd.sh '<命令>'                    # 执行一条命令就返回
~/.server_sync.sh <本地目录> <远程目录>       # 同步代码（两个参数！）
scp -P 13811 <本地文件> root@<host>:<远程路径>  # 传单个文件（注意大写 P）

# ---------- 看状态 ----------
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'   # ★ 最常用
~/.server_cmd.sh 'tail -3 /root/Test/mini-r1/logs/sft.log'
~/.server_cmd.sh 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
~/.server_cmd.sh 'pgrep -af "[t]rain_sft.py"'

# ---------- 启动 / 停止 ----------
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/run_sft.sh'
~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/watch_sft.sh'
~/.server_cmd.sh 'pkill -f "[w]atch_sft.sh"'      # 只停守护，不停训练

# ---------- 磁盘 ----------
~/.server_cmd.sh 'df -h /root/Test'
~/.server_cmd.sh 'du -sh /root/Test/mini-r1/*'
```

---

下一篇 → [03-环境搭建](03-环境搭建.md)：从一台空机器开始装环境。
