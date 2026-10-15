#!/bin/bash
# GRPO 启动: 自动挑空闲 GPU, 从 SFT 检查点做强化学习 (策略 + 参考模型各一份)
#
# 用法: bash run_grpo.sh [sft_ckpt] [max_steps] [n_gpu] [tag] [format_w]
#
#   tag      产物后缀: ckpts/grpo<tag>/ + logs/grpo<tag>.log。默认空 = 老路径。
#   format_w 格式分权重, 默认 1.0; 传 0 就是"只留数学分"的奖励消融臂。
#   例: bash run_grpo.sh ckpts/sft_math6/sft.pt 300 1 _lowfmt 0
#
# n_gpu=1 (默认) 单卡; n_gpu=2 用 torchrun 起 DDP, 每张卡负责一半的 prompt。
# ★ 多卡不是我加个参数就有的 —— train_grpo.py 的 DDP 支持是 2026-10-14 补的:
#   rollout 按 rank::world 交错切题、梯度更新走 DDP 包住的 policy、
#   推理路径 (采样/评测) 走裸模型 net、存检查点也只存 net。
#   单卡那条路一行没动, world=1 时逐行行为与以前完全一致。
#
# ⚠️⚠️ **但多卡对 GRPO 来说不是加速, 是等价交换 —— 别指望 2 卡跑一半时间。**
#   这是读完更新循环 (train_grpo.py:444-454) 之后算出来的, 和预训练那边
#   «2 卡 1.77×» 的结论**不通用**, 原因在于两件事:
#     ① GRPO 的更新是**逐组**做的: `for j, (...) in enumerate(samples)` 里
#        每组各来一次 zero_grad/backward/step。一步 = rollout_batch × update_epochs
#        = 24 × 2 = **48 次参数更新** (每次只用 8 条 completion 的梯度)。
#     ② DDP 下每张卡只拿到 rollout_batch/world 组, 于是每步的更新次数也变成
#        48/world。2 卡 = 24 次。
#   而墙钟上 rollout 和 update **两段都**随 world 减半 (rollout 实测占一步的 ~84%)。
#   于是:
#       单位时间的参数更新次数 = (3600/每步秒数) × (48/world), 而每步秒数 ∝ 1/world
#                              = 3600 × 48 / (rollout + update)   ← **与 world 无关**
#   2 卡省掉的是墙钟, 代价是「48 次单组更新」变成「24 次双组平均更新」——
#   优化器语义变了, 而单位时间内的优化进度一模一样。它买到的是**吞吐**(每小时见过的
#   prompt 数 × world), 不是**收敛速度**。
#   ⇒ 所以这一轮正式 GRPO 仍然用**单卡**: 既不多花时间, 又能和单卡跑出来的
#     300 步负对照逐行对齐 (卡数一变, 学习率曲线/更新次数的口径就全对不上了)。
#   ⇒ 想让第二张卡真正变成加速, 得先改更新循环 (把 48 次逐组更新合成「整个 batch
#     一次 loss、一次 step」+ grad_accum), 那才是标准 GRPO 的写法, 也是多卡能
#     真正提速的前提。但那会让负对照再多一处不可比, 所以不在这一轮做。
set -euo pipefail

PROJ=/root/Test/mini-r1
BASE="${1:-$PROJ/ckpts/sft/sft.pt}"
STEPS="${2:-300}"
NGPU="${3:-1}"
# TAG: 给这一轮起个后缀, 把产物挪到 ckpts/grpo$TAG 和 logs/grpo$TAG.log。
#   默认空串 ⇒ 路径和以前一模一样, 老调用方零影响 (两条臂并行时才用得上)。
TAG="${4:-}"
FW="${5:-1.0}"          # 格式分权重: 1.0 = 原样, 0 = 关掉格式奖励 (奖励消融臂)
OUT=$PROJ/ckpts/grpo$TAG
LOG=$PROJ/logs/grpo$TAG.log
[ -f "$BASE" ] || { echo "找不到 SFT 检查点: $BASE"; exit 1; }

# ---- 显存门槛 (MiB 空闲, nvidia-smi 口径；不能信 PyTorch 自报数, 见 AGENTS.md §7 第 1 条) ----
# 单卡 19000: 驱动器口径实测峰值 17,718 MiB, 而 PyTorch 只报 14.0 GiB, 差 3.7 GB
#   (fp32 的 logits 2.43 GB × 2 + expandable_segments 的池子)。门槛只保 16000 的话,
#   最坏情况(空闲刚好 16000)一跑到更新那步就 OOM, 然后守护脚本把它变成 3 分钟一次的
#   崩溃循环 (2026-10-13 连崩 78 次, 见 §7 第 18 条)。
# 双卡 21000: 单卡那 17.7 GB 之上还要多一份 DDP 的 fp32 梯度桶 (407M 参数 ≈ 1.6 GB),
#   再留一点余量。宁可偏高 —— 选不出卡只是等 60 秒重试, 比崩溃循环便宜得多。
MIN1=19000
MIN2=21000

# ★ 计数用 grep -c '[0-9]' 而不是 wc -l: printf 不补结尾换行, 单张卡时 wc -l 会返回 0,
#   再拿去做除法就是 division by 0, set -e 直接让启动脚本退出 (见 AGENTS.md §7 第 11 条)。
gpus_over() {   # $1 = 门槛 MiB → 可用卡列表 "0,2"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' -v t="$1" '$3 - $2 > t && $4 < 80 {printf "%s,", $1}' | sed 's/,$//'
}
n_of() { [ -z "${1:-}" ] && echo 0 || printf '%s' "$1" | tr ',' '\n' | grep -c '[0-9]'; }
first_n() { printf '%s' "$1" | tr ',' '\n' | head -n "$2" | paste -sd, -; }

# ---- 选卡: 优先凑够用户要的卡数, 凑不齐就退回单卡 (和 run_pretrain.sh 一个策略) ----
USE=$NGPU
GPUS=""
if [ "$NGPU" -gt 1 ]; then
  GPUS=$(gpus_over "$MIN2")
  if [ "$(n_of "$GPUS")" -lt "$NGPU" ]; then
    echo "只有 $(n_of "$GPUS") 张卡能满足多卡门槛 (${MIN2} MiB 空闲), 退回单卡"
    USE=1; GPUS=""
  fi
fi
if [ "$USE" -eq 1 ]; then
  GPUS=$(gpus_over "$MIN1")
  if [ -z "$GPUS" ]; then
    echo "没有空闲 GPU (GRPO 实测峰值 17.7 GB, 门槛 ${MIN1} MiB):"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 1
  fi
  GPUS="${GPUS%%,*}"          # 单卡只取第一张
else
  GPUS=$(first_n "$GPUS" "$USE")
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES ($USE 卡)"

mkdir -p $OUT $PROJ/logs
cd $PROJ
# ★ expandable_segments: rollout 每生成一个 token 都要 torch.cat 一次不断变长的
#   KV-Cache / 序列, 384 步下来分配器会碎成一地: 实测**峰值 reserved 22.54 GiB 而
#   allocated 只有 2.24 GiB**, 在 24 GB 的共享卡上等于自己独占一整张。
#   打开这个选项后 reserved 2.27 GiB (几乎等于 allocated), 速度不变。
#   (测这个数字必须单开一个进程 —— 同一个进程里先跑别的形状, 池子已经被撑大了。)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ★ 追加而不是覆盖 (>> 不是 >): watch_grpo.sh 靠「本次启动时的字节偏移」判断
#   「训练完成」是本次还是上次。截断模式会把偏移全作废, 分不清。见 AGENTS.md §7 第 6 条。
echo "===== [$(date '+%m-%d %H:%M')] 启动 GRPO (GPU $CUDA_VISIBLE_DEVICES, ${USE} 卡, format_w=$FW) =====" >> $LOG

PY=$PROJ/miniforge/envs/mini-r1/bin/python
ARGS=(
  --base_ckpt "$BASE"
  --tokenizer $PROJ/tokenizer/tokenizer.json
  --gsm8k_dir $PROJ/data/raw/gsm8k
  --out_dir $OUT
  --G 8 --rollout_batch 24 --max_new 384
  --beta 0.04 --clip_eps 0.2 --lr 5e-6 --max_steps $STEPS
  --format_w $FW
)
if [ "$USE" -eq 1 ]; then
  setsid nohup $PY -u src/train_grpo.py "${ARGS[@]}" \
    >> $LOG 2>&1 < /dev/null &
else
  # ★ 端口别用默认 29500: 同机上另一个 torchrun (比如还在跑的预训练/bench) 会 EADDRINUSE。
  #   ★ 两条臂并行时端口也得错开, 否则第二个 torchrun 起来就 EADDRINUSE。
  PORT=$((29513 + $(printf '%s' "$TAG" | cksum | cut -d' ' -f1) % 50))
  setsid nohup $PROJ/miniforge/envs/mini-r1/bin/torchrun --standalone \
    --nproc_per_node="$USE" --master_port=$PORT \
    src/train_grpo.py "${ARGS[@]}" \
    >> $LOG 2>&1 < /dev/null &
fi
echo "GRPO 已启动, 日志: $LOG  检查点: $OUT/"
