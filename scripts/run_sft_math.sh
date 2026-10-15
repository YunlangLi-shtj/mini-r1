#!/bin/bash
# GSM8K-CoT 热身 SFT: 从 SFT 模型接着训 GSM8K 训练集的解题过程 (约 15 分钟)。
#
# 为什么要有这一段 —— 见 AGENTS.md §8 第 1 条:
#   基座在 GSM8K 上 pass@1 = 0, 而 GRPO 靠「组内相对优势」学习, 8 条全错 = 优势全 0 =
#   这组白采。RL 只能放大基座已经偶尔做对的行为, 不能无中生有。
#   2026-10-14 那次 300 步 GRPO 的实测证实了这一点: 它一步数学分都没学到,
#   唯一被强化的是那个**可以从提示词里照抄**的格式分。
#
# ⚠️ 顺序: 必须先修好 format_reward (已修, 见 scripts/test_reward_fn.py),
#    否则热过身之后模型会抄得更起劲 —— 奖励可 hack 的话, 基座越强它抄得越准。
#
# 用法: bash run_sft_math.sh [base_ckpt] [epochs] [out_dir]
#   默认 ckpts/sft/sft.pt / 1 / ckpts/sft_math
#
# ★ 为什么 epoch 是个参数 (2026-10-14 实测之后加的):
#   1 个 epoch (233 步) **不够** —— 贪心 pass@1 只有 3/256, 采样口径前 50 题 pass@1 0.0000 /
#   maj@8 0.02。原始输出给出了原因: 模型学会了「数学解题的句子长什么样」但没学会算
#   (`80+50=100` 这种形状对、结果不对的句子), 217/256 的格式合规率是学到了 shell,
#   壳里没有东西。1 epoch × 7473 条对 CoT 烘焙来说本来就远低于常规 (3~6 轮)。
#   RL 只能放大已经偶尔做对的行为 —— 基座停在 ~1% 的话, 8 条里至少一条对的概率只有 ~8%,
#   也就是 92% 的组优势恒为 0, GRPO 照样学不到东西 (见 AGENTS.md §8 第 1 条)。
#
# ★ 要换 epoch 数请**从 base 重跑, 别在原目录上改 --epochs 续训**:
#   train_sft.py 的 cosine 是 cos(π·step/total_steps), total_steps=每轮步数×epochs。
#   改大 epochs 会让 LR 从退火末端的 ~0 直接跳回 1.9e-5 (暖重启), 把上一轮已经
#   收敛到的位置顶掉。重跑一遍成本相同 (都是「总轮数 − 已训轮数」的工作量), 但
#   LR 走一条完整的 warmup→退火曲线。产物换个目录, 旧的那版留着当对照。
set -euo pipefail

PROJ=/root/Test/mini-r1
BASE="${1:-$PROJ/ckpts/sft/sft.pt}"
EPOCHS="${2:-1}"
OUTDIR="${3:-$PROJ/ckpts/sft_math}"
[ -f "$BASE" ] || { echo "找不到 SFT 检查点: $BASE"; exit 1; }

DATA=$PROJ/data/sft_math.json
[ -f "$DATA" ] || {
  echo "缺少热身数据 $DATA, 先生成:"
  echo "  python src/make_sft_math.py --gsm8k_dir data/raw/gsm8k --out data/sft_math.json"
  exit 1
}

# 门槛同 run_sft.sh: 这套档位 (mb1+grad_ckpt) 实测约 10.6G ≈ 10,850 MiB (驱动器口径)
WAIT_MAX=600    # 这是个 15 分钟的短活, 等太久不如让人看一眼; 不像 SFT 那样等 30 分钟
FREE=""; waited=0
while [ -z "$FREE" ]; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$3 - $2 > 12000 && $4 < 80 {printf "%d,", $1}' | sed 's/,$//')
  if [ -z "$FREE" ]; then
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      echo "等了 ${WAIT_MAX}s 仍没有空闲 GPU:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
      exit 1
    fi
    [ "$waited" -eq 0 ] && echo "暂无空闲卡, 每 30s 重试 (最多 ${WAIT_MAX}s)..."
    sleep 30; waited=$(( waited + 30 ))
  fi
done
export CUDA_VISIBLE_DEVICES="${FREE%%,*}"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES (等待 ${waited}s)"

# 日志跟着 out_dir 走: 默认仍是 logs/sft_math.log, 换目录就是 logs/sft_math6.log,
# 免得两次不同配置的记录混在一个文件里 (守护脚本那套按字节偏移判断的逻辑也怕混)。
LOG=$PROJ/logs/$(basename "$OUTDIR").log

mkdir -p "$OUTDIR" $PROJ/logs
cd $PROJ
echo "===== [$(date '+%m-%d %H:%M')] 启动 GSM8K-CoT 热身 (GPU ${FREE%%,*}, base=$BASE, ${EPOCHS} epochs) =====" \
  >> $LOG

# 7473 条 / (mb1 × ga32) = 233 个优化器步/轮, 约 16 分钟/轮 (1 轮 ≈ 16 分钟, 6 轮 ≈ 96 分钟)。
#   --max_len 2048 富余很多: 这条数据的 assistant 轮平均只有 252 字符 (~100 token)。
#   --lr 2e-5 和主 SFT 同档。
#   --warmup_steps 20 (不是 100): 步数少, 拿 100 步热身等于一半时间在爬坡。
#   --ckpt_every 100: 每 100 步原子写一次 sft_latest.pt (4.9 GB, 约半分钟)。6 轮下来
#       约 7% 的墙钟开销, 换「被抢占最多丢 7 分钟」—— 和主 SFT 一样的取舍。
# ★ out_dir 默认 ckpts/sft_math, 不是 ckpts/sft —— 别把主 SFT 的检查点覆盖了。
setsid nohup $PROJ/miniforge/envs/mini-r1/bin/python -u src/train_sft.py \
  --base_ckpt "$BASE" \
  --tokenizer $PROJ/tokenizer/tokenizer.json \
  --data_file "$DATA" \
  --out_dir "$OUTDIR" \
  --max_samples 7473 --max_len 2048 \
  --micro_batch 1 --grad_accum 32 --grad_ckpt 1 \
  --lr 2e-5 --epochs "$EPOCHS" --warmup_steps 20 \
  --ckpt_every 100 \
  >> $LOG 2>&1 < /dev/null &
echo "热身 SFT 已启动 (${EPOCHS} epochs), 日志: $LOG"
echo "  产物: $OUTDIR/sft.pt   (GRPO 从这里起跑)"
