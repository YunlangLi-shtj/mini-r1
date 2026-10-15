#!/bin/bash
# GRPO 冒烟探针: 在押上 ~12 小时之前, 先跑 5 步看「有信号组」。
#
# 用法: bash scripts/probe_grpo.sh <base_ckpt> [steps]
#   例: bash scripts/probe_grpo.sh /root/Test/mini-r1/ckpts/sft_math6/sft.pt
#
# 为什么要有它 (2026-10-14):
#   一次 300 步 GRPO 是 12 小时。跑之前真正该知道的只有**一个**数 ——
#   「有信号组 n/24」= 这一步 24 组里有多少组的 8 条奖励**有差异**。
#   组内奖励全一样 → 优势按定义全是 0 → 这组数据白采、一点梯度都不产生。
#   这个数低, 后面 299 步基本是白跑。探针 5 步约 13 分钟, 拿它换 12 小时的确定性。
#
# ★ 读出来的两个数怎么判 (2026-10-14 在 sft_math 基座上实测过一次):
#
#   ① r (原始奖励) 逼近 0.1 = 奖励几乎**全是格式分**, 数学分没进来。
#      格式满分是 0.1, 数学分是 1.0。所以 r≈0.1 意味着"8 条里基本没有做对的"。
#      实测 sft_math 基座: r≈0.096 ⇒ 反推每条采样的格式合规率约 86%
#      (1.0×1.03% + 0.1×86% ≈ 0.096)。健康的样子是 r **明显高于 0.1**。
#
#   ② 有信号组 /24 要拿去和 **pass@8** (8 条里至少一条对) 比:
#        - 两者接近  → 信号基本全来自数学, 是好局面;
#        - 前者远大于后者 → 信号大半来自**格式分在组内没饱和**。
#          这不是第 19 条那个 hack (占位符已经判 0 了), 但它是同一类东西的近亲:
#          「把盒子写满」仍然是一种不用真做题就能拿到的分。实测那次
#          16/24 = 67% 而 pass@8 只有 7.8% ⇒ 约 87% 的梯度在推格式, 不在推数学。
#          这时**别急着跑 300 步** —— 先把格式分在基座上喂饱 (多训几轮 SFT),
#          或者考虑这一轮把格式权重降到 0, 让信号全部来自数学。
#
#   (峰值显存也顺手量了: 实测 13.4~13.8 GiB (PyTorch 口径), 和正式跑的 14.0 一致,
#    说明探针的显存特性和正式跑一样, 拿它验门槛是有效的。)
set -euo pipefail

PROJ=/root/Test/mini-r1
BASE="${1:-$PROJ/ckpts/sft_math6/sft.pt}"
STEPS="${2:-5}"
[ -f "$BASE" ] || { echo "找不到基座检查点: $BASE"; exit 1; }

# ★ 用**独立的 out_dir**: 探针绝不能碰 ckpts/grpo/ ——
#   它跑完会存一次检查点 (--ckpt_every 挡不住收尾那次, 和 SFT 一个道理),
#   落进 ckpts/grpo/ 就会让正式跑以为"已经训过 5 步"。
OUT=$PROJ/ckpts/grpo_probe
LOG=$PROJ/logs/grpo_probe.log
[ -e "$OUT" ] && { echo "先清掉旧的探针目录: $OUT"; rm -rf "$OUT"; }

# 门槛同 run_grpo.sh 单卡: 驱动器口径峰值 17.7 GB ⇒ 19000 MiB 空闲 (见 AGENTS.md §7 第 18 条)
FREE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 - $2 > 19000 && $4 < 80 {printf "%s,", $1}' | sed 's/,$//')
if [ -z "$FREE" ]; then
  echo "没有能满足门槛 (19000 MiB 空闲) 的卡:"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${FREE%%,*}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 见 AGENTS.md §7 第 17 条
echo "冒烟探针: GPU $CUDA_VISIBLE_DEVICES, 基座 $BASE, ${STEPS} 步 (约 $((STEPS * 150 / 60 + 3)) 分钟)"

cd $PROJ
mkdir -p $PROJ/logs
# --eval_every / --ckpt_every 都设成比 STEPS 大: 探针不做 256 题评测 (那要 4~5 分钟),
# 也不存中间检查点。收尾那次存盘是挡不住的, 所以跑完必须 rm -rf $OUT。
set +e
$PROJ/miniforge/envs/mini-r1/bin/python -u src/train_grpo.py \
  --base_ckpt "$BASE" \
  --tokenizer $PROJ/tokenizer/tokenizer.json \
  --gsm8k_dir $PROJ/data/raw/gsm8k \
  --out_dir "$OUT" \
  --G 8 --rollout_batch 24 --max_new 384 \
  --beta 0.04 --clip_eps 0.2 --lr 5e-6 \
  --max_steps "$STEPS" --eval_every 100000 --ckpt_every 100000 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if grep -q "Traceback" "$LOG"; then
  echo; echo "❌ 探针崩了, 看 $LOG"; grep -A 5 "Traceback" "$LOG" | tail -20; exit 1
fi

# ★ 收尾那次存盘会留下 4.9 GB —— 冒烟产物要随手删。
#   根盘曾经被占满到 datasets 拒绝一切新文件 (AGENTS.md §7 第 20 条),
#   而 4.9 GB 一个攒几次就是几十 G。
[ -f "$OUT/grpo_log.jsonl" ] && cp "$OUT/grpo_log.jsonl" $PROJ/logs/grpo_probe_log.jsonl
rm -rf "$OUT"
echo; echo "探针跑完 (rc=$rc), 逐指标留在 logs/grpo_probe_log.jsonl, 已清掉 $OUT"
echo
echo "★ 判读 (详见本脚本顶部注释):"
echo "   - r 逼近 0.1 且没有明显上探 ⇒ 奖励几乎全是格式分, 数学分没进来"
echo "   - 有信号组 /24 远大于 pass@8 ⇒ 大半梯度在推格式, 不在推数学"
