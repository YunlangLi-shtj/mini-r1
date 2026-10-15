#!/bin/bash
# GRPO 守护: 进程被杀 (共享机器上别人抢卡吃显存) 后自动从检查点续训。
#
# 和 watch_sft.sh 一个套路, 不做换卡逻辑。续训靠 grpo_latest.pt
# (train_grpo.py 每 5 步原子写一次), 所以被抢占最多损失 5 步的 rollout。
#
# ★ 参数原样转发给 run_grpo.sh —— 尤其 n_gpu:
#   卡数决定了每个优化器步平均几个样本 (单卡=逐样本 SGD, 双卡=两份梯度取平均),
#   两边混着跑等于中途偷偷换了优化器, 曲线就没法解释了。守护重启必须用同一个卡数。
#
# 用法: setsid nohup bash scripts/watch_grpo.sh [sft_ckpt] [max_steps] [n_gpu] [tag] [format_w] >/dev/null 2>&1 &
# 停:   pkill -f "[w]atch_grpo.sh"   (只杀守护, 训练进程不动)
#       ★ 只有一条臂时这样杀没问题; 两条臂并行时它会**同时杀掉两个守护**,
#         要单独停一条得按 TAG 认: pkill -f "[w]atch_grpo.sh .* _lowfmt"
PROJ=/root/Test/mini-r1
cd $PROJ

# TAG: 只有同时跑两条 GRPO 臂时才用得上 (第 4 个位置参数, 空 = 老路径)。
# ★ 有了它之后, pid 文件 / 守护日志 / 训练日志 / 检查点目录 / **进程识别** 全都得带上它。
#   漏掉任何一处的症状都是静默的, 最典型的是 running() 认到另一条臂的进程 ——
#   于是自己这条死了它以为还在跑, 永远不拉起来 (不报错, 进度就停在那)。
TAG="${4:-}"
DIR=$PROJ/ckpts/grpo$TAG
LOG=$PROJ/logs/grpo$TAG.log

# 单实例保护 (pid 文件; 不用 flock 的原因见 watch_pretrain.sh 的注释)
PIDF=$PROJ/logs/.watch_grpo$TAG.pid
if [ -f "$PIDF" ]; then
  OLD=$(cat "$PIDF" 2>/dev/null)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q "watch_grpo.sh"; then
    echo "已有 GRPO 守护在跑 (pid $OLD), 退出"; exit 1
  fi
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

WLOG=$PROJ/logs/grpo${TAG}_watch.log
log() { echo "[watch $(date '+%m-%d %H:%M')] $*" >> $WLOG; }
# 按 --out_dir 认**自己这条臂**的进程。末尾那个空格是必须的:
# 没有它, "[--]out_dir .../ckpts/grpo" 会连 .../ckpts/grpo_lowfmt 一起匹配上。
running() { pgrep -f -- "[t]rain_grpo.py .*--out_dir $DIR " > /dev/null; }

# 已经跑完过就别守着 ("GRPO 训练完成" = 跑满 max_steps; "已训满" = 重启时步数已经够了)。
# ★ 只看**最后一次启动之后**那一段, 不是整份日志 —— 日志是追加模式 (AGENTS.md §7 第 6 条),
#   历次记录都在里面。重跑时上一轮留下的"训练完成"会让守护**当场退出**,
#   而你以为它在守着 (守护不在 = 被抢占后没人拉起来 = 进度停在那没人知道)。
#   2026-10-14 重跑 GRPO 时发现的: 不手工挪走 logs/grpo.log 就必然踩到。
#   另外两个守护没这个问题 —— 它们只在循环里按「本次启动的字节偏移」判断。
if [ -f "$LOG" ]; then
  last_start=$(grep -n "启动 GRPO" "$LOG" | tail -1 | cut -d: -f1)
  if [ -n "$last_start" ] && tail -n "+$last_start" "$LOG" | grep -qE "GRPO 训练完成|已训满"; then
    log "最后一次启动已经跑完, 守护不启动"; exit 0
  fi
fi

# 偏移取「守护启动那一刻」的日志长度, 不是 -1:
# 被守护的那个训练可能是守护启动之前就跑着的, 它完成时写下的完成行
# 也在偏移之后, 所以照样能认出来。取 -1 反而会把历史记录里的完成误判成本次。
off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
log "GRPO 守护启动 (pid $$), 日志偏移 $off"
while true; do
  if ! running; then
    # 完成了就别再拉起来 (日志是追加模式, 所以要按偏移区分是哪一次的"完成")
    if [ "$off" -ge 0 ] && tail -c "+$((off + 1))" "$LOG" 2>/dev/null \
       | grep -qE "GRPO 训练完成|已训满"; then
      log "GRPO 已正常结束, 守护退出"; exit 0
    fi
    off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    out=$(bash scripts/run_grpo.sh "$@" 2>&1) && rc=0 || rc=$?
    echo "$out" >> $WLOG
    if [ $rc -ne 0 ]; then
      # 没卡 / 没检查点: 空出来的卡很快会被别人抢走, 盯紧一点
      log "启动失败 (rc=$rc), 1 分钟后重试"
      sleep 60; continue
    fi
    log "已拉起 GRPO: $(echo "$out" | grep '使用 GPU' | tail -1)"
  fi
  # 在跑: 60 秒巡一次, 进程一死马上回到循环顶部重启
  for _ in $(seq 1 5); do
    sleep 60
    running || break
  done
done
