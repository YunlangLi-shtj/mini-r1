#!/bin/bash
# 训练状态速查 —— 一条命令看清「跑没跑 / 到哪了 / 还要多久 / 卡怎么样」
#
# 在服务器上:  cd /root/Test/mini-r1 && bash scripts/status.sh
# 在本地:      ~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'
cd /root/Test/mini-r1 || exit 1

echo "════════ Mini-R1 状态  $(date '+%m-%d %H:%M') ════════"

# ---------- 1. 进程 ----------
train=$(pgrep -cf "[t]rain_pretrain.py")
watch=$(pgrep -cf "[w]atch_pretrain.sh")
if [ "$train" -gt 0 ]; then
  echo "训练进程: 在跑 (共 $train 个: torchrun + 训练进程)"
else
  echo "训练进程: ⚠️  不在跑"
fi
if [ "$watch" -gt 0 ]; then
  echo "守护进程: 在跑 —— 训练被杀后会自动重选卡续训"
else
  echo "守护进程: ⚠️  不在跑 —— 训练一旦被打断就没人拉起来了!"
  echo "           拉起: cd /root/Test/mini-r1 && bash scripts/run_watch.sh"
fi

# ---------- 2. 训练进度 + 预计剩余 ----------
echo
echo "── 进度 (logs/pretrain.log 最后一行) ──"
last=$(grep "^step" logs/pretrain.log 2>/dev/null | tail -1)
if [ -n "$last" ]; then
  echo "$last"
  # 从 "step 481/38146 | 0.06B tok | loss 5.4 | lr 2.9e-04 | 15,965 tok/s | 9.61G" 里抠步数与吞吐。
  # 按 "|" 切段后找含 "tok/s" 的那段 —— 不能用 $NF 或贪婪匹配, 末尾的 "9.61G" 是显存不是吞吐。
  echo "$last" | awk '{
    split($2, st, "/"); step = st[1] + 0; total = st[2] + 0;
    n = split($0, f, "|"); tps = 0;
    for (i = 1; i <= n; i++) if (f[i] ~ /tok\/s/) { t = f[i]; gsub(/[^0-9]/, "", t); tps = t + 0 }
    if (tps > 0 && total > 0) {
      remain_s = (total - step) * 131072 / tps;       # 每步 131072 token
      printf "剩余: %d 步 ≈ %.1f 天 (按当前 %d tok/s)\n", total - step, remain_s / 86400, tps;
    } else print "  (吞吐解析失败, 跳过剩余时间估算)"
  }'
else
  echo "  (日志里还没有 step 行)"
fi

# ---------- 3. 结构化指标 / 验证集 ----------
echo
echo "── 最新指标 (ckpts/pretrain/train_log.jsonl) ──"
tail -1 ckpts/pretrain/train_log.jsonl 2>/dev/null || echo "  (无)"
vals=$(grep val_loss ckpts/pretrain/train_log.jsonl 2>/dev/null | tail -3)
if [ -n "$vals" ]; then
  echo "── 验证集 loss (最近 3 次) ──"
  echo "$vals"
fi

# ---------- 4. 检查点 ----------
echo
echo "── 检查点 ──"
if [ -f ckpts/pretrain/latest.pt ]; then
  ls -lh ckpts/pretrain/latest.pt | awk '{print "  latest.pt  "$5"  最后写入 " $6" "$7" "$8}'
  echo "  已有检查点 → 重启会自动 --resume 续训 (绝不从头再来)"
else
  echo "  ⚠️  还没有检查点 (每 200 步存一次, 约 45 分钟)"
fi

# ---------- 5. GPU ----------
echo
echo "── GPU (本机 6 张, 共享机器) ──"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
  | awk -F', ' '{
      free = $3 - $2;
      mark = (free > 11800) ? "  <- 可用 (预训练门槛 11800 MiB)" : "";
      printf "  GPU %s  %6d / %6d MiB  已用%s  空闲 %5d MiB%s\n", $1, $2, $3, $4, free, mark;
    }'
echo
echo "提示: 训练想用更多卡时, 守护脚本会自动等到有卡空出来再重启换卡, 不用手动干预"
