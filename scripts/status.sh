#!/bin/bash
# 训练状态速查 —— 一条命令看清「跑没跑 / 到哪了 / 还要多久 / 卡怎么样」
#
# 在服务器上:  cd /root/Test/mini-r1 && bash scripts/status.sh
# 在本地:      ~/.server_cmd.sh 'cd /root/Test/mini-r1 && bash scripts/status.sh'
#
# 会自动认预训练 / SFT / GRPO 哪个阶段在跑 —— 不要写死成预训练, 否则跑 SFT 时
# 这一屏会报「训练进程不在跑」(假警报, 见下), 进度那段读的也是预训练的旧日志。
cd /root/Test/mini-r1 || exit 1

echo "════════ Mini-R1 状态  $(date '+%m-%d %H:%M') ════════"

# ---------- 0. 在跑哪个阶段 ----------
# 有进程在跑的优先; 都没在跑就按日志的修改时间挑最新的, 至少能报出「上次跑到哪」。
detect_stage() {
  for s in pretrain sft grpo; do
    [ "$(pgrep -cf "[t]rain_${s}.py")" -gt 0 ] && { echo "$s"; return; }
  done
  ls -t logs/pretrain.log logs/sft.log logs/grpo.log 2>/dev/null \
    | head -1 | xargs -r basename | sed 's/\.log$//'
}

STAGE=$(detect_stage)

# GRPO 可以同时跑多条臂 (run_grpo.sh 的第 4 个参数是 TAG 后缀, 空 = 主臂)。
# ★ 这里必须逐条认出来, 不能只看 logs/grpo.log —— 两条臂并行时
#   「只报一条」和「报错阶段」一样糟: 另一条死了、或者压根没起来, 屏幕上完全看不出来
#   (AGENTS.md §7 第 13 条: 假警报比没有警报更糟)。
_grpo_arm_suffixes() {   # 每行一个后缀, 主臂是空行
  # awk 逐行取 --out_dir, 两种写法都认 ("--out_dir <path>" 和 "--out_dir=<path>")。
  # 用 grep -oE + sed 也能得到一样的结果 (grep -oE 本来就一行一个匹配, sed 的贪婪
  # 只在那**一行**里贪婪) —— 我一度以为那种写法会把两条臂并成一条, 是看错了:
  # 真正的输出被我自己在后面接的 tail -1 截掉了。这里用 awk 纯粹是因为
  # "从一行里挑出某个选项的值" 是它擅长的事, 顺带不用假设参数一定带空格。
  pgrep -af "[t]rain_grpo.py" 2>/dev/null | awk '
    {
      tag = ""
      for (i = 1; i <= NF; i++)
        if ($i ~ /^--out_dir/) {
          p = $(i + 1);                      # "--out_dir <path>" 两个字段
          sub(/^--out_dir=/, "", $i)         # 或者 "--out_dir=<path>" 一个字段
          if (p ~ /^--/) p = $i
          sub(/.*\/ckpts\/grpo/, "", p)      # 剥掉公共前缀, 剩下的就是后缀
          tag = p
        }
      print tag
    }' | sort -u
}
_grpo_arm_names() {      # 一行, 给"训练进程"那句话用
  _grpo_arm_suffixes | sed 's/^$/(主)/' | tr '\n' ' '
}

case "$STAGE" in
  pretrain) NAME=预训练; LOG=logs/pretrain.log; WLOG=logs/pretrain_watch.log
            CKDIR=ckpts/pretrain; CKPT=latest.pt;     JLOG=ckpts/pretrain/train_log.jsonl
            THRESH=13000; MINS="200 步, 约 45 分钟" ;;
  sft)      NAME=SFT;    LOG=logs/sft.log;      WLOG=logs/sft_watch.log
            CKDIR=ckpts/sft;      CKPT=sft_latest.pt; JLOG=""
            THRESH=12000; MINS="100 步, 约 7 分钟" ;;
  grpo)     NAME=GRPO;   LOG=logs/grpo.log;     WLOG=logs/grpo_watch.log
            # ★ 这里认 grpo_latest.pt 而不是 grpo.pt: 后者每 25 步才写一次,
            #   刚起步 (step < 25) 时它根本还不存在, 报「还没有检查点」是**假警报**——
            #   而 grpo_latest.pt 每 5 步就写, 那才是"有没有在存"的真实信号。
            CKDIR=ckpts/grpo;     CKPT=grpo_latest.pt; JLOG=ckpts/grpo/grpo_log.jsonl
            THRESH=18000; MINS="25 步 (grpo.pt) / 5 步 (grpo_latest.pt)" ;;
  *)        NAME="?" ;   LOG=logs/pretrain.log; WLOG=logs/pretrain_watch.log
            CKDIR=ckpts/pretrain; CKPT=latest.pt;     JLOG=ckpts/pretrain/train_log.jsonl
            THRESH=13000; MINS="200 步, 约 45 分钟" ;;
esac

# ---------- 1. 进程 ----------
echo "当前阶段: $NAME  (按进程自动识别)"
train=$(pgrep -cf "[t]rain_${STAGE}.py")
watch=$(pgrep -cf "[w]atch_${STAGE}.sh")
if [ "$train" -gt 0 ]; then
  # GRPO 没有 dataloader worker (数据全在内存里), 有几个进程就是几条臂 ——
  # 沿用"主进程 + dataloader worker"那句话会把两条臂读成"一个进程加个 worker"。
  if [ "$STAGE" = grpo ]; then
    echo "训练进程: 在跑 (共 $train 条臂: $(_grpo_arm_names))"
  else
    echo "训练进程: 在跑 (共 $train 个: 主进程 + dataloader worker)"
  fi
else
  echo "训练进程: ⚠️  不在跑"
fi
if [ "$watch" -gt 0 ]; then
  echo "守护进程: 在跑 —— 训练被杀后会自动重选卡续训"
else
  # 三个阶段各有各的守护脚本, 拉起的命令不一样 (曾对 GRPO 印过"还没有守护脚本" ——
  # 加 watch_grpo.sh 之后就成了假警报, 见 AGENTS.md §7 第 13 条)
  case "$STAGE" in
    pretrain) WATCH_CMD="bash scripts/run_watch.sh" ;;
    *)        WATCH_CMD="setsid nohup bash scripts/watch_${STAGE}.sh >/dev/null 2>&1 &" ;;
  esac
  echo "守护进程: ⚠️  不在跑 —— 训练一旦被打断就没人拉起来了!"
  echo "           拉起: cd /root/Test/mini-r1 && $WATCH_CMD"
fi

# 顺带提一句别的阶段 (正常情况下只有一个是活的)
for s in pretrain sft grpo; do
  [ "$s" = "$STAGE" ] && continue
  [ "$(pgrep -cf "[t]rain_${s}.py")" -gt 0 ] && echo "另外 ${s} 也在跑 ($(pgrep -cf "[t]rain_${s}.py") 个进程)"
done

# ---------- 1.5 其余 GRPO 臂的进度 ----------
# 主臂上面那段已经报了; 这里的重点就是**别让第二条臂无声无息**。
# 非主臂只报最后一行 step —— 两条臂的日志行格式完全一样, 把上面那套"预计剩余"
# 再写一遍没有新信息, 而它有一半的代码是在读进程命令行, 读错了更糟。
if [ "$STAGE" = grpo ]; then
  for tag in $(_grpo_arm_suffixes); do
    [ -z "$tag" ] && continue                 # 空后缀 = 主臂, 跳过
    alog="logs/grpo$tag.log"
    echo
    echo "── 另一条臂 grpo$tag ($alog) ──"
    if [ ! -f "$alog" ]; then
      echo "  ⚠️  进程在跑但日志不存在 —— 路径对不上?"
      continue
    fi
    alast=$(grep "^step" "$alog" 2>/dev/null | tail -1)
    if [ -z "$alast" ]; then
      echo "  ⚠️  进程在跑, 但日志里一行 step 都没有 (刚启动? 还是卡在加载?)"
      echo "      存活检查: pgrep -af '[t]rain_grpo.py .*--out_dir .*grpo$tag'"
    else
      echo "  $alast"
      acur=$(printf '%s' "$alast" | awk '{print $2 + 0}')
      # ★ 起点取**第一条** step 行。日志是追加模式, 被守护重启后会接着往下写,
      #   所以早于本进程的那些行也在这儿 —— 用 (末 − 首) 会高估进度, 但不会假装在读一个
      #   不存在的进程; 真实每步耗时还是要看相邻两行的 Δ, 和主臂那段同一个道理。
      afirst=$(grep "^step" "$alog" 2>/dev/null | head -1 | awk '{print $2 + 0}')
      awatch=$(pgrep -cf "[w]atch_grpo.sh .*$tag" 2>/dev/null || echo 0)
      echo "  (step ${afirst:-?} → $acur; 本臂守护进程 ${awatch} 个)"
    fi
  done
fi

# ---------- 2. 训练进度 + 预计剩余 ----------
echo
echo "── 进度 ($LOG 最后一行) ──"
last=$(grep "^step" "$LOG" 2>/dev/null | tail -1)
if [ -z "$last" ]; then
  echo "  (日志里还没有 step 行)"
elif [ "$STAGE" = sft ]; then
  # SFT 行: "step 3700/6240 | loss 1.4276 | lr 7.13e-06 | 0.12s/micro | 10.17G" —— 没有 tok/s,
  # 所以按「每 micro-batch 秒数 × grad_accum」估, GA 从进程命令行里读。
  echo "$last"
  ga=$(pgrep -af "[t]rain_sft.py" | grep -oE 'grad_accum[ =]+[0-9]+' | head -1 | grep -oE '[0-9]+')
  [ -n "$ga" ] || ga=32
  echo "$last" | awk -v GA="$ga" '{
    split($2, st, "/"); step = st[1] + 0; total = st[2] + 0;
    n = split($0, f, "|"); sm = 0;
    for (i = 1; i <= n; i++) if (f[i] ~ /s\/micro/) { t = f[i]; gsub(/[^0-9.]/, "", t); sm = t + 0 }
    if (sm > 0 && total > 0) {
      remain_s = (total - step) * GA * sm;
      printf "剩余: %d 步 ≈ %.1f 小时 (按 %.2fs/micro × GA=%d, 每步 %.1fs)\n",
             total - step, remain_s / 3600, sm, GA, GA * sm;
    } else print "  (解析失败, 跳过剩余时间估算)"
  }'
elif [ "$STAGE" = grpo ]; then
  # GRPO 行: "step 26 | loss ... | 297s | 峰值 13.1G"
  #   · 秒数是**本次进程从启动起累计**的, 峰值是从 2026-10-13 起新加的字段
  #     (崩溃 78 次之后加的, 见 AGENTS.md §7 第 18 条)。
  #   · ★ 不能拿 `秒数 / 步数` 当每步耗时: 秒数是本次进程的, 步数是全局的。
  #     14:07 那次从 step 25 续起, step 26 的 297s 其实只含 2 步 —— 297/26 = 11 s/步,
  #     而真实是 148 s/步, 剩余时间被低估 13 倍 (报 51 分钟, 实际 11 小时)。
  #     两行之差不受这个影响: 同一进程内 Δ秒/Δ步 就是真实耗时; 刚重启时 Δ≤0, 那时直说。
  echo "$last"
  # ★ GRPO 的 step 行没有 "26/300" 这种写法 (那是 SFT 的格式), 总步数只能从
  #   进程命令行里读 --max_steps —— 和上面 SFT 读 grad_accum 是同一个套路。
  ms=$(pgrep -af "[t]rain_grpo.py" | grep -oE 'max_steps[ =]+[0-9]+' | head -1 | grep -oE '[0-9]+')
  [ -n "$ms" ] || ms=0
  grep "^step" "$LOG" 2>/dev/null | tail -2 | awk -v TOTAL="$ms" '
    {
      step[NR] = $2 + 0;
      n = split($0, f, "|");
      for (i = 1; i <= n; i++)
        if (f[i] ~ /s[ \t]*$/) { t = f[i]; gsub(/[^0-9]/, "", t); el[NR] = t + 0 }
    }
    END {
      if (NR < 2) { print "  (只有 1 行 step, 等下一步之后再估剩余时间)"; exit }
      per = (el[2] - el[1]) / (step[2] - step[1]);
      if (!(per > 0)) { print "  (本进程刚重启, Δ秒 ≤ 0, 下一条 step 行之后再估)"; exit }
      if (TOTAL > 0)
        printf "实际 %.0fs/步; 剩余: %d 步 ≈ %.1f 小时\n", per, TOTAL - step[2], (TOTAL - step[2]) * per / 3600;
      else printf "实际 %.0fs/步\n", per;
    }'
  # 显存是 GRPO 最容易翻车的地方 (见 docs/08 Q3′, 那次的代价是连崩 78 次)。
  # 每步的 peak 是 reset 过的, 所以 jsonl 里每行是「那一步自己的峰值」, 取最近 20 步的最大值。
  if [ -f "$JLOG" ]; then
    mp=$(grep -o '"peak_gib": [0-9.]*' "$JLOG" 2>/dev/null | tail -20 \
         | awk -F': ' '{if ($2 + 0 > m) m = $2 + 0} END {if (m > 0) printf "%.1f", m}')
    [ -n "$mp" ] && echo "最近 20 步峰值显存: ${mp}G  (超过 21G 就会 OOM, 见 docs/08 Q3′)"
  fi
else
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
fi

# ---------- 3. 结构化指标 / 验证集 ----------
# 只有预训练会写 jsonl (SFT/GRPO 目前只打 stdout, 上面那段就是全部指标)。
if [ -n "$JLOG" ] && [ -f "$JLOG" ]; then
  echo
  echo "── 最新指标 ($JLOG) ──"
  tail -1 "$JLOG"
  vals=$(grep val_loss "$JLOG" 2>/dev/null | tail -3)
  if [ -n "$vals" ]; then
    echo "── 验证集 loss (最近 3 次) ──"
    echo "$vals"
  fi
fi

# ---------- 4. 检查点 ----------
echo
echo "── 检查点 ($CKDIR) ──"
if [ -f "$CKDIR/$CKPT" ]; then
  ls -lh "$CKDIR/$CKPT" | awk '{print "  "$9"  "$5"  最后写入 " $6" "$7" "$8}'
  echo "  已有检查点 → 重启会自动 --resume 续训 (绝不从头再来)"
else
  echo "  ⚠️  还没有检查点 (每 $MINS 存一次)"
fi
# ★ 但别只认上面那一个别名: `grpo.pt` 是每 25 步才写一次的粗存, 紧接着某次粗存之后进程
#   被杀, 它也会显示成"一小时前" —— 看着像训练卡住了, 其实 `grpo_latest.pt` 5 分钟前刚写过。
#   所以再挑目录里**最新**的那个 .pt 报一句。假警报比没有警报更糟 (§7 第 13 条)。
newck=$(ls -t "$CKDIR"/*.pt 2>/dev/null | head -1)
if [ -n "$newck" ] && [ "$(basename "$newck")" != "$CKPT" ]; then
  age=$(( ($(date +%s) - $(stat -c %Y "$newck")) / 60 ))
  echo "  目录里最新的是 $(basename "$newck") —— ${age} 分钟前"
  if [ "$age" -gt 30 ]; then
    # ★ 两句话必须分开写: 预训练每 200 步才写一次, 如果每隔 200 步就报一次
    #   「30 分钟没写检查点」, 那它每 45 分钟就得喊一次 —— 正是第 13 条那种假警报。
    #   预训练要按它自己的节奏判 (200 步 ≈ 45 分钟), GRPO 才是"超过 30 分钟就不正常"。
    case "$STAGE" in
      pretrain) echo "  (预训练每 200 步 ≈ 45 分钟存一次, 这个间隔属正常)" ;;
      *)        echo "  ⚠️  超过 30 分钟没写新检查点, 配合上面的进程状态确认一下是不是卡住了" ;;
    esac
  fi
fi

# ---------- 5. GPU ----------
echo
echo "── GPU (本机 6 张, 共享机器) ──"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
  | awk -F', ' -v th="$THRESH" -v nm="$NAME" '{
      free = $3 - $2;
      mark = (free > th) ? "  <- 可用 (" nm " 门槛 " th " MiB)" : "";
      printf "  GPU %s  %6d / %6d MiB  已用%s  空闲 %5d MiB%s\n", $1, $2, $3, $4, free, mark;
    }'
# 训练在用哪张卡: nvidia-smi 的 compute-apps 报的是**宿主 pid**, 容器里 /proc/<宿主pid>
# 不存在, 所以没法用 pgrep 的结果去反查; 直接看守护脚本记下的选卡结果最省事。
echo
echo "── 训练在用哪张卡 ──"
# 守护日志里这一行有两种形态: 干净的 "使用 GPU: 1,2 (等待 3s)" 和
# 带着前缀的 "[watch 10-12 16:15] 已拉起 SFT: 使用 GPU: 0 (等待 0s)"。
# 用 sed 砍掉 "使用 GPU:" 之前的所有东西, 两种都能认 (${sel#...} 只认行首, 会漏掉后者)。
sel=$(grep "使用 GPU" "$WLOG" 2>/dev/null | tail -1 | sed 's/.*使用 GPU: *//')
[ -n "$sel" ] && echo "  守护脚本最近一次选卡: $sel" \
              || echo "  (日志里还没有选卡记录)"
echo
echo "提示: 训练被抢占(别人抢卡导致 OOM)后守护脚本会自动重选卡续训, 不用手动干预"
if [ "$STAGE" = grpo ]; then
  # 上面那句以前对 GRPO 印的是「还没有守护脚本, 被打断要手动重跑」——
  # 2026-10-13 加 watch_grpo.sh 之后就成了假警报 (同一屏里"守护进程: 在跑"和这句自相矛盾)。
  # 假警报比没有警报更糟: 会让人去"修"一个没坏的东西。见 AGENTS.md §7 第 13 条。
  echo "      ⚠️ 但**先看一眼是怎么死的再决定拉不拉**: 守护脚本不问原因, 确定性崩溃会被它"
  echo "         变成崩溃循环 (2026-10-13 连崩 78 次)。先 grep -c OutOfMemoryError logs/grpo.log"
fi
