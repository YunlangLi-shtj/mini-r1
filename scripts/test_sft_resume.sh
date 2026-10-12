#!/bin/bash
# SFT 断点续训的端到端测试 (三段对比法)。
#
# 为什么要它: 续训路径「在真正被中断之前从来没被执行过」。守卫脚本、原子写、
# 跳过前 N 个 micro-batch —— 这些都是只在「出事时」才走到的代码, 出事时人不在,
# 调试不了。所以要在没事的时候先跑一遍。
# (2026-10-12 写第一版时就靠它抓到 torch.save 不建父目录的 bug:
#  崩点在「第一个检查点」而不是启动时, 即白跑 7 分钟才死, 而且没留下检查点,
#  守护脚本重启后还是从头再来 —— 是个死循环。)
#
# 三段:
#   A  不中断跑完 63 步          -> 参照
#   B1 跑到「检查点 step 32 已写」立刻杀掉 (模拟关机/被抢占)
#   B2 从检查点续训到结束
# 判定: B2 必须打印「★ 从 SFT 检查点续训: 已完成 32/63 步」, 且之后同步数的 loss 与 A 一致。
#   若「跳过前 N 个 micro-batch」的实现错了 (比如 shuffle 顺序没固定住), B2 会跳到
#   另一批数据上接着训, 表现为 loss 对不上或者跳变 —— 这种错不会报任何异常。
#
# 用法: setsid nohup bash scripts/test_sft_resume.sh >/dev/null 2>&1 &
#       结果看 logs/test_sft_resume.log (约 10 分钟, 需要一张 12G 空闲的卡)
cd /root/Test/mini-r1
OUT=/root/Test/sft_rt
LOG=logs/test_sft_resume.log
mkdir -p logs
: > "$LOG"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# 等一张够空的卡 (和生产同口径: nvidia-smi 12,000 MiB + 利用率 < 20%)
for i in $(seq 1 20); do
  GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', ' '$3 - $2 > 12000 && $4 < 20 {print $1; exit}')
  if [ -n "$GPU" ]; then break; fi
  log "等空闲卡... ($i)"; sleep 60
done
if [ -z "$GPU" ]; then log "没有空闲卡, 放弃"; exit 1; fi
log "用 GPU $GPU"

rm -rf $OUT; mkdir -p $OUT
COMMON="--base_ckpt ckpts/pretrain/latest.pt --tokenizer tokenizer/tokenizer.json \
  --data_file data/raw/openhermes/openhermes2_5.json --max_samples 512 --max_len 1024 \
  --micro_batch 1 --grad_accum 8 --grad_ckpt 1 --lr 2e-5 --epochs 1 --warmup_steps 10 \
  --seed 1234 --log_every 8"
run() {  # $1=out_dir  $2=日志文件
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    miniforge/envs/mini-r1/bin/python -u src/train_sft.py $COMMON \
    --out_dir "$1" --ckpt_every 8 > "$2" 2>&1
}

# 注意: 数据加载 (解析 1.9G 的 json + 分词) 每次要 3~4 分钟, 而 `timeout N` 是从
# 进程启动就开始计的 —— 拿它当「跑 30 秒就杀」用, 实际会在加载阶段就被杀掉,
# 一个检查点都没有 (第一版测试就是这么白跑一轮的)。所以改成盯日志里的检查点行。
log "=== 1/3 A: 不中断跑完 (参照) ==="
run $OUT/A $OUT/A.log
rc=$?; log "A rc=$rc, $(grep -c '^step ' $OUT/A.log) 条 step 记录"
if [ $rc -ne 0 ]; then log "A 就没跑通, 先修这个"; tail -5 $OUT/A.log | tee -a "$LOG"; exit 1; fi

log "=== 2/3 B1: 跑到 step 32 的检查点就杀掉 ==="
setsid nohup env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  miniforge/envs/mini-r1/bin/python -u src/train_sft.py $COMMON \
  --out_dir $OUT/B --ckpt_every 8 > $OUT/B1.log 2>&1 &
for i in $(seq 1 300); do
  if grep -q "\[ckpt\] step 32 已存" $OUT/B1.log 2>/dev/null; then break; fi
  sleep 2
done
# 只杀我们自己的这个测试进程: 括号技巧让本命令自己的命令行不匹配这个正则
pkill -9 -f "sft_rt/[B]" 2>/dev/null
sleep 3
log "B1 被杀在: $(grep -o '\[ckpt\] step [0-9]*' $OUT/B1.log | tail -1); 检查点 $(ls -lh $OUT/B/sft_latest.pt 2>/dev/null | awk '{print $5}')"

log "=== 3/3 B2: 续训 ==="
run $OUT/B $OUT/B2.log
log "B2 rc=$?"
grep -E "^★|^无 SFT|^SFT 完成" $OUT/B2.log | tee -a "$LOG"

log "--- 同步数 loss 对比 (A 不中断 vs B 续训) ---"
# 只比 40/48/56: 一共 63 步、每 8 步记一次, 最后一条记录是 step 56 (没有 64)
fail=0
for s in 40 48 56; do
  a=$(grep "^step $s/" $OUT/A.log  | sed 's/.*loss \([0-9.]*\).*/\1/')
  b=$(grep "^step $s/" $OUT/B2.log | sed 's/.*loss \([0-9.]*\).*/\1/')
  d=$(awk -v x="$a" -v y="$b" 'BEGIN{
        if (x=="" || y=="") {print "NA"; exit}
        d=(x-y)/x; if (d<0) d=-d; printf "%.1f%%", d*100 }')
  log "  step $s:  A=$a   B=$b   相对差 $d"
  # 5% 的门槛: 修好之后两边应该几乎一样 (bf16 的非确定性只有千分之几),
  # 而 2026-10-12 那个「没跳过已消费数据」的 bug 是 18%。中间空得很, 不会误报。
  over=$(awk -v x="$a" -v y="$b" 'BEGIN{
        if (x=="" || y=="") {print 1; exit}
        d=(x-y)/x; if (d<0) d=-d; print (d>0.05)}')
  [ "$over" = "1" ] && fail=1
done
if [ "$fail" -eq 0 ]; then
  log "✅ 通过: 续训后的 loss 与不中断一致 —— 训练确实从被中断的那一点接着走了"
else
  log "❌ 不通过: 续训后的 loss 与不中断对不上。最可能的原因是「跳过前 N 个 micro-batch」"
  log "   没跳对 (比如 micro 初值写成了 start_micro, 跳过一个都不跳), 或者是 shuffle 顺序"
  log "   没固定住。这种错不会抛异常, 只会让 loss 曲线莫名偏低 —— 别忽略这个结果。"
fi
log "完成。"
