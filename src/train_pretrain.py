"""
预训练脚本: DDP 分布式 + bf16 混合精度 + 梯度检查点 + 余弦退火

启动 (3 卡):
  torchrun --nproc_per_node=3 train_pretrain.py \
      --data_dir /root/Test/mini-r1/data --ckpt_dir /root/Test/mini-r1/ckpts/pretrain

单卡调试:
  python train_pretrain.py --data_dir ... --ckpt_dir ... --micro_batch 4 --grad_accum 2
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MiniR1Model, ModelConfig, enable_grad_checkpointing

# ---------------- 数据: memmap 分片顺序流 ----------------

class ShardLoader:
    """顺序读取二进制分片, 按 rank 错开起点 (每 rank 读全量数据, 顺序不同)。

    start_pos: 跳过开头 N 个 token (断点续训时从断点数据继续)。
    """

    def __init__(self, data_dir, seq_len, rank, world_size, start_pos=0):
        files = sorted(f"{data_dir}/train/shard_{i:05d}.bin"
                       for i in range(sum(1 for f in os.listdir(f"{data_dir}/train"))))
        self.maps = [np.memmap(f, dtype=np.uint32, mode="r") for f in files]
        self.seq_len = seq_len
        self.n_tokens = sum(m.size for m in self.maps)
        # 初始错开: 第 rank 从第 rank 个分片开始, 并跳过已训数据 (可跨多个分片)
        self.shard = rank % len(self.maps)
        self.pos = start_pos
        while self.pos >= self.maps[self.shard].size:
            self.pos -= self.maps[self.shard].size
            self.shard = (self.shard + 1) % len(self.maps)

    def next_batch(self, batch_size, device):
        """返回 (x, y) 各 (B, T), 不足一个分片末尾时跨分片续读。"""
        T = self.seq_len
        x = torch.empty(batch_size, T + 1, dtype=torch.long)
        for i in range(batch_size):
            row, n = [], 0
            while n < T + 1:
                m = self.maps[self.shard]
                take = min(T + 1 - n, m.size - self.pos)
                row.append(m[self.pos:self.pos + take])
                self.pos += take
                n += take
                if self.pos >= m.size:          # 转到下一个分片
                    self.shard = (self.shard + 1) % len(self.maps)
                    self.pos = 0
            x[i] = torch.from_numpy(np.concatenate(row).astype(np.int64))
        return x[:, :-1].to(device), x[:, 1:].to(device)


# ---------------- 优化器分组: 1D 参数不衰减 ----------------

def param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------- 主函数 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--micro_batch", type=int, default=8)      # 每卡微批次序列数
    ap.add_argument("--grad_accum", type=int, default=8)       # 梯度累积步数
    # 3e-4 对齐 Pythia-410M (同为 ~0.4B 级别、公开可查的配方)。我们的批更小 (131k vs 2M),
    # 小批噪声小、能吃更大的 LR, 但共享机器上要连跑几天没人盯着, 取稳不取快。
    ap.add_argument("--peak_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--decay_frac", type=float, default=0.15)  # WSD 末尾退火占比
    ap.add_argument("--max_steps", type=int, default=0)        # 0 = 跑完整数据集一遍
    ap.add_argument("--max_tokens", type=int, default=0)       # 0 = 跑完整数据集一遍 (优先于此)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--grad_ckpt", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- 分布式初始化 ----
    # 只有 world>1 才真的上 DDP: 单卡时 DDP 的梯度桶是白白多占一两 GB 显存
    # (torchrun 即使只起一个进程也会设 RANK, 不能只看 RANK 是否存在)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = (rank == 0)

    torch.manual_seed(1337 + rank)

    # ---- 模型 ----
    cfg = ModelConfig(max_seq_len=args.seq_len)
    model = MiniR1Model(cfg).to(device)
    if args.grad_ckpt:
        enable_grad_checkpointing(model)
    if master:
        print(f"模型参数量: {model.num_params/1e6:.1f}M | DDP={ddp} world={world} | grad_ckpt={args.grad_ckpt}")
    if ddp:
        model = DDP(model, device_ids=[device.index])

    # ---- 优化器 ----
    opt = torch.optim.AdamW(param_groups(model), lr=args.peak_lr,
                            betas=(0.9, 0.95), eps=1e-8)

    # ---- 进度记账: 以「已消耗 token 数」为准, 不用 step 推算 ----
    # world (卡数) 是每步 token 量的因子。共享机器上守护脚本每次重启用几张卡是随机的,
    # 若按 step*tok_per_step 推算数据偏移, 换卡后续训会跳到错误位置 (漏读/重读大量数据),
    # 所以检查点里存绝对的 tokens_seen; LR 调度也用 token 换算出的「全局步数」, 保证与卡数无关。
    #
    # 注意 grad_accum 是随卡数调整的 (卡多了就降 GA), 所以「每卡每步 token」= 131072/world,
    # 只有乘上 world 才是真正的一步优化器所消耗的 token。步数一律用 tok_per_step 换算,
    # 否则 world=2 时步数会翻倍、和 world=1 的日志没法比。
    per_gpu_tps = args.seq_len * args.micro_batch * args.grad_accum
    tok_per_step = per_gpu_tps * world
    start_tokens, ckpt_step = 0, 0
    if args.resume:
        ckpt = torch.load(f"{args.ckpt_dir}/latest.pt", map_location=device)
        (model.module if ddp else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ckpt_step = ckpt.get("step", 0)
        start_tokens = ckpt.get("tokens_seen", ckpt_step * tok_per_step)   # 兼容旧检查点

    # ---- 数据 (续训时跳过已训 token) ----
    loader = ShardLoader(args.data_dir, args.seq_len, rank, world, start_pos=start_tokens)
    target_tokens = args.max_tokens or (args.max_steps * tok_per_step if args.max_steps else loader.n_tokens)
    total_steps, n_steps = plan_progress(target_tokens, start_tokens, tok_per_step)
    ds = decay_start(total_steps, start_tokens, tok_per_step, args.decay_frac)
    if master:
        print(f"数据总量: {loader.n_tokens/1e9:.2f}B tokens | 目标: {target_tokens/1e9:.2f}B tokens")
        print(f"起点: {start_tokens:,} tokens ({start_tokens/1e9:.3f}B) | ckpt step {ckpt_step} | "
              f"world={world} | 每步 {tok_per_step:,} tokens")
        print(f"[rank{rank}] 本次计划步数: {n_steps} (全局共 {total_steps})")
        print(f"WSD 调度: warmup {args.warmup_steps} 步 -> 恒定 {args.peak_lr:g} "
              f"-> 第 {ds} 步起退火到 {args.min_lr:g}")

    # ---- 训练循环 ----
    logf = open(f"{args.ckpt_dir}/train_log.jsonl", "a")
    t0, running = time.time(), 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    model.train()
    done = start_tokens
    global_step = start_tokens // tok_per_step
    for i in range(n_steps):
        lr = lr_schedule(global_step, args, total_steps, ds)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = loader.next_batch(args.micro_batch, device)
            with torch.autocast("cuda", torch.bfloat16):
                loss, _ = model(x, y)
            (loss / args.grad_accum).backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done = start_tokens + (i + 1) * tok_per_step
        global_step = done // tok_per_step

        if i % args.log_every == 0 and master:
            window = min(i + 1, args.log_every)      # 开头不足一个窗口
            loss_avg = running / (window * args.grad_accum)
            tps = tok_per_step * window / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            print(f"step {global_step:6d}/{total_steps} | {done/1e9:.3f}B tok | "
                  f"loss {loss_avg:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s | {peak:.2f}G",
                  flush=True)
            logf.write(json.dumps({"step": global_step, "tokens": done, "loss": loss_avg,
                                   "lr": lr, "tok_per_s": tps}) + "\n"); logf.flush()
            running, t0 = 0.0, time.time()

        if i > 0 and i % args.val_every == 0 and master:   # i=0 不做: 刚起步时省一份前向图
            val_loss = evaluate(model, args.data_dir, args.seq_len, device)
            if val_loss is not None:
                print(f"  >> val loss {val_loss:.4f}", flush=True)
                logf.write(json.dumps({"step": global_step, "tokens": done,
                                       "val_loss": val_loss}) + "\n"); logf.flush()

        if (i + 1) % args.ckpt_every == 0 and master:
            save_ckpt(model, opt, args.ckpt_dir, global_step, done, ddp)
            print(f"  >> checkpoint saved @ step {global_step} ({done/1e9:.3f}B tok)", flush=True)
    if master:
        # 收尾保存: 否则最后一段进度 (不足 ckpt_every 的部分) 会丢
        save_ckpt(model, opt, args.ckpt_dir, global_step, done, ddp)
        print(f"  >> 最终 checkpoint saved @ step {global_step} ({done/1e9:.3f}B tok)", flush=True)
    print(f"[rank{rank}] 训练完成")


def plan_progress(target_tokens, start_tokens, tok_per_step):
    """把「目标/已训 token」换算成步数。抽成纯函数是为了能在 CPU 上验证:
    换卡数时 grad_accum 跟着变, 只有 tok_per_step = seq*mb*GA*world 才守恒,
    步数口径必须由它算出来, 否则 world=2 的日志步数会是 world=1 的两倍。"""
    total_steps = max(1, target_tokens // tok_per_step)
    n_steps = max(0, (target_tokens - start_tokens) // tok_per_step)
    return total_steps, n_steps


def decay_start(total_steps, start_tokens, tok_per_step, decay_frac):
    """WSD 退火起点。对续训起点取 max: 剩余步数不足退火窗口时立刻开始退火,
    否则会一直停在恒定 LR 上、直到训练结束都没退火。"""
    return max(int(total_steps * (1 - decay_frac)), start_tokens // tok_per_step)


def save_ckpt(model, opt, ckpt_dir, step, tokens_seen, ddp):
    """原子写检查点: 先写 .tmp 再 os.replace。
    共享机器上训练随时可能被杀, 直接覆盖 latest.pt 若中途被杀会留下损坏文件,
    下次 --resume 就再也起不来了。tokens_seen 是绝对进度, 换卡数也能正确续训。"""
    path = f"{ckpt_dir}/latest.pt"
    tmp = f"{path}.tmp"
    torch.save({"step": step, "tokens_seen": tokens_seen,
                "model": (model.module if ddp else model).state_dict(),
                "opt": opt.state_dict()}, tmp)
    os.replace(tmp, path)


def lr_schedule(step, args, total_steps, ds):
    """WSD 调度 (Warmup-Stable-Decay): warmup 后**恒定** LR, 只在末尾退火。

    为什么不用一路余弦: 余弦要求开跑前就把总步数定死, 而共享机器上能跑多久、
    要不要加练都是未知数。WSD 下任意时刻停下都还能"只跑一段退火尾巴"补出一个
    收敛的模型, 想延长也不用重新热身 (MiniCPM / DeepSeek 用的就是这个)。
    ds (退火起点) 由 decay_start() 单独算好传进来。
    """
    if step < args.warmup_steps:
        return args.peak_lr * max(step, 1) / args.warmup_steps
    if step < ds:
        return args.peak_lr
    p = (step - ds) / max(1, total_steps - ds)
    return args.min_lr + 0.5 * (args.peak_lr - args.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, data_dir, seq_len, device, n_batches=20):
    """验证集上取平均交叉熵。val.bin 由数据管线最后一步生成, 还没有就跳过验证。"""
    path = f"{data_dir}/val.bin"
    if not os.path.exists(path):
        return None
    val = np.memmap(path, dtype=np.uint32, mode="r")
    model.eval()
    losses = []
    n_batches = min(n_batches, val.size // (seq_len + 1))
    for i in range(n_batches):
        start = i * seq_len
        chunk = torch.from_numpy(val[start:start + seq_len + 1].astype(np.int64)).to(device)
        x, y = chunk[:-1][None, :], chunk[1:][None, :]
        with torch.autocast("cuda", torch.bfloat16):
            loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


if __name__ == "__main__":
    main()
