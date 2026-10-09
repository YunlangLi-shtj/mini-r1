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
    ap.add_argument("--peak_lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
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
    ddp = "RANK" in os.environ
    if ddp:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        master = (rank == 0)
    else:
        rank, world, device, master = 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"), True

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
    ref_tps = args.seq_len * args.micro_batch * args.grad_accum      # world=1 时每步 token
    tok_per_step = ref_tps * world
    start_tokens, ckpt_step = 0, 0
    if args.resume:
        ckpt = torch.load(f"{args.ckpt_dir}/latest.pt", map_location=device)
        (model.module if ddp else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ckpt_step = ckpt.get("step", 0)
        start_tokens = ckpt.get("tokens_seen", ckpt_step * ref_tps)   # 兼容旧检查点

    # ---- 数据 (续训时跳过已训 token) ----
    loader = ShardLoader(args.data_dir, args.seq_len, rank, world, start_pos=start_tokens)
    target_tokens = args.max_tokens or (args.max_steps * ref_tps if args.max_steps else loader.n_tokens)
    total_steps = max(1, target_tokens // ref_tps)                   # 全局步数 (LR 调度用)
    n_steps = max(0, (target_tokens - start_tokens) // tok_per_step)  # 本次进程要跑的步数
    if master:
        print(f"数据总量: {loader.n_tokens/1e9:.2f}B tokens | 目标: {target_tokens/1e9:.2f}B tokens")
        print(f"起点: {start_tokens:,} tokens ({start_tokens/1e9:.3f}B) | ckpt step {ckpt_step} | "
              f"world={world} | 每步 {tok_per_step:,} tokens")
        print(f"[rank{rank}] 本次计划步数: {n_steps} (全局共 {total_steps})")

    # ---- 训练循环 ----
    logf = open(f"{args.ckpt_dir}/train_log.jsonl", "a")
    t0, running = time.time(), 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    model.train()
    done = start_tokens
    global_step = start_tokens // ref_tps
    for i in range(n_steps):
        lr = lr_schedule(global_step, args, total_steps)
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
        global_step = done // ref_tps

        if i % args.log_every == 0 and master:
            window = min(i + 1, args.log_every)      # 开头不足一个窗口
            loss_avg = running / (window * args.grad_accum)
            tps = tok_per_step * window / (time.time() - t0)
            print(f"step {global_step:6d}/{total_steps} | {done/1e9:.3f}B tok | "
                  f"loss {loss_avg:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s", flush=True)
            logf.write(json.dumps({"step": global_step, "tokens": done, "loss": loss_avg,
                                   "lr": lr, "tok_per_s": tps}) + "\n"); logf.flush()
            running, t0 = 0.0, time.time()

        if i % args.val_every == 0 and master:
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


def lr_schedule(step, args, total_steps):
    """warmup + 余弦退火到 min_lr (step 为与卡数无关的全局步数)。"""
    if step < args.warmup_steps:
        return args.peak_lr * max(step, 1) / args.warmup_steps
    p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
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
