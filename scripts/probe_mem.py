"""显存探测: 在共享 24G 卡上找出真正跑得动的 micro_batch。

背景: 407M 模型 + 152k 词表, 显存瓶颈不是模型本身而是 LM head 的输出与交叉熵
(整批 log_softmax 的 fp32 临时张量能顶到 4.6GB 的单次分配)。
只靠估算容易翻车, 这里实打实跑一遍 前向+反向+优化器 step, 报峰值显存。

用法: CUDA_VISIBLE_DEVICES=4 python scripts/probe_mem.py [--seq_len 2048]
"""
import argparse
import sys

import torch

sys.path.insert(0, "/root/Test/mini-r1/src")
from model import MiniR1Model, ModelConfig, enable_grad_checkpointing  # noqa: E402

VOCAB = 152064


def try_cfg(micro_batch, seq_len, grad_accum=1):
    """跑一步完整训练 (含优化器), 返回峰值 GiB; OOM 返回 None。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = MiniR1Model(ModelConfig(max_seq_len=seq_len)).cuda()
        enable_grad_checkpointing(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95))
        for _ in range(grad_accum):
            x = torch.randint(0, VOCAB, (micro_batch, seq_len), device="cuda")
            with torch.autocast("cuda", torch.bfloat16):
                loss, _ = model(x, x)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        del model, opt, x, loss
        torch.cuda.empty_cache()
        return peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--grad_accum", type=int, default=1)
    args = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    print(f"可用显存 {free/2**30:.2f} / {total/2**30:.2f} GiB | seq_len={args.seq_len} "
          f"grad_accum={args.grad_accum}")
    for mb in (1, 2, 4, 8):
        peak = try_cfg(mb, args.seq_len, args.grad_accum)
        if peak is None:
            print(f"  micro_batch {mb:2d}: 显存不够")
        else:
            print(f"  micro_batch {mb:2d}: 峰值 {peak:5.2f} GiB  "
                  f"(每步 {mb*args.seq_len*args.grad_accum:,} tokens)")
        if peak is None and mb >= 2:
            break


if __name__ == "__main__":
    main()
