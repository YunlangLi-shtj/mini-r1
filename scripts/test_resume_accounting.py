"""续训记账冒烟测试 (CPU, 小 seq_len): 验证「按 token 记账」在换批量/换卡数后仍然对齐。

背景: 守护脚本每次重启可能拿到不同数量的 GPU, 而每步消耗的 token 数与卡数成正比。
若续训的数据偏移用 step*tok_per_step 推算, 换卡后就会跳到错误的数据位置。
现在改为检查点里存绝对的 tokens_seen, 本测试验证这条链路:
  1) ShardLoader(start_pos=T) 真的从第 T 个 token 开始读 (用位置编码的数据验证)
  2) 训练脚本写出的 ckpt["tokens_seen"] == 跑过的 token 数
  3) 用不同的 micro_batch/grad_accum 续训, 起点仍取 tokens_seen, 剩余步数按新批量算

用法: python scripts/test_resume_accounting.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

PROJ = "/root/Test/mini-r1"
PY = f"{PROJ}/miniforge/envs/mini-r1/bin/python"
WORK = "/root/Test/mini-r1/data_smoke_acct"
VOCAB = 152064
SEQ, MB, GA = 32, 2, 2
REF_TPS = SEQ * MB * GA           # 128


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"命令失败: {cmd}")
    return r.stdout


def make_data():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(f"{WORK}/data/train", exist_ok=True)
    # 每个 shard 8000 个 token, 值 = 该 token 在全局流中的绝对位置
    for s in range(2):
        base = s * 8000
        np.arange(base, base + 8000, dtype=np.uint32).tofile(f"{WORK}/data/train/shard_{s:05d}.bin")
    np.arange(0, 8000, dtype=np.uint32).tofile(f"{WORK}/data/val.bin")


def test_loader_offset():
    """ShardLoader(start_pos=T) 的首个 token 必须是 T。"""
    sys.path.insert(0, f"{PROJ}/src")
    from train_pretrain import ShardLoader
    for start in (0, 100, 7990, 8000, 16000 + 5):
        ld = ShardLoader(f"{WORK}/data", SEQ, rank=0, world_size=1, start_pos=start)
        if start < 16000:
            x, _ = ld.next_batch(1, "cpu")
            got = x[0, 0].item()
            assert got == start, f"start_pos={start}: 期望首个 token {start}, 实际 {got}"
    print("  [1] ShardLoader 起点偏移正确 (含跨分片)")


def run_train(extra):
    # CUDA_VISIBLE_DEVICES="" 强制走 CPU: 共享机器上 GPU 常年被占满, 这个测试不该去抢卡
    return sh(f"cd {PROJ} && CUDA_VISIBLE_DEVICES='' {PY} -u src/train_pretrain.py "
              f"--data_dir {WORK}/data "
              f"--ckpt_dir {WORK}/ckpt --seq_len {SEQ} --micro_batch {MB} --grad_accum {GA} "
              f"--val_every 1000 --log_every 1 --ckpt_every 4 {extra} 2>&1")


def main():
    make_data()
    test_loader_offset()

    # ---- 阶段 A: 跑 4 步 (512 token), 应写出 tokens_seen=512 ----
    out = run_train(f"--max_tokens {REF_TPS * 4}")
    ck = f"{WORK}/ckpt/latest.pt"
    assert os.path.exists(ck), "没有生成检查点"
    import torch
    c = torch.load(ck, map_location="cpu", weights_only=False)
    assert c["tokens_seen"] == REF_TPS * 4, f"tokens_seen 应为 {REF_TPS*4}, 实际 {c['tokens_seen']}"
    assert c["step"] == 4, f"step 应为 4, 实际 {c['step']}"
    print(f"  [2] 阶段A: tokens_seen={c['tokens_seen']}, step={c['step']} ✓")

    # ---- 阶段 B: 换批量续训 (模拟换卡数: 每步 token 数变了), 起点必须仍是 512 ----
    out = sh(f"cd {PROJ} && CUDA_VISIBLE_DEVICES='' {PY} -u src/train_pretrain.py "
             f"--data_dir {WORK}/data "
             f"--ckpt_dir {WORK}/ckpt --seq_len {SEQ} --micro_batch 4 --grad_accum 2 "
             f"--val_every 1000 --log_every 1 --ckpt_every 4 --resume "
             f"--max_tokens {REF_TPS * 8} 2>&1")
    assert "起点: 512 tokens" in out, f"续训起点不对:\n{out[-2000:]}"
    # 目标 1024 token, 已跑 512, 新的每步 token = 32*4*2 = 256 -> 只剩 2 步
    assert "本次计划步数: 2" in out, f"剩余步数不对:\n{out[-2000:]}"
    c = torch.load(ck, map_location="cpu", weights_only=False)
    assert c["tokens_seen"] == REF_TPS * 8, f"续训后 tokens_seen 应为 {REF_TPS*8}, 实际 {c['tokens_seen']}"
    print(f"  [3] 阶段B(换批量续训): 起点取自 tokens_seen, 收尾 tokens_seen={c['tokens_seen']} ✓")

    # ---- 阶段 C: 已训满, 再续训应为 0 步 ----
    out = sh(f"cd {PROJ} && CUDA_VISIBLE_DEVICES='' {PY} -u src/train_pretrain.py "
             f"--data_dir {WORK}/data "
             f"--ckpt_dir {WORK}/ckpt --seq_len {SEQ} --micro_batch {MB} --grad_accum {GA} "
             f"--val_every 1000 --log_every 1 --ckpt_every 4 --resume "
             f"--max_tokens {REF_TPS * 8} 2>&1")
    assert "本次计划步数: 0" in out, f"应判定为已训满:\n{out[-2000:]}"
    assert "训练完成" in out
    print("  [4] 阶段C(已训满): 0 步, 打印「训练完成」(守护脚本据此退出) ✓")

    shutil.rmtree(WORK, ignore_errors=True)
    print("\n全部通过: 换卡数续训的数据偏移与步数记账正确")


if __name__ == "__main__":
    main()
