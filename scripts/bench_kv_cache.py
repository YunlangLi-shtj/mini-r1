"""KV-Cache 实测: 同一段 rollout 跑两遍 (带缓存 / 不带缓存), 报吞吐、加速比、峰值显存。

为什么非要用真模型真形状测:
  「加缓存快 10 倍」是个定律, 但**实际快多少**取决于模型多大、序列多长。
  序列短的时候, 单步 forward 的时间大头是 kernel 启动开销 (每生成一个 token 都要发几十个
  kernel), 而不是计算 —— 这时缓存带来的加速远小于理论值。所以数字必须以实测为准。

形状完全对齐 GRPO 的真实调用 (src/train_grpo.py sample_group):
  一次 1 个 prompt 扩成 G=8 条, 生成 max_new 个 token; 每步 24 个 prompt。
  不开 eos 提前停止 —— 提前停会让两条路径结束在不同位置, 比出来的是假数字。

用法: python scripts/bench_kv_cache.py --gpu 2
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/root/Test/mini-r1/src")
from model import MiniR1Model, ModelConfig      # noqa: E402

FREE_MIN_MIB = 8000     # 低于这个空闲量就不跑: 共享机器上别把别人的卡跑爆


def free_mib(idx):
    free, _ = torch.cuda.mem_get_info(idx)
    return free / 2 ** 20


@torch.no_grad()
def timed(model, ids, max_new, use_cache, reps):
    """跑 reps 次取**最快**的一次。取最快而非平均: 共享机器上只会被拖慢, 不会变快,
    最小值最接近这台机器真实的能力上限。"""
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(ids, max_new, temperature=1.0, top_k=50, use_cache=use_cache)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prompt_len", type=int, default=128)   # GSM8K prompt 大致就这么长
    ap.add_argument("--max_new", type=int, default=384)      # 与 run_grpo.sh 一致
    ap.add_argument("--G", type=int, default=8)              # 每个 prompt 采样条数
    ap.add_argument("--reps", type=int, default=3)
    # 只测一条路径。★ 显存数字必须**单独进程**测: 同一个进程里先跑完无缓存那条,
    # 分配器的池子已经被撑大了, 后测的「峰值 reserved」是上一个的形状留下的,
    # 会虚高好几个 GiB (实测 20 GiB vs 真实值), 拿它去定显存门槛就会白等卡。
    ap.add_argument("--only", choices=["cache", "nocache"], default=None)
    args = ap.parse_args()
    paths = {"cache": [True], "nocache": [False], None: [False, True]}[args.only]

    freem = free_mib(args.gpu)
    print(f"GPU {args.gpu}: 空闲 {freem:.0f} MiB", flush=True)
    if freem < FREE_MIN_MIB:
        sys.exit(f"空闲显存不足 {FREE_MIN_MIB} MiB, 换一张卡 (共享机器上不要挤别人)")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)

    torch.manual_seed(0)
    model = MiniR1Model(ModelConfig()).to(device).eval()      # 真实配置: 407M
    print(f"模型 {sum(p.numel() for p in model.parameters())/1e6:.1f}M, fp32 "
          f"(与 train_grpo.py 一致, 那里没有 autocast)", flush=True)

    ids = torch.randint(0, model.cfg.vocab_size, (args.G, args.prompt_len), device=device)
    total_new = args.G * args.max_new

    # 预热: 第一次跑含 cudnn 算法选择/显存池扩张, 不计时
    model.generate(ids, 8, temperature=0.0, use_cache=True)

    # 先对一下答案: GPU fp32 路径下两条路也必须一致 (CPU 上是单测保证的, 这里防设备相关差异)
    torch.manual_seed(0)
    a = model.generate(ids, 32, temperature=0.0, use_cache=True)
    torch.manual_seed(0)
    b = model.generate(ids, 32, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "GPU 上两条路径不一致 —— 先别测速, 去查正确性"
    print("正确性自检: GPU 上贪心 32 步两条路径逐 token 一致 ✓\n", flush=True)

    print(f"形状: batch {args.G} × (prompt {args.prompt_len} + 生成 {args.max_new}) = "
          f"{args.G * (args.prompt_len + args.max_new)} token/组\n")

    res = {}
    for use_cache in paths:
        tag = "有缓存" if use_cache else "无缓存"
        torch.cuda.reset_peak_memory_stats()
        sec = timed(model, ids, args.max_new, use_cache, args.reps)
        res[tag] = (sec, total_new / sec,
                    torch.cuda.max_memory_allocated() / 2 ** 30,
                    torch.cuda.max_memory_reserved() / 2 ** 30)
        s, tps, alloc, rsv = res[tag]
        print(f"{tag}: {s:.2f} s/组 | {tps:,.0f} tok/s | "
              f"峰值 allocated {alloc:.2f} GiB / reserved {rsv:.2f} GiB", flush=True)

    if args.only:
        return
    speed = res["无缓存"][0] / res["有缓存"][0]

    # 折算到 GRPO 的一步: 24 个 prompt × 每组时间 (rollout 是串行的, 见 train_grpo.py)
    n_prompt = 24
    print(f"\n加速比: {speed:.2f}×")
    print(f"折算 GRPO 一步 (rollout {n_prompt} 个 prompt, 其余部分不受影响): "
          f"{n_prompt * res['无缓存'][0] / 60:.1f} min -> {n_prompt * res['有缓存'][0] / 60:.1f} min")
    print(f"300 步的 rollout 总时长: "
          f"{300 * n_prompt * res['无缓存'][0] / 3600:.1f} h -> "
          f"{300 * n_prompt * res['有缓存'][0] / 3600:.1f} h")


if __name__ == "__main__":
    main()
