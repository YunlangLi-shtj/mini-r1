"""GRPO 多卡 (DDP) 的三项 CPU 测试 —— 纯 CPU / gloo, 秒级, 不抢卡。

跑法:
    CUDA_VISIBLE_DEVICES="" python scripts/test_grpo_ddp.py

背景: `train_grpo.py` 的 DDP 支持是 2026-10-14 补的 (见 AGENTS.md §8 第 1 条),
单卡那条路一行没动, 但多卡有三处**错了也不报错、或者只在特定时刻才报错**的地方,
所以在花 6 个 GPU·小时之前先把它们钉在测试里:

  [1] 题库切分: `randperm(seed+step)[:B][rank::world]` 必须各卡不重不漏,
      而且**每次重启都一样**。写错成 `[rank*per:(rank+1)*per]` + 各卡独立抽取的话,
      各卡抽的是不同的 randperm, 切出来的"不同"题目其实会重复 —— 不报错,
      只表现为 GRPO 悄悄用了一批重复样本 (等效批变小)。
  [2] 存/取检查点的 `net` vs `policy`: DDP 包装后的 `state_dict()` 键名带 `module.`
      前缀。存的时候用 `net` (无前缀) 是对的; 但**读**的时候如果写成
      `policy.load_state_dict(...)`, 检查点里的无前缀键全部落进 missing/unexpected。
      首跑碰不到 (没有检查点), 只有被抢占后重启才炸 —— 而守护脚本只问"进程在不在",
      会把它变成崩溃循环 (AGENTS.md §7 第 18 条那个模式)。
  [3] 两卡 DDP 的语义: 每个优化器步平均的是**两个样本各自的梯度**,
      而不是单卡那种"24 个样本逐条 SGD"。两边混着跑 = 中途换了优化器,
      曲线没法解释。这里验证 DDP 确实做了这个平均, 并且两个 rank 的参数始终一致。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402


def pick(train_n, rollout_batch, seed, step, rank, world):
    """train_grpo.py 里那句切题, 原样抄过来。

    ★ 用「步号派生的种子」而不是全局 RNG: 全局 RNG 会被 model.generate 消耗掉,
      只要各卡的消耗量有一点不同, 后面抽到的题目就对不上了。
    """
    gen = torch.Generator().manual_seed(seed * 1000003 + step)
    return torch.randperm(train_n, generator=gen)[:rollout_batch][rank::world]


def test_sharding():
    train_n, seed, B = 7473, 0, 24
    for world in (1, 2, 3, 4):
        for step in (0, 1, 137):
            assert B % world == 0, "整除是前置条件 (train_grpo 里是 assert)"
            per = [sorted(pick(train_n, B, seed, step, r, world).tolist())
                   for r in range(world)]
            flat = [i for p in per for i in p]
            assert len(flat) == B, f"[1] world={world} 切出来的总数不是 {B}: {len(flat)}"
            assert len(set(flat)) == B, f"[1] world={world} 各卡题目有重复, 等效批变小了"
            expect = sorted(torch.randperm(
                train_n, generator=torch.Generator().manual_seed(seed * 1000003 + step)
            )[:B].tolist())
            assert sorted(flat) == expect, f"[1] world={world} 合起来不是那 B 道题"

    # ★ 真正要钉住的是**可复现**, 不是"去重"。
    #   踩过的错写法不是"会抽重", 而是"用全局 RNG": 全局 RNG 会被 model.generate
    #   按各卡的生成量消耗掉, 于是同一个 (seed, step) 在重启后会抽出**另一批题**,
    #   而续训记录的是"已训到第几步" —— 步数对得上、题目对不上, 不报错。
    #   (顺带纠正一条被我说过头的话: 早期注释写「各卡独立抽取会抽出重复的题」。
    #    7473 里抽 24 个, 两次独立抽样撞车的概率只有 ~1.9%, 实际几乎不重复。
    #    按 rank 派生种子、再连续切分其实也是对的; 唯一真实的分歧是"哪张卡拿到哪几题"。)
    torch.manual_seed(12345)
    torch.rand(1000)                      # 模拟 generate 消耗全局 RNG
    assert torch.equal(pick(train_n, B, seed, 5, 0, 2), pick(train_n, B, seed, 5, 0, 2)), \
        "[1] 同一个 (seed, step) 两次抽样不一致 —— 续训会跳到另一批数据上"
    gen_a = torch.Generator().manual_seed(seed * 1000003 + 5)
    gen_b = torch.Generator().manual_seed(seed * 1000003 + 5)
    torch.rand(1000)                      # 全局 RNG 被消耗
    assert torch.equal(torch.randperm(train_n, generator=gen_a),
                       torch.randperm(train_n, generator=gen_b)), \
        "[1] 步号派生的 generator 竟然受全局 RNG 影响"
    print("[1] 题库切分: 不重不漏 + 不受全局 RNG 消耗影响 ✓")


def test_state_dict_roundtrip():
    # init_process_group 默认 init_method 是 env://, 会去读 MASTER_ADDR/MASTER_PORT。
    # 单进程测试也得给, 否则直接报 "MASTER_ADDR not set"。
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29516")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        net = nn.Linear(8, 4)
        sd = net.state_dict()                      # 无前缀, 和 ckpt 里存的一样
        assert not any(k.startswith("module.") for k in sd)

        net2 = nn.Linear(8, 4)
        ddp = DDP(net2, device_ids=None)
        assert any(k.startswith("module.") for k in ddp.state_dict()), \
            "[2] DDP 的 state_dict 竟然没有 module. 前缀, 测试前提变了"

        net2.load_state_dict(sd)                   # ★ 正确的写法 (train_grpo.py 用的)
        for k in sd:
            assert torch.equal(net2.state_dict()[k], sd[k])
        print("[2a] net.load_state_dict(无前缀 ckpt) 通过 ✓")

        # 错误写法: 直接 load 进 DDP 包过的对象。这里只是**记录实际行为**,
        # 不断言它必须抛错 —— 万一某个 torch 版本会自己剥前缀, 那也是好事。
        net3 = nn.Linear(8, 4)
        ddp3 = DDP(net3, device_ids=None)
        try:
            ddp3.load_state_dict(sd)
            print("[2b] ⚠️ policy.load_state_dict(无前缀 ckpt) 竟然通过了 —— "
                  "这个 torch 版本会自己处理前缀, 修 net/policy 那处的必要性降低了")
        except RuntimeError as e:
            first = str(e).split("\n")[0][:90]
            print(f"[2b] policy.load_state_dict(无前缀 ckpt) 如期报错 ✓ ({first}...)")
    finally:
        dist.destroy_process_group()


RANK_ENV = "GRPO_DDP_TEST_RANK"


def _worker_main(outdir):
    """两卡 DDP: 每个 rank 各算自己那份梯度, backward 后应当拿到两者的平均。

    这里**不用 mp.spawn + 共享 Queue** —— gloo 和 spawn 凑一起会 SIGSEGV (试过)。
    改成「同一条命令起两个子进程, 各自落盘, 父进程读回来比」, 和 test_sft_resume.sh
    一个路子: 进程之间只通过文件交换, 不共享任何运行时状态。
    """
    rank = int(os.environ[RANK_ENV])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)

    net = nn.Linear(4, 2, bias=False)
    ddp = DDP(net, device_ids=None)
    x = torch.full((1, 4), float(rank + 1))        # 两个 rank 喂不同数据
    ddp(x).pow(2).mean().backward()
    g_ddp = net.weight.grad.clone()                # 这一份已经被 DDP all-reduce 过了

    # 手工算"本 rank 的局部梯度": 拿一份权重相同的非 DDP 模型, 只喂自己这份数据。
    # backward 不改权重, 所以此刻 net.weight 还是更新前的值, 抄过来是对的。
    m = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        m.weight.copy_(net.weight)
    m(x).pow(2).mean().backward()

    torch.save({"g_ddp": g_ddp, "g_local": m.weight.grad.clone(),
                "w": net.weight.detach().clone()}, os.path.join(outdir, f"rank{rank}.pt"))
    dist.destroy_process_group()


def test_ddp_averaging():
    import subprocess
    import tempfile

    world = 2
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, WORLD_SIZE=str(world),
                   MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
        procs = []
        for r in range(world):
            env[RANK_ENV] = str(r)
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--worker", td],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        for p in procs:
            out, _ = p.communicate(timeout=180)
            assert p.returncode == 0, f"[3] 子进程退出码 {p.returncode}\n{out.decode()[-800:]}"
        got = [torch.load(os.path.join(td, f"rank{r}.pt")) for r in range(world)]

    g_ddp = got[0]["g_ddp"]
    mean_local = (got[0]["g_local"] + got[1]["g_local"]) / 2
    assert torch.allclose(g_ddp, mean_local, atol=1e-6), (
        "[3] DDP 的梯度不是两份局部梯度的平均 —— 多卡和单卡的优化器语义对不上,\n"
        f"    DDP  ={g_ddp.flatten()[:4].tolist()}\n"
        f"    mean ={mean_local.flatten()[:4].tolist()}")
    assert torch.allclose(got[0]["w"], got[1]["w"], atol=1e-7), "[3] 两个 rank 的参数竟然不一致"
    print("[3] 两卡 DDP: 梯度 = 两份局部梯度的平均, 参数各卡一致 ✓")
    print("    ⚠️ 这正是「单卡 = 24 条逐样本 SGD / 双卡 = 12 步两份平均」那个语义差异的来源,")
    print("       守护脚本重启时必须沿用同一个卡数 (watch_grpo.sh 会转发 n_gpu)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2])
        sys.exit(0)
    test_sharding()
    test_state_dict_roundtrip()
    test_ddp_averaging()
    print("\n全部通过 ✓")
