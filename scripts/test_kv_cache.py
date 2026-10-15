"""KV-Cache 等价性测试 (纯 CPU + 小模型, 不抢卡)。

为什么非要有这个测试: 加缓存要动两处**错了也不报错**的地方 ——
  1) 位置编码的起点: 新 token 的 RoPE 角度必须从 past_len 开始数。写成 cos[:T] 形状
     照样对得上, 只是新 token 拿到的是位置 0,1,2... 的旋转角, attention 全乱;
  2) 缓存拼接顺序: 必须是 [历史, 当前]。接反了形状也对, 结果全错。
两者都只表现为「生成的文本变差」, 在 0.4B 模型上肉眼根本看不出来。

所以判据不是「看起来能跑」, 而是**逐位置比对 logits**:
整条前向 (非缓存) 与「预填充 + 逐个 token 增量」在同一个位置必须给出同一个分布。

用法: python scripts/test_kv_cache.py
"""
import sys

import torch

sys.path.insert(0, "/root/Test/mini-r1/src")
from model import MiniR1Model, ModelConfig      # noqa: E402

TOL = 1e-5          # logits 允许的绝对误差 (CPU fp32 下实测通常正好是 0)
FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  {name} ✓  {detail or ''}")
    except AssertionError as e:
        FAILURES.append(f"{name}: {e}")
        print(f"  {name} ✗  {e}")


def small_model(**kw):
    """随机小模型 —— 词表 512 / 2 层, CPU 上跑几秒。测的是**逻辑**, 不是效果。"""
    cfg = ModelConfig(vocab_size=512, max_seq_len=kw.pop("max_seq_len", 64),
                      d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128)
    torch.manual_seed(0)
    return MiniR1Model(cfg).eval()


# ---------------- 核心: 逐位置 logits 比对 ----------------

def t_prefill(model):
    """预填充不该改变任何东西: 它和非缓存前向是同一串算子。"""
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        a = model(ids)
        b, past = model(ids, use_cache=True)
    d = (a - b).abs().max().item()
    assert d <= TOL, f"预填充与整条前向不一致, 最大差 {d}"
    n_layers = len(past)
    assert n_layers == model.cfg.n_layers, f"缓存层数 {n_layers} != {model.cfg.n_layers}"
    k, v = past[0]
    assert k.shape[1] == model.cfg.n_kv_heads, \
        f"缓存应存未复制的 KV ({model.cfg.n_kv_heads} 头), 实际 {k.shape[1]}"
    assert k.shape[2] == 16, f"预填充后缓存长度应为 16, 实际 {k.shape[2]}"
    return f"最大差 {d:.2e}, 缓存 {n_layers} 层 × {k.shape[1]} KV 头 × {k.shape[2]} 步"


def t_incremental(model):
    """★ 最关键的测试: 预填充 + 逐 token 增量, 每个位置的 logits 都要对得上整条前向。

    用**真实的**下一个 token 喂进去 (而不是采样出来的), 这样每一步都有确定的目标可比。
    """
    ids = torch.randint(0, 512, (1, 24))
    P = 8
    with torch.no_grad():
        full = model(ids)                                     # (1, 24, V) 每步的分布
        nxt, past = model(ids[:, :P], use_cache=True)          # 预填充前 8 个
        worst = 0.0
        for t in range(P, 24):
            # 此时 nxt[:, -1, :] 应该是「第 t 个 token」的预测分布
            d = (nxt[:, -1, :] - full[:, t - 1, :]).abs().max().item()
            worst = max(worst, d)
            nxt, past = model(ids[:, t:t + 1], past_kv=past, use_cache=True)
        assert past[0][0].shape[2] == 24, f"缓存长度应长到 24, 实际 {past[0][0].shape[2]}"
    assert worst <= TOL, f"增量解码与整条前向不一致, 最大差 {worst} (位置编码起点写错就会这样)"
    return f"16 个位置逐个比对, 最大差 {worst:.2e}"


def t_chunked(model):
    """一次续写多个 token (T>1 且带缓存) 也要等价 —— 这条走的是自造的右下角掩码分支。

    ★ 下标口径是这个测试最容易写错的地方 (第一版就写错了, 白查了半天):
      「第 j 行的 logits」= 喂完下标 j 的 token **之后**的分布, 对应**预测下标 j+1**。
      所以续写块的第 i 行 (吃掉的是下标 P+i) 要跟 full 的**第 P+i 行**比,
      不是第 P-1+i 行 —— 差一行的话, 每行都在跟前一个 token 的分布比,
      数字看着像"全错"(随机初始化的小模型相邻位置差 ~1), 其实模型是对的。
      干脆拼成一整条 [预填充末行, 续写各行] 跟 full[P-1 : P+C] 比, 少一次下标算术。
    """
    ids = torch.randint(0, 512, (1, 20))
    P, C = 6, 5
    with torch.no_grad():
        full = model(ids)
        lg, past = model(ids[:, :P], use_cache=True)
        lg2, past2 = model(ids[:, P:P + C], past_kv=past, use_cache=True)
    pred = torch.cat([lg[:, -1:, :], lg2], dim=1)             # (1, 1+C, V)
    d = (pred[0] - full[0, P - 1:P + C]).abs().max().item()
    assert d <= TOL, f"多 token 续写与整条前向不一致, 最大差 {d:.2e} (右下角掩码写错就会这样)"
    assert past2[0][0].shape[2] == P + C, f"缓存长度应为 {P+C}, 实际 {past2[0][0].shape[2]}"
    return f"一次喂 {C} 个 token (第 1 行 + 块内 {C} 行), 最大差 {d:.2e}"


def t_greedy(model):
    """贪心解码: 两条路径必须给出**完全相同**的 token 序列 (逐 token 相等)。"""
    ids = torch.randint(0, 512, (1, 6))
    with torch.no_grad():
        a = model.generate(ids, 12, temperature=0.0)
        b = model.generate(ids, 12, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), f"贪心结果不一致:\n缓存 {(a[0, 6:].tolist())}\n朴素 {(b[0, 6:].tolist())}"
    return f"12 个 token 完全一致: {a[0, 6:].tolist()}"


def t_sampling(model):
    """同一随机种子下, 两条采样路径也应给出相同结果 (采样逻辑是共用的那一份)。"""
    ids = torch.randint(0, 512, (1, 6))
    with torch.no_grad():
        torch.manual_seed(1234)
        a = model.generate(ids, 10, temperature=1.0, top_k=20)
        torch.manual_seed(1234)
        b = model.generate(ids, 10, temperature=1.0, top_k=20, use_cache=False)
    assert torch.equal(a, b), f"同种子采样不一致:\n缓存 {(a[0, 6:].tolist())}\n朴素 {(b[0, 6:].tolist())}"
    return f"同种子同结果: {a[0, 6:].tolist()}"


def t_batch_g(model):
    """GRPO 的真实形状: 一次 8 条 (G=8) 并行采样, batch 维的缓存不能串。"""
    ids = torch.randint(0, 512, (8, 6))          # 8 条不同的 prompt
    with torch.no_grad():
        a = model.generate(ids, 8, temperature=0.0)
        b = model.generate(ids, 8, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "batch=8 时缓存路径与非缓存路径结果不一致"
    assert len({tuple(r) for r in a[:, 6:].tolist()}) > 1, "8 条结果全一样, 这个测试没测到东西"
    return "8 条并行, 与非缓存逐 token 一致"


def t_eos(model):
    """EOS 提前停止: 两条路径要在同一位置停。"""
    ids = torch.randint(0, 512, (1, 6))
    with torch.no_grad():
        eos = int(model(ids)[:, -1, :].argmax(-1))         # 贪心的下一个 token 一定就是它
        a = model.generate(ids, 10, temperature=0.0, eos_id=eos)
        b = model.generate(ids, 10, temperature=0.0, eos_id=eos, use_cache=False)
    assert a.shape == b.shape == (1, 7), f"应在第 1 个 token 处停止, 实际 {a.shape} vs {b.shape}"
    return f"两条路径都在生成 1 个 token 后停止 (eos={eos})"


def t_max_len():
    """超过 max_seq_len 不能崩 (缓存只增不减, 到顶就停)。"""
    model = small_model(max_seq_len=24)
    ids = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        out = model.generate(ids, 40, temperature=0.0)
    assert out.shape[1] <= 24, f"长度 {out.shape[1]} 超过了 max_seq_len=24"
    return f"prompt 20 + 请求 40 -> 实际停在 {out.shape[1]} (上限 24), 未崩"


def main():
    model = small_model()
    print(f"KV-Cache 等价性测试 (CPU, {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数小模型)\n")
    check("[1] 预填充 logits == 整条前向", lambda: t_prefill(model))
    check("[2] 增量解码逐位置 logits 一致", lambda: t_incremental(model))
    check("[3] 一次续写多个 token 也一致", lambda: t_chunked(model))
    check("[4] 贪心解码逐 token 一致", lambda: t_greedy(model))
    check("[5] 同种子采样结果一致", lambda: t_sampling(model))
    check("[6] batch=G=8 并行正确", lambda: t_batch_g(model))
    check("[7] EOS 提前停止一致", lambda: t_eos(model))
    check("[8] 超过 max_seq_len 不崩", t_max_len)

    print()
    if FAILURES:
        for f in FAILURES:
            print(f"失败: {f}")
        sys.exit(f"\n{len(FAILURES)} 项未通过 —— KV-Cache 不可用")
    print("全部通过: KV-Cache 与朴素实现数值等价, 可以直接替换")


if __name__ == "__main__":
    main()
