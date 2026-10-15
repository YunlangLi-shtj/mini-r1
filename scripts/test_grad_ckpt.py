"""梯度检查点 + KV-Cache 共存的等价性测试 (纯 CPU, 秒级, 不抢卡)

背景 (AGENTS.md §7 第 18 条): GRPO 里**同一个模型**要干两件互相矛盾的事 ——
rollout 用带缓存的 `generate()` 采样 (no_grad), 更新时又要把 16 层激活全存下来反传。
激活那一份实测 ~5 GB, 正是把 24G 卡撑爆的最后一根稻草, 所以要给策略模型开梯度检查点。
但原来的 `enable_grad_checkpointing` 一见到 use_cache 就直接 raise —— 理由是「训练用不上缓存」。
GRPO 打破了这个前提, 于是加了 `cache_passthrough=True`: 缓存路径原样透传、不套检查点。

这个开关一旦接错, 症状都是**静默**的:
  - 透传写成套检查点 → 采样路径白算一遍 (只是慢, 看不出来)
  - 检查点路径漏传 is_causal → 掩码变成默认值, 梯度算错 (loss 照样下降, 结果是错的)
所以三项都逐张量比对。跑法:

    CUDA_VISIBLE_DEVICES="" python scripts/test_grad_ckpt.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from model import MiniR1Model, ModelConfig, enable_grad_checkpointing  # noqa: E402


def build(cfg):
    torch.manual_seed(0)
    return MiniR1Model(cfg).float()


def main():
    torch.manual_seed(0)
    cfg = ModelConfig()
    cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim = 64, 4, 8, 4, 128
    cfg.vocab_size, cfg.max_seq_len = 256, 64

    ids = torch.randint(0, cfg.vocab_size, (2, 7))

    # ---- [1] 带缓存的 generate: 透传后必须和没打补丁的模型逐 token 一致 ----
    plain = build(cfg).eval()
    patched = build(cfg).eval()
    enable_grad_checkpointing(patched, cache_passthrough=True)
    with torch.no_grad():
        a = plain.generate(ids.clone(), 12, temperature=0.0)
        b = patched.generate(ids.clone(), 12, temperature=0.0)
    assert torch.equal(a, b), f"[1] 带缓存的 generate 不一致\n{a}\n{b}"
    print("[1] 带缓存 generate 一致 ✓")

    # ---- [2] 不开缓存的反传: 梯度必须和没打补丁的逐张量相同 ----
    # 检查点的语义是「重算一遍前向」, 数学上无损; 对不上就说明 is_causal/mask 传歪了。
    def run(use_ckpt):
        m = build(cfg).train()
        if use_ckpt:
            enable_grad_checkpointing(m, cache_passthrough=True)
        loss = m(ids).float().pow(2).mean()
        loss.backward()
        return loss.item(), {k: v.grad.clone() for k, v in m.named_parameters()
                             if v.grad is not None}

    l0, g0 = run(False)
    l1, g1 = run(True)
    assert l0 == l1, f"[2] loss 不同: {l0} vs {l1}"
    assert set(g0) == set(g1), "[2] 有参数没拿到梯度 (检查点把某条支路断开了)"
    worst, worst_name = 0.0, ""
    for k in g0:
        d = (g0[k] - g1[k]).abs().max().item()
        if d > worst:
            worst, worst_name = d, k
    assert worst < 1e-6, f"[2] 梯度不一致: {worst_name} 最大差 {worst}"
    print(f"[2] 反传梯度一致 ✓ (loss {l0:.6f}, {len(g0)} 个参数, 最大差 {worst:.2e})")

    # ---- [3] 默认参数 (预训练/SFT) 下 use_cache 仍然报错, 别把护栏拆没了 ----
    m = build(cfg)
    enable_grad_checkpointing(m)
    try:
        with torch.no_grad():
            m.generate(ids, 3, temperature=0.0)
    except RuntimeError as e:
        assert "KV-Cache" in str(e), f"[3] 报错信息不对: {e}"
        print("[3] 默认 False 时 use_cache 仍然报错 ✓")
    else:
        raise AssertionError("[3] 默认参数下 use_cache 没报错 —— 护栏失效了")

    print("全部通过")


if __name__ == "__main__":
    main()
