"""
Mini-R1: 从零实现的 Llama 风格 Decoder-Only Transformer (~0.4B)

架构要点 (现代 LLM 标配, 与 Llama3 / Qwen2 一致):
  - RMSNorm (Pre-Norm), 无 LayerNorm 偏置
  - RoPE 旋转位置编码
  - SwiGLU 前馈网络 (gate/up/down, 无偏置)
  - GQA 分组查询注意力 (16 个查询头 / 8 个 KV 头)
  - Tied Embeddings (输入嵌入与输出头共享权重)
  - 无 dropout, 无偏置

参数量: 嵌入 155.7M + 16 层 × 15.7M ≈ 407M
仅依赖 PyTorch, 注意力内核自动选择 (SDPA/flash-attn)。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 152064
    max_seq_len: int = 4096
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16        # 查询头数 (head_dim = d_model / n_heads = 64)
    n_kv_heads: int = 8      # KV 头数 (GQA: 每 2 个查询头共享 1 组 KV)
    ffn_dim: int = 4096      # SwiGLU 中间维度 (4x)
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


def precompute_rope(max_seq_len: int, head_dim: int, theta: float, device) -> torch.Tensor:
    """预计算 RoPE 的 cos/sin 表, 形状 (max_seq_len, head_dim)"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          chunk_rows: int = 512) -> torch.Tensor:
    """分块计算交叉熵, 等价于 F.cross_entropy(logits, targets)。

    为什么要分块: 词表 152k 时, log_softmax 要在 fp32 下铺一张 (行数 x 词表) 的临时张量。
    micro_batch 4 x seq 2048 = 8192 行 -> 8192*152064*4B ≈ 4.6GB 的**单次分配**,
    比模型本身的 logits 还大一倍, 是 24G 卡上真正撑爆显存的那一项。
    取 512 行: 单块 512*152064*4B ≈ 0.3GB。共享机器上余量比速度重要 ——
    2048 行时要 1.2GB, 恰好把"能不能在别人腾出的卡上跑起来"卡在边界上。
    """
    v = logits.size(-1)
    total, count = logits.new_zeros(()), targets.new_zeros(())
    for i in range(0, logits.size(0), chunk_rows):
        lg = logits[i:i + chunk_rows]
        tg = targets[i:i + chunk_rows]
        total = total + F.cross_entropy(lg.reshape(-1, v).float(), tg.reshape(-1),
                                        reduction="sum")
        count = count + (tg != -100).sum()
    return total / count.clamp(min=1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (..., seq, head_dim) 张量施加旋转位置编码。

    cos/sin 需已与 x 的 (..., seq, head_dim) 形状广播兼容 (见 Attention.forward)。
    旋转: [x1;x2] -> [x1*cos - x2*sin; x1*sin + x2*cos], 按 head_dim 对半。
    """
    d = x.shape[-1] // 2
    c, s = cos[..., :d], sin[..., :d]
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class RMSNorm(nn.Module):
    """RMSNorm: y = x / RMS(x) * weight, 无偏置、无均值中心化。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保持 fp32 计算方差, 混合精度训练更稳定
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f32 * rms).to(x.dtype) * self.weight


class Attention(nn.Module):
    """GQA 注意力: 每个 KV 头服务 n_heads/n_kv_heads 个查询头, 显存占用显著降低。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads          # KV 复制倍数
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask=None, past_kv=None, use_cache=False,
                is_causal=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        cos = cos[None, :, None, :]                          # (T, d) -> (1, T, 1, d)
        sin = sin[None, :, None, :]
        q = apply_rope(q, cos, sin).transpose(1, 2)          # (B, n_heads, T, d)
        k = apply_rope(k, cos, sin).transpose(1, 2)          # (B, n_kv_heads, T, d)
        v = v.transpose(1, 2)
        present = None
        if use_cache:
            if past_kv is not None:                          # 把历史 KV 接到前面
                k = torch.cat([past_kv[0], k], dim=2)         # 沿 seq 维拼
                v = torch.cat([past_kv[1], v], dim=2)
            # ★ 缓存的是**未复制**的 KV: 存 n_kv_heads 份而不是 n_heads 份, 显存直接减半。
            #   repeat_interleave 放到使用前做 —— 每步都要复制一遍, 但那是临时的, 不驻留。
            present = (k, v)
        # GQA: KV 头沿组维复制
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        # SDPA: 若装了 flash-attn, PyTorch 自动走 flash 内核; 否则退化为内存高效内核。
        # is_causal 由调用方决定 (带缓存时**不能**用, 见 MiniR1Model.forward 里的说明)。
        if is_causal is None:
            is_causal = mask is None
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(y)
        return (out, present) if use_cache else out


class MLP(nn.Module):
    """SwiGLU 前馈: down( silu(gate(x)) * up(x) ), 三层均无偏置。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-Norm 残差块: x = x + attn(norm(x)); x = x + mlp(norm(x))"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask=None, past_kv=None, use_cache=False,
                is_causal=None):
        # 分支写开: 不开缓存时 attn 返回的是张量, 直接 `a, present = ...` 在 B==2 时
        # 会沿第 0 维解包成功而不报错 —— 一个只在 batch=2 时出现的静默错误。
        if use_cache:
            a, present = self.attn(self.attn_norm(x), cos, sin, mask, past_kv, True,
                                   is_causal)
        else:
            a = self.attn(self.attn_norm(x), cos, sin, mask, is_causal=is_causal)
        x = x + a
        x = x + self.mlp(self.ffn_norm(x))
        return (x, present) if use_cache else x


class MiniR1Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        if cfg.tie_embeddings:
            self.lm_head = None                      # 复用 tok_emb.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # RoPE 表 (注册为 buffer, 随模型自动迁移设备/dtype)
        cos, sin = precompute_rope(cfg.max_seq_len, cfg.d_model // cfg.n_heads,
                                   cfg.rope_theta, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """GPT-2 风格初始化: 所有线性层/嵌入 N(0, 0.02), RMSNorm 权重为 1。

        比 PyTorch 默认 (Embedding N(0,1)) 收敛更快: 初始 loss 直接 ≈ ln(vocab) ≈ 11.9,
        避免前几百步的高 loss 平台期。
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        """idx: (B, T) 长整型 token id。若给 targets 则返回 (loss, logits), 否则只返回 logits。

        use_cache=True 时额外返回 present (每层的 (k, v)), 并在最后返回 —— 用来做增量解码:
        第一次传整条 prompt (past_kv=None), 之后每次只传**1 个新 token** + 上一次的 present。
        """
        B, T = idx.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        total = past_len + T
        assert total <= self.cfg.max_seq_len, f"序列长度 {total} 超过上限 {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        # ★ 位置必须从 past_len 开始数。写成 cos[:T] 是个不报错的错误:
        #   形状对得上, 只是新 token 拿到了位置 0/1/2... 的旋转角, attention 全乱。
        cos = self.cos[past_len:total].to(x.dtype)
        sin = self.sin[past_len:total].to(x.dtype)

        # ★★ 带缓存时**绝不能**给 SDPA 传 is_causal=True。
        #   PyTorch 的默认因果掩码是**左上角对齐**的 (ones(T, total).tril()), 不是右下角。
        #   增量解码时 T=1、total=past+1, 于是那个掩码只让唯一的 query 看到**第 0 个 key** ——
        #   等于把整条历史都屏蔽掉、只拿 prompt 的第一个 token 做 attention。
        #   不报错、不崩, 只是生成的东西全是垃圾。实测: q_len=1/kv_len=9 时
        #   可见 key 的下标均值 = 0.000 (全可见应为 4.000)。
        #   正确做法: T==1 时不需要任何掩码 (历史全在 query 之前);
        #   T>1 时自己造右下角对齐的掩码 (第 i 行允许到 i+past 列)。
        if past_len == 0:
            mask, is_causal = None, True
        elif T == 1:
            mask, is_causal = None, False
        else:
            mask = torch.ones(T, total, dtype=torch.bool, device=x.device).tril(past_len)
            is_causal = False

        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pk = past_kv[i] if past_kv is not None else None
            if use_cache:
                x, present = layer(x, cos, sin, mask, past_kv=pk, use_cache=True,
                                   is_causal=is_causal)
                presents.append(present)
            else:
                x = layer(x, cos, sin, mask, is_causal=is_causal)
        x = self.norm(x)
        w = self.tok_emb.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(x, w)
        if targets is not None:
            # reshape 而非 view: F.linear 输出/切片 targets 可能非连续 (CPU/fp32 路径会炸)
            loss = chunked_cross_entropy(logits.reshape(-1, logits.size(-1)),
                                         targets.reshape(-1))
            return (loss, logits, presents) if use_cache else (loss, logits)
        return (logits, presents) if use_cache else logits

    @staticmethod
    def _sample(logits, temperature, top_k, top_p):
        """从 logits 采下一个 token。抽出来是为了让两条生成路径**用同一份采样逻辑** ——
        复制一份的话, 改了一边忘了另一边, 两条路就会给出不同的分布, 而且很难发现。"""
        logits = logits / max(temperature, 1e-8)   # temperature≈0 -> 近似贪心
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < v[:, -1:], -float("inf"))
        if top_p is not None:
            srt, srt_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(srt, -1), -1)
            keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool),
                              cum[:, :-1] < top_p], -1)
            logits = logits.scatter(1, srt_idx, torch.where(keep, srt, -float("inf")))
        # temperature≈0 时 logits 被放得极大, 但 softmax 内部会先减最大值,
        # 所以不会溢出成 NaN —— 结果就是一个 one-hot (等价于 argmax)。
        return torch.multinomial(F.softmax(logits, -1), 1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 eos_id=None, use_cache=True):
        """自回归采样 (评测/演示用)。

        use_cache=True (默认): **增量解码** —— prompt 只前向一次, 之后每步只喂 1 个新 token,
        历史 K/V 从缓存里取。计算量从「每步 O(L²)」降到「每步 O(L)」。
        use_cache=False: 原来的朴素实现 (每步重算整条序列), 留作对照与兜底。
        """
        if idx.size(1) > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len:]

        if not use_cache:
            for _ in range(max_new_tokens):
                ctx = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
                next_tok = self._sample(self(ctx)[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat([idx, next_tok], dim=1)
                if eos_id is not None and (next_tok == eos_id).all():
                    break
            return idx

        # 预填充 (prefill): 整条 prompt 走一遍, 顺手把每层的 K/V 留下
        logits, past = self(idx, use_cache=True)
        for _ in range(max_new_tokens):
            next_tok = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_id is not None and (next_tok == eos_id).all():
                break
            if idx.size(1) >= self.cfg.max_seq_len:
                # 缓存只增不减, 到顶就接不下去了。老的非缓存版是滑窗截断 —— 但那样
                # 位置会重新从 0 数起, 对一条连续序列本来就是错的。这里选择停下。
                break
            logits, past = self(next_tok, past_kv=past, use_cache=True)   # 只喂 1 个 token
        return idx

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def enable_grad_checkpointing(model: MiniR1Model, cache_passthrough: bool = False) -> None:
    """打开逐层激活重算 (显存换时间, 24G 卡上跑大微批次的关键)。

    注意: 必须捕获替换前的绑定方法, 否则 checkpoint 重算时会经 __call__ 再次进入
    被替换过的 fwd, 导致无限递归 (RecursionError)。

    cache_passthrough: 同一个模型上**既要带缓存的采样、又要省显存的更新**时才设 True
    (GRPO 就是这种)。此时 use_cache 的调用直接走原路径、不套检查点 ——
    采样都在 no_grad 下跑, 存不存激活本来就没区别, 套上去只会白算一遍。
    预训练/SFT 用不上缓存, 保持默认 False, 这样万一误传 use_cache 会当场报错而不是静默变慢。
    """
    from torch.utils.checkpoint import checkpoint

    def make_forward(orig_forward):
        def fwd(x, cos, sin, mask=None, past_kv=None, use_cache=False, is_causal=None):
            if use_cache:
                if not cache_passthrough:
                    raise RuntimeError(
                        "梯度检查点与 KV-Cache 不能同时用: 训练不需要缓存, "
                        "推理请用没打过这个补丁的模型 (评测/GRPO/chat 都没开梯度检查点)")
                # 增量解码: 原样透传 (no_grad 路径, 检查点没有意义)
                return orig_forward(x, cos, sin, mask, past_kv, use_cache, is_causal)
            # 训练永远走「无缓存 + 标准因果掩码」这条路, 保持 checkpoint 调用原样不动。
            assert is_causal is not False, "训练路径不该出现非因果掩码"
            return checkpoint(orig_forward, x, cos, sin, mask, use_reentrant=False)
        return fwd

    for layer in model.layers:
        layer.forward = make_forward(layer.forward)


def create_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> MiniR1Model:
    model = MiniR1Model(ModelConfig())
    model.to(device=device, dtype=dtype)
    return model


if __name__ == "__main__":
    # 快速自检: 前向 + 反向 + 采样 + KV-Cache 等价性
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = create_model(device, dtype)
    print(f"参数量: {model.num_params / 1e6:.1f}M, 设备: {device}")
    x = torch.randint(0, 152064, (2, 64), device=device)
    y = torch.randint(0, 152064, (2, 64), device=device)
    loss, logits = model(x, y)
    loss.backward()
    print(f"前向+反向 OK, loss = {loss.item():.4f}, logits 形状 = {tuple(logits.shape)}")
    out = model.generate(x[:1, :8], 8, temperature=1.0, top_k=50)
    print(f"采样 OK, 输出形状 = {tuple(out.shape)}")
    # 缓存等价性: 预填充的 logits 必须和整条前向的 logits 完全一致
    prompt = x[:1, :16]
    lg_full = model(prompt)
    lg_pf, _ = model(prompt, use_cache=True)
    diff = (lg_full - lg_pf).abs().max().item()
    assert diff == 0, f"KV-Cache 预填充与整条前向不一致, 最大差 {diff}"
    print(f"KV-Cache 预填充等价 OK (最大差 {diff})")
