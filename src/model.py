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

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        cos = cos[None, :, None, :]                          # (T, d) -> (1, T, 1, d)
        sin = sin[None, :, None, :]
        q = apply_rope(q, cos, sin).transpose(1, 2)          # (B, n_heads, T, d)
        k = apply_rope(k, cos, sin).transpose(1, 2)          # (B, n_kv_heads, T, d)
        v = v.transpose(1, 2)
        # GQA: KV 头沿组维复制
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        # SDPA: 若装了 flash-attn, PyTorch 自动走 flash 内核; 否则退化为内存高效内核
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=(mask is None)
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


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

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


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

    def forward(self, idx, targets=None):
        """idx: (B, T) 长整型 token id。若给 targets 则返回 (loss, logits), 否则只返回 logits。"""
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"序列长度 {T} 超过上限 {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)
        cos = self.cos[:T].to(x.dtype)
        sin = self.sin[:T].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        w = self.tok_emb.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(x, w)
        if targets is None:
            return logits
        # reshape 而非 view: F.linear 输出/切片 targets 可能非连续 (CPU/fp32 路径会炸)
        loss = chunked_cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 eos_id=None):
        """朴素自回归采样 (评测/演示用, 未做 KV-Cache 优化)。"""
        for _ in range(max_new_tokens):
            ctx = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = -float("inf")
            if top_p is not None:
                srt, srt_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(srt, -1), -1)
                keep = torch.cat([torch.ones_like(cum[:, :1], dtype=torch.bool),
                                  cum[:, :-1] < top_p], -1)
                logits = logits.scatter(1, srt_idx, torch.where(keep, srt, -float("inf")))
            probs = F.softmax(logits, -1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_id is not None and (next_tok == eos_id).all():
                break
        return idx

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def enable_grad_checkpointing(model: MiniR1Model) -> None:
    """打开逐层激活重算 (显存换时间, 24G 卡上跑大微批次的关键)。

    注意: 必须捕获替换前的绑定方法, 否则 checkpoint 重算时会经 __call__ 再次进入
    被替换过的 fwd, 导致无限递归 (RecursionError)。
    """
    from torch.utils.checkpoint import checkpoint

    def make_forward(orig_forward):
        def fwd(x, cos, sin, mask=None):
            return checkpoint(orig_forward, x, cos, sin, mask, use_reentrant=False)
        return fwd

    for layer in model.layers:
        layer.forward = make_forward(layer.forward)


def create_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> MiniR1Model:
    model = MiniR1Model(ModelConfig())
    model.to(device=device, dtype=dtype)
    return model


if __name__ == "__main__":
    # 快速自检: 前向 + 反向 + 采样
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
