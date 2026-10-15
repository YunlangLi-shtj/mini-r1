#!/usr/bin/env python3
"""EOS 截断的回归测试 (纯 CPU, 秒级)。

    CUDA_VISIBLE_DEVICES="" python scripts/test_eos_truncate.py

为什么要有它 (2026-10-14):
    `generate` 是**整批**都出 EOS 才提前返回。所以一批 G 条里先停下来的那些,
    后面还会被继续推着生成 —— 解出来是"EOS 之后的幻觉"。而 `extract_answer`
    取的是**最后一个**盒子, 于是幻觉里的盒子会把真答案盖掉。

    这个 bug 的特征和 §7 第 15/19 条一模一样: **不报错、不崩、只是数字悄悄变差**。
    当时是翻 `show_completions` 的原始输出才看见的 —— 一条采样先写了
    `\\boxed{60,000}`, 然后自己往下编了 8 个 `\\boxed{14000}`。

    所以这里逐条钉死: 截断在哪、掩码对不对齐、损失有没有真的丢掉那段。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from train_grpo import decode_completion, answer_mask, grpo_loss, extract_answer

IM_END = 5
FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"[{'✓' if ok else '✗'}] {name}" + ("" if ok else f"\n     得到 {got!r}\n     期望 {want!r}"))
    if not ok:
        FAILED.append(name)


class FakeTok:
    """只需要 .decode —— decode_completion 只用它, 所以不必读真的分词器。"""
    CH = {0: "1", 1: "8", 2: " ", 3: "\\boxed{", 4: "}", 5: "<EOS>", 6: "9", 7: "0"}

    def decode(self, ids):
        return "".join(self.CH.get(int(i), f"<{int(i)}>") for i in ids)


tok = FakeTok()

print("=== [1] decode_completion: 在第一个 EOS 处截断 ===")
# "18 " + EOS + "90"  →  只该留 "18 "
check("[1a] 截断掉 EOS 之后的 token",
      decode_completion(tok, torch.tensor([0, 1, 2, IM_END, 6, 7])), "18 ")
# EOS 自己也不留在文本里 (它是控制符, 不是回答内容)
check("[1b] EOS 本身不进文本",
      decode_completion(tok, torch.tensor([0, 1, IM_END])), "18")
check("[1c] 没有 EOS 就原样解码",
      decode_completion(tok, torch.tensor([0, 1, 2])), "18 ")
check("[1d] 空生成", decode_completion(tok, torch.tensor([], dtype=torch.long)), "")
check("[1e] 只接受 list (调用方传 .tolist() 也照样行)",
      decode_completion(tok, [0, 1, IM_END, 6]), "18")

print("\n=== [2] ★ 真实 bug 的形状: 幻觉盖掉真答案 ===")
# 模型先答对 18, 出 EOS, 然后(被整批拖着)继续编出一个假答案 90
gen = [3, 0, 1, 4, IM_END, 3, 6, 7, 4]      # "\boxed{18}<EOS>\boxed{90}"
full_text = tok.decode(gen)                  # 老写法: 整段解码
trunc_text = decode_completion(tok, gen)     # 新写法: 截断
check("[2a] 老写法抠到幻觉里的 90", extract_answer(full_text), "90")
check("[2b] 新写法拿到真答案 18", extract_answer(trunc_text), "18")

print("\n=== [3] answer_mask: 对齐与边界 ===")
# ⚠️ 填充 token 只用 6/7/8/9 —— **绝不能用 5**。IM_END 就是 5, 拿它当普通 token 会让
#    每一行都意外"含 EOS" (这个坑第一次写这个测试时就踩了, 4 项假失败)。
prompt_len = 3
outs = torch.tensor([[1, 2, 3,  6, 7, IM_END, 8,  9],     # 回答第 3 个 token 才出 EOS
                     [1, 2, 3,  6, IM_END, 7, 8, 9],      # 回答第 2 个 token 就出 EOS
                     [1, 2, 3,  6, 7, 8, 9, 6]])          # 到顶都没出 EOS
m = answer_mask(outs, prompt_len, eos_id=IM_END)
check("[3a] 形状对齐 logp_of 的输出 (G, T_ans)",
      tuple(m.shape), (3, outs.size(1) - prompt_len))
check("[3b] 第 0 行: EOS 保留, 之后全丢",
      m[0].tolist(), [1.0, 1.0, 1.0, 0.0, 0.0])
check("[3c] 第 1 行: EOS 保留, 之后全丢",
      m[1].tolist(), [1.0, 1.0, 0.0, 0.0, 0.0])
check("[3d] 第 2 行: 没有 EOS ⇒ 全 1 (与老行为一致)",
      m[2].tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])

# 逐列对齐: mask 第 i 列对应 outs[:, prompt_len + i]; EOS 自己算 1, 其后的算 0
mismatch = None
for g in range(outs.size(0)):
    seen_eos = False
    for i in range(m.size(1)):
        want = 0.0 if seen_eos else 1.0
        if outs[g, prompt_len + i].item() == IM_END:
            seen_eos = True
        if m[g, i].item() != want:
            mismatch = (g, i, m[g, i].item(), want)
check("[3e] 逐列对齐 outs[:, prompt_len + i]", mismatch, None)

print("\n=== [4] grpo_loss: 被屏蔽的位置真的不进梯度 ===")
G, T = 2, 4
torch.manual_seed(0)
logps_new = torch.zeros(G, T, requires_grad=True)
logps_old = torch.zeros(G, T)
adv = torch.ones(G, 1)
ref = torch.zeros(G, T)

# 把最后两列做得"代价极高", 掩掉它们之后结果不该变
loud_new = logps_new.detach().clone().requires_grad_(True)
loud_new.data[:, 2:] = 50.0
loud_old = torch.zeros(G, T)
loud_old[:, 2:] = -50.0

mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * G)
l_masked = grpo_loss(loud_new, loud_old, adv, ref, 0.0, mask=mask)
l_clean = grpo_loss(logps_new.detach()[:, :2], logps_old[:, :2], adv, ref[:, :2], 0.0)
check("[4a] 掩掉的位置不贡献损失", round(l_masked.item(), 6), round(l_clean.item(), 6))

# mask=None 必须和全 1 掩码等价 (保证老路径没被改坏)
ones = torch.ones(G, T)
check("[4b] mask=None ≡ 全 1 掩码",
      round(grpo_loss(logps_new, logps_old, adv, ref, 0.04).item(), 6),
      round(grpo_loss(logps_new, logps_old, adv, ref, 0.04, mask=ones).item(), 6))

# 归一化用 mask.sum() 而不是 T: 否则分母混进被屏蔽的位置, 梯度整体被缩小。
# 推导: per_tok 对有效位置是 -adv_g, loss = Σ(per_tok·mask)/mask.sum(),
#       所以第 g 行每个有效位置的梯度是 -adv_g/mask.sum()。
half = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * G)
g_t = logps_new.detach().clone().requires_grad_(True)
grpo_loss(g_t, logps_old, adv, ref, 0.04, mask=half).backward()
n_valid = half.sum().item()
want_grad = -(adv.squeeze(-1) * half.sum(-1)).sum().item() / n_valid
check("[4c] 梯度按有效 token 数归一化",
      round(g_t.grad.sum().item(), 5), round(want_grad, 5))
# 逐位置核一遍, 别只看总和 (总和能对上、分布错了的情况是有的)
check("[4c'] 每个有效位置的梯度 = -adv/mask.sum()",
      [[round(x, 5) for x in row] for row in g_t.grad.tolist()],
      [[round(-adv[j].item() / n_valid, 5)] * 2 + [0.0, 0.0] for j in range(G)])

print()
if FAILED:
    print(f"❌ {len(FAILED)} 项没过: {FAILED}")
    sys.exit(1)
print("✅ 全部通过 —— EOS 截断与掩码行为符合预期")
