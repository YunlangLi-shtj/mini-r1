"""
从零实现 GRPO (Group Relative Policy Optimization, DeepSeek-R1 的核心算法)

算法流程 (每步):
  1. 对每个 prompt 采样 G 条回答 o_i ~ π_θ (温度采样)
  2. 规则奖励 r_i: 数学答案正确性 + 格式奖励
  3. 组内相对优势: A_i = (r_i - mean(r)) / std(r)   <- 无需价值网络!
  4. 策略损失: L = -E[ min(ρ_i A_i, clip(ρ_i, 1-ε, 1+ε) A_i) ] + β·KL(π_ref || π_θ)
     ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)  (逐 token 概率比, 重要性采样)
     KL 用 k3 无偏估计: exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1
  5. π_ref = SFT 检查点 (冻结)

用法:
  python train_grpo.py --base_ckpt /root/Test/mini-r1/ckpts/sft/sft.pt \
      --out_dir /root/Test/mini-r1/ckpts/grpo
"""
import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

from model import MiniR1Model, ModelConfig

IM_START, IM_END = 4, 5
PROMPT_TMPL = (
    "<|im_start|>user\n{question} Please reason step by step, "
    "and put your final answer within \\boxed{{...}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ---------------- 奖励函数: 规则化, 无外部模型 ----------------

def extract_answer(text: str) -> str | None:
    """提取最终答案: 优先 \boxed{...}, 否则取最后一个数值表达式。"""
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()
    # 兜底: 最后一个数字 (支持 1,234.56 / -3 / 1/2 / 3% )
    nums = re.findall(r"-?\d[\d,]*\.?\d*|\\frac\{[^{}]+\}\{[^{}]+\}", text)
    return nums[-1] if nums else None


def normalize(ans: str | None) -> str | None:
    if ans is None:
        return None
    a = ans.strip().lower().replace(",", "").replace("%", "").replace("$", "")
    a = re.sub(r"\s+", "", a)
    a = a.rstrip(".")
    return a


def math_reward(completion: str, gold: str) -> float:
    return 1.0 if normalize(extract_answer(completion)) == normalize(gold) else 0.0


def format_reward(completion: str) -> float:
    """格式奖励: 使用 \boxed{} 输出答案 (鼓励 R1 式推理格式)。"""
    return 0.1 if re.search(r"\\boxed\{", completion) else 0.0


# ---------------- 采样与对数概率 ----------------

@torch.no_grad()
def sample_group(model, prompt_ids, G, max_new, temperature, eos_ids):
    """对 1 个 prompt 采样 G 条回答, 返回 (ids, 逐 token 旧策略对数概率)。"""
    p = prompt_ids.unsqueeze(0).repeat(G, 1)
    model.eval()
    outs = model.generate(p, max_new, temperature=temperature, eos_id=eos_ids[0])
    model.train()
    logps = logp_of(model, outs, prompt_len=prompt_ids.numel())
    return outs, logps


@torch.no_grad()
def logp_of(model, ids, prompt_len):
    """逐 token 对数概率 (只算回答部分)。返回 (G, T_ans)。"""
    logits = model(ids)
    logp = F.log_softmax(logits.float(), -1)
    target = ids[:, 1:]
    logp = logp[:, :-1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return logp[:, prompt_len - 1:]


# ---------------- GRPO 损失 ----------------

def grpo_loss(logps_new, logps_old, adv, ref_logps, beta, eps=0.2):
    """per-token: surr = -min(ρ·A, clip(ρ)·A), kl = k3 估计。返回标量。"""
    rho = (logps_new - logps_old).exp()
    surr = -torch.minimum(rho * adv, rho.clamp(1 - eps, 1 + eps) * adv)
    kl = (ref_logps - logps_new).exp() - (ref_logps - logps_new) - 1
    return (surr + beta * kl).mean()


# ---------------- 数据 ----------------

def load_gsm8k(tokenizer_path, n_eval=256, gsm8k_dir=None):
    """GSM8K: 训练集做 rollout, 测试集前 n_eval 题做训练中评测。"""
    if gsm8k_dir:   # 本地 parquet (绕过 HF 数据集枚举)
        train = load_dataset("parquet", data_files=f"{gsm8k_dir}/train.parquet", split="train")
        test = load_dataset("parquet", data_files=f"{gsm8k_dir}/test.parquet", split="train")
    else:
        train = load_dataset("openai/gsm8k", "main", split="train")
        test = load_dataset("openai/gsm8k", "main", split="test")
    tok = Tokenizer.from_file(tokenizer_path)
    # 手动构造成 list[dict]: datasets.map 对 ragged list-of-lists 会展平成多行
    mk = lambda ds: [{"ids": tok.encode(PROMPT_TMPL.format(question=q)).ids, "gold": a}
                     for q, a in zip(ds["question"], ds["answer"])]
    train = mk(train)
    test = mk(test.select(range(min(n_eval, len(test)))))
    return train, test, tok


@torch.no_grad()
def eval_pass1(model, eval_ds, tok, max_new=384):
    """贪心解码 pass@1 (温度 0 = top-1 采样近似)。"""
    correct = 0
    for ex in eval_ds:
        ids = torch.tensor(ex["ids"], device="cuda").unsqueeze(0)
        out = model.generate(ids, max_new, temperature=0.0, eos_id=IM_END)
        text = tok.decode(out[0, len(ex["ids"]):].tolist())
        correct += math_reward(text, ex["gold"])
    return correct / len(eval_ds)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)        # SFT 检查点
    ap.add_argument("--tokenizer", default="/root/Test/mini-r1/tokenizer/tokenizer.json")
    ap.add_argument("--gsm8k_dir", default=None,
                    help="本地 gsm8k parquet 目录 (绕过 HF 数据集枚举)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--G", type=int, default=8)          # 每 prompt 采样数
    ap.add_argument("--rollout_batch", type=int, default=24)   # 每步 prompt 数
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.04)  # KL 系数
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--update_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, eval_ds, tok = load_gsm8k(args.tokenizer, gsm8k_dir=args.gsm8k_dir)
    print(f"GSM8K: 训练 {len(train_ds)} 题, 评测 {len(eval_ds)} 题")

    # ---- 策略模型 (待优化) 与参考模型 (冻结) ----
    policy = MiniR1Model(ModelConfig()).to(device)
    ckpt = torch.load(args.base_ckpt, map_location=device)
    policy.load_state_dict(ckpt["model"])
    ref = MiniR1Model(ModelConfig()).to(device)
    ref.load_state_dict(ckpt["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    logf = open(f"{args.out_dir}/grpo_log.jsonl", "a")
    t0 = time.time()
    for step in range(args.max_steps):
        lr = args.lr * min(1.0, step / args.warmup_steps) if step < args.warmup_steps \
            else args.lr * 0.5 * (1 + math.cos(math.pi * step / args.max_steps))
        for g in opt.param_groups:
            g["lr"] = lr

        # ---- 1. Rollout: 采样 + 奖励 + 优势 ----
        samples = []      # (prompt_ids, answer_ids, logps_old, adv)
        idxs = torch.randperm(len(train_ds))[:args.rollout_batch]
        for i in idxs:
            ex = train_ds[int(i)]
            prompt = torch.tensor(ex["ids"], device=device)
            outs, logps = sample_group(policy, prompt, args.G, args.max_new,
                                       args.temperature, eos_ids=[IM_END])
            texts = [tok.decode(outs[j, len(ex["ids"]):].tolist()) for j in range(args.G)]
            rs = [math_reward(t, ex["gold"]) + format_reward(t) for t in texts]
            rs = torch.tensor(rs, device=device)
            adv = (rs - rs.mean()) / (rs.std() + 1e-8)
            samples.append((prompt, outs, logps, adv))

        # ---- 2. 参考模型 logp (每步对这批回答算一次) ----
        with torch.no_grad():
            ref_logps = [logp_of(ref, outs, len(ex["ids"])) for _, outs, _, _ in samples]

        # ---- 3. K 轮梯度更新 ----
        total_loss, total_r = 0.0, 0.0
        for _ in range(args.update_epochs):
            for j, (prompt, outs, logps_old, adv) in enumerate(samples):
                full = outs
                logps_new = logp_of(policy, full, prompt.numel())
                loss = grpo_loss(logps_new, logps_old, adv[:, None],
                                 ref_logps[j], args.beta, args.clip_eps)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                total_r += adv.mean().item()

        n = len(samples) * args.update_epochs
        print(f"step {step} | loss {total_loss/n:.4f} | avg_r {total_r/n:.4f} | "
              f"lr {lr:.2e} | {time.time()-t0:.0f}s", flush=True)
        logf.write(json.dumps({"step": step, "loss": total_loss / n,
                               "avg_reward": total_r / n, "lr": lr}) + "\n")
        logf.flush()

        if (step + 1) % args.eval_every == 0:
            acc = eval_pass1(policy, eval_ds, tok)
            print(f"  >> GSM8K pass@1 = {acc:.4f}", flush=True)
            logf.write(json.dumps({"step": step, "eval_pass1": acc}) + "\n")
            logf.flush()
            torch.save({"step": step, "model": policy.state_dict()},
                       f"{args.out_dir}/grpo.pt")
    print("GRPO 训练完成")


if __name__ == "__main__":
    main()
