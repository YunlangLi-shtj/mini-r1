"""
最终评测: GSM8K 测试集 pass@1 / pass@k (多数投票), 可对比任意检查点

用法:
  python eval_gsm8k.py --ckpt /root/Test/mini-r1/ckpts/sft/sft.pt \
      --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json \
      --n_samples 8 --max_new 384
"""
import argparse
import json
from collections import Counter

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from model import MiniR1Model, ModelConfig
from train_grpo import (PROMPT_TMPL, math_reward, format_reward, extract_answer,
                        normalize, gold_answer, read_parquet, head,
                        decode_completion)

IM_END = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gsm8k_dir", default=None,
                    help="本地 gsm8k parquet 目录 (绕过 HF 数据集枚举)")
    ap.add_argument("--n_eval", type=int, default=0)     # 0 = 全量测试集
    ap.add_argument("--n_samples", type=int, default=8)  # 每题采样数 (maj@k)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)               # 结果 json 路径
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    model = MiniR1Model(ModelConfig()).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    if args.gsm8k_dir:
        raw = read_parquet(f"{args.gsm8k_dir}/test.parquet")   # 不经过 datasets, 见该函数注释
    else:
        raw = load_dataset("openai/gsm8k", "main", split="test")
    if args.n_eval:
        raw = head(raw, args.n_eval)
    # 转成 list[dict] 并用统一的 gold 字段 (parquet 里叫 answer, 是整段解题过程,
    # 末行 "#### 18" 才是答案 —— 取错了会让 maj@k 恒为 0, 见 train_grpo.gold_answer)
    ds = [{"question": q, "gold": gold_answer(a)} for q, a in zip(raw["question"], raw["answer"])]
    print(f"评测 {len(ds)} 题, 每题采样 {args.n_samples} 次")

    correct_at_1 = maj_correct = fmt_ok = 0
    fmt_sample_ok = 0          # ★ 按**条**数, 不是按题
    n_fmt_signal_groups = 0    # ★ 组内格式分有差异的组数 = GRPO 里靠格式就能拿到梯度的比例
    details = []
    for ex in ds:
        prompt = tok.encode(PROMPT_TMPL.format(question=ex["question"])).ids
        ids = torch.tensor(prompt, device=device).unsqueeze(0).repeat(args.n_samples, 1)
        outs = model.generate(ids, args.max_new, temperature=args.temperature,
                              eos_id=IM_END)
        # ★ 按 EOS 截断再解码 —— generate 是整批都出 EOS 才停, 先停的那些后面还有
        #   一段"幻觉", 不截断的话 extract_answer 会抠到幻觉里的盒子 (见 decode_completion)
        texts = [decode_completion(tok, outs[j, len(prompt):]) for j in range(args.n_samples)]
        rs = [math_reward(t, ex["gold"]) for t in texts]
        # pass@1: 第一条正确
        correct_at_1 += rs[0]
        # ★ format_ratio 是**按题**算的 (8 条里至少一条有盒子), 这个数到 1.0 不代表每条都有盒子。
        #   而 GRPO 是按组算优势的: 只有组内 8 条**有差异**才有梯度。格式分在组内饱和
        #   (全对或全错) 的话它对学习毫无贡献 —— 所以真正要看的是下面这两个按条的统计。
        fs = [format_reward(t) > 0 for t in texts]
        fmt_ok += any(fs)
        fmt_sample_ok += sum(fs)
        n_fmt_signal_groups += (0 < sum(fs) < len(fs))
        # maj@k: 多数投票 (提取答案后取众数)
        votes = Counter(normalize(extract_answer(t)) for t in texts)
        votes.pop(None, None)
        maj = votes.most_common(1)[0][0] if votes else None
        maj_correct += (normalize(maj) == normalize(ex["gold"]))
        details.append({"q": ex["question"][:80], "gold": ex["gold"],
                        "answers": [extract_answer(t) for t in texts],
                        "correct": rs[0]})
        if len(details) % 50 == 0:
            print(f"  {len(details)}/{len(ds)} | pass@1 {correct_at_1/len(details):.4f} | "
                  f"maj@{args.n_samples} {maj_correct/len(details):.4f}")

    n_tot = len(ds) * args.n_samples
    result = {
        "ckpt": args.ckpt, "n": len(ds), "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pass@1": correct_at_1 / len(ds),
        f"maj@{args.n_samples}": maj_correct / len(ds),
        "format_ratio": fmt_ok / len(ds),
        # ↓ 给 GRPO 看的两个数: 格式分按条的比例, 以及"光靠格式分就有方差的组"占比
        "fmt_per_sample": fmt_sample_ok / n_tot,
        "fmt_signal_groups": n_fmt_signal_groups / len(ds),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"result": result, "details": details}, f, indent=2, ensure_ascii=False)
        print(f"已保存至 {args.out}")


if __name__ == "__main__":
    main()
