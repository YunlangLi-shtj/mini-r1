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
from train_grpo import PROMPT_TMPL, math_reward, format_reward, extract_answer, normalize

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
        ds = load_dataset("parquet", data_files=f"{args.gsm8k_dir}/test.parquet", split="train")
    else:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n_eval:
        ds = ds.select(range(args.n_eval))
    print(f"评测 {len(ds)} 题, 每题采样 {args.n_samples} 次")

    correct_at_1 = maj_correct = fmt_ok = 0
    details = []
    for ex in ds:
        prompt = tok.encode(PROMPT_TMPL.format(question=ex["question"])).ids
        ids = torch.tensor(prompt, device=device).unsqueeze(0).repeat(args.n_samples, 1)
        outs = model.generate(ids, args.max_new, temperature=args.temperature,
                              eos_id=IM_END)
        texts = [tok.decode(outs[j, len(prompt):].tolist()) for j in range(args.n_samples)]
        rs = [math_reward(t, ex["gold"]) for t in texts]
        # pass@1: 第一条正确
        correct_at_1 += rs[0]
        fmt_ok += any(format_reward(t) > 0 for t in texts)
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

    result = {
        "ckpt": args.ckpt, "n": len(ds), "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pass@1": correct_at_1 / len(ds),
        f"maj@{args.n_samples}": maj_correct / len(ds),
        "format_ratio": fmt_ok / len(ds),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"result": result, "details": details}, f, indent=2, ensure_ascii=False)
        print(f"已保存至 {args.out}")


if __name__ == "__main__":
    main()
