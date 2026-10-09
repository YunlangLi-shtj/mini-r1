"""
交互式聊天 demo: 加载任意检查点 (pretrain/sft/grpo) 对话

用法:
  python chat.py --ckpt /root/Test/mini-r1/ckpts/grpo/grpo.pt \
      --tokenizer /root/Test/mini-r1/tokenizer/tokenizer.json
"""
import argparse

import torch
from tokenizers import Tokenizer

from model import MiniR1Model, ModelConfig

IM_START, IM_END = 4, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    model = MiniR1Model(ModelConfig()).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    print("Mini-R1 对话 demo (输入 quit 退出, clear 清空上下文)")
    history = []
    while True:
        try:
            q = input("\n你: ").strip()
        except EOFError:
            break
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "clear":
            history = []
            continue
        history.append(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
        prompt = tok.encode("".join(history)).ids
        ids = torch.tensor(prompt, device=device).unsqueeze(0)
        out = model.generate(ids, args.max_new, temperature=args.temperature,
                             top_p=args.top_p, eos_id=IM_END)
        reply = tok.decode(out[0, len(prompt):].tolist())
        print(f"\nMini-R1: {reply}")
        history.append(reply + f"<|im_end|>\n")


if __name__ == "__main__":
    main()
