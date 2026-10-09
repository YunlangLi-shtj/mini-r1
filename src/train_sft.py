"""
SFT 指令微调: OpenHermes-2.5 (Apache-2.0) 子集, ChatML 格式, 仅对助手回复计算 loss

用法 (单卡即可):
  python train_sft.py --base_ckpt /root/Test/mini-r1/ckpts/pretrain/latest.pt \
      --out_dir /root/Test/mini-r1/ckpts/sft --max_samples 200000
"""
import argparse
import json
import math
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model import MiniR1Model, ModelConfig

IM_START, IM_END = 4, 5  # <|im_start|> / <|im_end|> (与分词器 SPECIAL_TOKENS 顺序一致)


def format_chatml(convs):
    """OpenHermes 的 conversations -> ChatML 文本。"""
    parts = []
    for turn in convs:
        role = turn["from"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{turn['value']}<|im_end|>\n")
        elif role == "human":
            parts.append(f"<|im_start|>user\n{turn['value']}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>assistant\n{turn['value']}<|im_end|>\n")
    return "".join(parts)


def build_example(convs, tok, max_len):
    """返回 (input_ids, labels): 非 assistant 部分 label=-100, 不产生梯度。"""
    ids, labels = [], []
    for turn in convs:
        role = turn["from"]
        seg = tok.encode(f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n").ids
        is_asst = role == "assistant"
        ids.extend(seg)
        labels.extend(seg if is_asst else [-100] * len(seg))
    ids, labels = ids[:max_len], labels[:max_len]
    return ids, labels


def collate(examples):
    """动态 padding 到本批最长样本 (右侧 pad, label 侧用 -100)。"""
    max_len = max(len(e["input_ids"]) for e in examples)
    xs, ys = [], []
    for e in examples:
        pad = max_len - len(e["input_ids"])
        xs.append(torch.tensor(e["input_ids"] + [3] * pad, dtype=torch.long))
        ys.append(torch.tensor(e["labels"] + [-100] * pad, dtype=torch.long))
    return {"input_ids": torch.stack(xs), "labels": torch.stack(ys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--tokenizer", default="/root/Test/mini-r1/tokenizer/tokenizer.json")
    ap.add_argument("--data_file", default=None,
                    help="本地 OpenHermes json 文件 (绕过 HF 数据集枚举)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=200000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)

    # ---- 数据: OpenHermes-2.5, 过滤过长样本 ----
    if args.data_file:
        ds = load_dataset("json", data_files=args.data_file, split="train")
    else:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"SFT 样本数: {len(ds):,}")

    def prep(batch):
        outs = [build_example(c, tok, args.max_len) for c in batch["conversations"]]
        return {"input_ids": [o[0] for o in outs], "labels": [o[1] for o in outs]}

    ds = ds.map(prep, batched=True, batch_size=128, remove_columns=ds.column_names)
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, num_workers=4,
                    pin_memory=True, collate_fn=collate)

    # ---- 模型 ----
    model = MiniR1Model(ModelConfig(max_seq_len=args.max_len)).to(device)
    ckpt = torch.load(args.base_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"从预训练 checkpoint 加载: step={ckpt.get('step', '?')}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = math.ceil(len(ds) / (args.micro_batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"总步数: {total_steps}, 每步 token: {args.micro_batch*args.grad_accum*args.max_len:,}")

    model.train()
    step, running, t0 = 0, 0.0, time.time()
    for epoch in range(args.epochs):
        for batch in dl:
            lr = args.lr * min(1.0, step / args.warmup_steps) if step < args.warmup_steps \
                else args.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            (loss / args.grad_accum).backward()
            running += loss.item()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total_steps} | loss {running/100:.4f} | lr {lr:.2e} | "
                      f"{(time.time()-t0)/100:.2f}s/step", flush=True)
                running, t0 = 0.0, time.time()
    torch.save({"step": step, "model": model.state_dict()}, f"{args.out_dir}/sft.pt")
    print(f"SFT 完成, 保存至 {args.out_dir}/sft.pt")


if __name__ == "__main__":
    main()
