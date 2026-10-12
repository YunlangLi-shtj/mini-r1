"""
SFT 指令微调: OpenHermes-2.5 (Apache-2.0) 子集, ChatML 格式, 仅对助手回复计算 loss

用法 (单卡即可):
  python train_sft.py --base_ckpt /root/Test/mini-r1/ckpts/pretrain/latest.pt \
      --out_dir /root/Test/mini-r1/ckpts/sft --max_samples 200000
"""
import argparse
import gc
import json
import math
import os
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model import MiniR1Model, ModelConfig, enable_grad_checkpointing

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
    """返回 (input_ids, labels): 非 assistant 部分 label=-100, 不产生梯度。
    若截断后没有任何 assistant token (labels 全 -100), 返回 None 丢弃该样本。"""
    ids, labels = [], []
    for turn in convs:
        role = turn["from"]
        # OpenHermes 用 "human"/"gpt", 映射到 ChatML 标准角色名
        role = "user" if role == "human" else ("assistant" if role == "gpt" else role)
        is_asst = role == "assistant"
        seg = tok.encode(f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n").ids
        ids.extend(seg)
        labels.extend(seg if is_asst else [-100] * len(seg))
    ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels):
        return None
    # ★ 必须左移一位: 语言模型是拿 logits[i] 去预测 ids[i+1], 而模型 forward 是按位置
    #   逐一对齐算交叉熵的 (见 model.py 的 chunked_cross_entropy)。
    #   上面构造时 labels[i] == ids[i], 不移位就等于让模型预测它自己 —— 2026-10-12 实测
    #   loss 从 12.94 起步 (比随机 ln(152064)=11.93 还差), 而基座 val loss 才 3.13,
    #   就是这个 bug 的症状。预训练那边本来就是对的 (x, y = chunk[:-1], chunk[1:])。
    #   左移后 labels[i] = 原来的 labels[i+1], 即"下一个 token 且属于 assistant 段"。
    labels = labels[1:] + [-100]
    return ids, labels


class TokenizedSFT(torch.utils.data.Dataset):
    """已分词的 SFT 样本列表 (避免 datasets.map 对 ragged list-of-lists 的展平问题)。"""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return {"input_ids": ids, "labels": labels}


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
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--grad_ckpt", type=int, default=1,
                    help="1=逐层激活重算 (省显存, 慢一些)。SFT 没开它时 micro_batch 8 "
                         "在 24G 卡上第一次前向就 OOM (实测 16.03 GiB)")
    ap.add_argument("--ckpt_every", type=int, default=200,
                    help="每多少个优化器步存一次检查点 (0=只在结束时存)。"
                         "共享机器会被抢卡/关机, 不存检查点就要从头再来")
    ap.add_argument("--seed", type=int, default=1234,
                    help="固定 DataLoader 的 shuffle 种子, 续训时才能重现同样的样本顺序")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)

    # ---- 数据: OpenHermes-2.5, 过滤过长样本 ----
    if args.data_file:
        ds = load_dataset("json", data_files=args.data_file, split="train")
    else:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    ds = ds.select(range(min(args.max_samples, len(ds))))
    # 逐条分词 (tokenizers 是 Rust 实现, 20 万条只需 ~1 分钟)。
    # 不用 datasets.map: 它会把 ragged 的 list-of-lists 展平成多行。
    convs = ds["conversations"]
    examples = [e for e in (build_example(c, tok, args.max_len) for c in convs) if e is not None]
    dropped = len(convs) - len(examples)
    print(f"SFT 样本数: {len(examples):,} (丢弃全掩码样本 {dropped})")
    n_asst = sum(sum(1 for l in labels if l != -100) for _, labels in examples)
    print(f"assistant token 数: {n_asst:,} (占 {n_asst/sum(len(i) for i,_ in examples):.1%})")

    ds = TokenizedSFT(examples)
    # shuffle 必须挂固定种子的 generator: 续训时要能重现完全相同的样本顺序,
    # 否则「跳过前 N 个 micro-batch」跳掉的是另一批数据。不给 generator 的话
    # PyTorch 用全局随机源, 每次重启的顺序都不一样。
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True, num_workers=4,
                    pin_memory=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(args.seed))

    # ---- 模型 ----
    model = MiniR1Model(ModelConfig(max_seq_len=args.max_len)).to(device)
    # 先落 CPU 再 load: map_location=device 会在显存里再铺一份完整 state_dict,
    # 且 ckpt 这个引用会一直活到函数结束。4.6G 检查点里约 1.6G 是 fp32 权重,
    # 在共享的 24G 卡上这就是「跑得起来 / 差 594 MiB 炸掉」的差别 (2026-10-12 实测)。
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"从预训练 checkpoint 加载: step={ckpt.get('step', '?')}")
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    if args.grad_ckpt:
        enable_grad_checkpointing(model)
        print("已开启梯度检查点 (激活重算)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = math.ceil(len(ds) / (args.micro_batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"总步数: {total_steps}, 每步 token: {args.micro_batch*args.grad_accum*args.max_len:,}")

    # ---- 断点续训: 共享机器上被抢卡、机器关机都是常态, 没这个就要从头再来 ----
    # 检查点存 model+optimizer+已消费的 micro-batch 数。续训靠「固定 seed 重现顺序 +
    # 跳过已消费的那些 micro-batch」实现, 所以 seed 和 grad_accum 换了就对不上了。
    # 目录自己建: torch.save 不会建父目录, 不存在就 RuntimeError。
    # run_sft.sh 虽然 mkdir -p 过, 但不能指望它 —— 2026-10-12 的续训测试就撞上了这个:
    # 检查点写在训练循环里, 崩的时间点是「第一个检查点」而不是启动时, 也就是
    # 白跑 7 分钟之后才死, 而且没留下任何检查点, 守护脚本重启后还是从头再来 → 死循环。
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = f"{args.out_dir}/sft_latest.pt"
    start_step, start_micro = 0, 0
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        start_step, start_micro = c["step"], c["micro"]
        print(f"★ 从 SFT 检查点续训: 已完成 {start_step}/{total_steps} 步 "
              f"(已消费 {start_micro:,} 个 micro-batch)")
        del c
        gc.collect()
    else:
        print("无 SFT 检查点, 从预训练权重开始")

    def save_ckpt(step, micro):
        """原子写: 先写 .tmp 再 os.replace。被抢占时正好在覆盖文件会留下损坏的检查点,
        下次再也 resume 不起来 —— 预训练那边踩过这个坑。"""
        tmp = ckpt_path + ".tmp"
        torch.save({"step": step, "micro": micro,
                    "model": model.state_dict(), "opt": opt.state_dict()}, tmp)
        os.replace(tmp, ckpt_path)

    model.train()
    # step 只数「优化器步数」, 另用 micro 数 micro-batch。两者口径必须分开:
    # total_steps 是按 grad_accum 折算过的优化器步数, 若拿 micro-batch 计数去比,
    # cos 会在 step=total_steps 处提前走完半个周期归零, 再往后按 cos(>π) 反弹 ——
    # LR 变成震荡而不是退火。2026-10-12 的探针日志里 "step 76/16" 就是这个错位。
    # ★ micro 必须从 0 起, 不能从 start_micro 起: 跳过循环的条件是 `micro < start_micro`,
    #   若 micro 一上来就等于 start_micro, 条件当场为假 —— 一个 micro-batch 都不会跳过,
    #   模型从优化器第 32 步的状态接着, 数据却从头读起, 等于把前 32 步见过的东西又学一遍。
    #   2026-10-12 的续训测试就是这么抓到的: 续训后的 loss 比不中断低 0.5
    #   (step40: A=2.8216 vs B=2.3211) —— 低, 不是高, 因为那批数据它已经见过了。
    #   这种错不报任何异常, 只会让 loss 曲线莫名偏低, 不专门测根本发现不了。
    step, micro, running, t0, nlog = start_step, 0, 0.0, time.time(), 0
    for epoch in range(args.epochs):
        for batch in dl:
            # 续训时把已消费的 micro-batch 空转跳过。顺序能对上是因为上面给了固定 seed 的
            # generator —— 这是「跳过前 N 个」成立的前提。只空转不取数会把 GPU 干等,
            # 所以 batch 取出来了直接 continue (DataLoader 自己的开销, 不做前向)。
            if micro < start_micro:
                micro += 1
                continue
            lr = args.lr * min(1.0, step / args.warmup_steps) if step < args.warmup_steps \
                else args.lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            with torch.autocast("cuda", torch.bfloat16):
                # 走模型自带的分块交叉熵, 不要在这儿手写 F.cross_entropy:
                # micro_batch 8 x seq 2048 = 16,384 行, 不分块时 log_softmax 要在 fp32 下
                # 铺一张 16384*152064*4B ≈ 10GB 的临时张量, 24G 卡(还常和别人共卡)必定 OOM。
                # model(x, y) 内部用 chunked_cross_entropy (chunk_rows=512, 单块 ~0.3GB),
                # 且它默认 ignore_index=-100, 与「只对 assistant 段算 loss」的语义一致。
                loss, _ = model(x, y)
            (loss / args.grad_accum).backward()
            running += loss.item()
            nlog += 1
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if args.ckpt_every and step % args.ckpt_every == 0:
                    # micro 是 grad_accum 的整数倍 (只在优化器步边界存), 续训跳过时才对得齐
                    save_ckpt(step, micro)
                    print(f"[ckpt] step {step} 已存 {ckpt_path}", flush=True)
                if step % args.log_every == 0:
                    # 报显存: SFT 的 batch 比预训练大, 显存靠估会踩坑, 实测才算数
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"step {step}/{total_steps} | loss {running/nlog:.4f} | lr {lr:.2e} | "
                          f"{(time.time()-t0)/nlog:.2f}s/micro | {peak:.2f}G", flush=True)
                    running, t0, nlog = 0.0, time.time(), 0
    # 结束时再存一次: 收尾的零头步数 (不满 ckpt_every 的那部分) 也进检查点,
    # 否则「进度条走到头但 latest.pt 还差几步」会让下次重启再白跑一遍。
    save_ckpt(step, micro)
    # 最终产物不带 optimizer (只有推理/GRPO 需要权重), 省一半磁盘
    torch.save({"step": step, "model": model.state_dict()}, f"{args.out_dir}/sft.pt")
    print(f"SFT 完成, 保存至 {args.out_dir}/sft.pt (可续训检查点: {ckpt_path})")


if __name__ == "__main__":
    main()
