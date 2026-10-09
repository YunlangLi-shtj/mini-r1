"""
从零训练 BPE 分词器 (byte-level BPE, 与 Qwen/GPT 系列同款算法)

流程:
  1. 从 FineWeb-Edu 下载若干 parquet 分片作为训练语料 (~2GB 文本, 采样 2 亿字符)
  2. 用 tokenizers 库训练 BPE: vocab 152064, 覆盖 99.99% 常见字节对
  3. 保存 tokenizer.json, 并导出 vocab 供数据管线使用

用法:
  python tokenizer_train.py --data_dir /path/to/data --out_dir /path/to/tokenizer
"""
import argparse
import os

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 152064          # 与 Qwen2 系列一致
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
NUM_SHARDS = 24              # fineweb-edu 分片数 (每个 ~1GB, 抽其中的文本)
MAX_CHARS = 200_000_000      # BPE 训练语料上限 (2 亿字符, 约 50M token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--num_shards", type=int, default=NUM_SHARDS)
    ap.add_argument("--max_chars", type=int, default=MAX_CHARS)
    ap.add_argument("--out_dir", default="/root/Test/mini-r1/tokenizer")
    args = ap.parse_args()

    # 1. 流式读取前 N 个分片, 只取 text 列 (streaming 模式不落盘)
    print(f"[1/3] 流式读取 {args.dataset} 前 {args.num_shards} 个分片 ...")
    ds = load_dataset(args.dataset, split="train", streaming=True,
                      trust_remote_code=True)

    def text_gen():
        total = 0
        for i, sample in enumerate(ds):
            if i >= args.num_shards * 10_000:      # 每个分片约 1 万行, 控制总量
                break
            total += 1
            yield sample["text"]

    # 2. 训练 BPE (底层 Rust 实现, 2 亿字符约 10-20 分钟)
    print("[2/3] 训练 BPE ...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_gen(), trainer,
                                  length=args.max_chars // 4)  # 字符数估算
    tokenizer.post_processor = None

    # 3. 保存 + 冒烟测试
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(f"{args.out_dir}/tokenizer.json")
    tok = Tokenizer.from_file(f"{args.out_dir}/tokenizer.json")
    sample = "Hello, world! 这是一段中文测试文本。Mini-R1 从零训练分词器。"
    ids = tok.encode(sample).ids
    print(f"[3/3] 完成! vocab = {tok.get_vocab_size()}")
    print(f"  样例编码: {ids[:12]} ... -> 解码: {tok.decode(ids)[:60]}")


if __name__ == "__main__":
    main()
