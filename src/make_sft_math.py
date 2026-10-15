"""
把 GSM8K 训练集的**解题过程**转成 SFT 用的 ChatML 数据 (GRPO 的冷启动热身)。

为什么需要这一步: OpenHermes 里几乎没有数学思维链, 所以 SFT 基座在 GSM8K 上
pass@1 = 0 (见 AGENTS.md §8 第 1 条)。而 GRPO 学的是「组内相对优势」——
8 条回答全都错的时候奖励没有差异、优势全是 0, **这组数据白采**。
做对率 0.2% 意味着 500 条 rollout 才有一条正样本, RL 只能强化"已经偶尔做对"的行为,
所以它推不动一个完全不会做题的基座 —— 实测也确实没推动, 唯一被强化的是那个
可以照抄的格式分。

这一段热身让基座先具备"会做题"的能力, GRPO 才有真信号可放大。

用法:
  python src/make_sft_math.py --gsm8k_dir data/raw/gsm8k --out data/sft_math.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_grpo import PROMPT_SUFFIX  # noqa: E402  保证 SFT/RL 两边输入逐字符一致

CALC = re.compile(r"<<[^>]*>>")   # GSM8K 的 <<48/2=24>> 计算器标注


def to_example(question: str, answer: str) -> dict | None:
    """GSM8K 的一行 -> OpenHermes 风格的 conversations。

    答案字段长这样 (最后一行才是答案):
        Natalia sold 48/2 = <<48/2=24>>24 clips in May.
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
        #### 72

    转成:
        human:      <问题> + PROMPT_SUFFIX      ← 和 GRPO 的提问逐字符一致
        assistant:  <解题过程> + \\boxed{72}     ← 教它两件事: 写过程、把答案放进盒子
    """
    if "####" not in answer:
        return None
    cot, ans = answer.rsplit("####", 1)
    ans = ans.strip()
    if not ans:
        return None
    # 去掉 <<...>> 标注: 那是给人看的验算, 不是自然的推理文本。
    # 留着会让模型学会吐 <<...>>, 白费 rollout 的 token (GRPO 里 max_new 只有 384)。
    cot = CALC.sub("", cot).strip()
    if not cot:
        return None
    return {"conversations": [
        {"from": "human", "value": f"{question.strip()}\n{PROMPT_SUFFIX}"},
        {"from": "gpt", "value": f"{cot}\n\n\\boxed{{{ans}}}"},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=f"{args.gsm8k_dir}/{args.split}.parquet",
                      split="train")
    rows = [e for e in (to_example(q, a) for q, a in zip(ds["question"], ds["answer"]))
            if e is not None]
    print(f"GSM8K {args.split}: {len(ds)} 条 -> {len(rows)} 条 SFT 样本 "
          f"(丢弃 {len(ds) - len(rows)})")

    # 自检: 生成的 human 轮必须**逐字符**等于 GRPO 模板渲染出来的那段。
    # 这两边一旦漂移, 模型在 SFT 见到的输入分布和 RL 阶段不一样, 等于换了个任务,
    # 而且不报任何错 —— 只会表现为"热身了但没效果"。
    from train_grpo import PROMPT_TMPL
    q = ds["question"][0].strip()
    rendered = PROMPT_TMPL.format(question=q)
    inner = rendered.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert rows[0]["conversations"][0]["value"] == inner, (
        "热身数据的 human 轮和 GRPO 的提问模板对不上 —— 先修, 别拿去训"
    )
    print("自检: 热身 human 轮 == GRPO 提问模板渲染结果 ✓")

    # 也自检一下 assistant 轮能过评测那套裁判 (别教出一个评不出分的格式)
    from train_grpo import math_reward, format_reward
    for r in rows[:200]:
        a = r["conversations"][1]["value"]
        assert math_reward(a, r["conversations"][1]["value"]) >= 0, "内部错误"
    m = rows[0]["conversations"][1]["value"]
    print(f"自检: 第一条 assistant 轮能拿到格式分 {format_reward(m)} "
          f"(必须是 0.1, 否则热身教的格式判不出来) ✓")
    assert format_reward(m) > 0, "热身数据里的答案拿不到格式分 —— 格式和裁判对不上"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    n_asst = sum(len(r["conversations"][1]["value"]) for r in rows)
    print(f"写出 {args.out}: {len(rows)} 条, assistant 平均 {n_asst/len(rows):.0f} 字符")
    print(f"示例:\n  human: {rows[0]['conversations'][0]['value'][:120]!r}...")
    print(f"  gpt  : {rows[0]['conversations'][1]['value'][:200]!r}...")


if __name__ == "__main__":
    main()
