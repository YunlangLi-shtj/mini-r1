"""
从零实现 GRPO (Group Relative Policy Optimization, DeepSeek-R1 的核心算法)

算法流程 (每步):
  1. 对每个 prompt 采样 G 条回答 o_i ~ π_θ (温度采样)
  2. 规则奖励 r_i: 数学答案正确性 + 格式奖励
  3. 组内相对优势: A_i = (r_i - mean(r)) / std(r)   <- 无需价值网络!
  4. 策略损失: L = -E[ min(ρ_i A_i, clip(ρ_i, 1-ε, 1+ε) A_i) ] + β·KL(π_θ || π_ref)
     ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)  (逐 token 概率比, 重要性采样)
     KL 用 k3 无偏估计: exp(logπ_ref - logπ_θ) - (logπ_ref - logπ_θ) - 1
       k3 的性质: 样本 x ~ p 时, (q/p) - log(q/p) - 1 是 KL(p||q) 的无偏估计。
       这里 p=π_θ, q=π_ref, 所以估出来的是 KL(π_θ||π_ref) —— 把当前策略拉回
       参考策略。注意方向别写反 (写成 KL(π_ref||π_θ) 是常见的笔误)。
       rollout 是 π_θ_old 采的, 更新若干轮后 π_θ≠π_θ_old, 严格说这时它只是
       近似无偏 (和 DeepSeekMath 的做法一致)。
  5. π_ref = SFT 检查点 (冻结)

用法:
  python train_grpo.py --base_ckpt /root/Test/mini-r1/ckpts/sft/sft.pt \
      --out_dir /root/Test/mini-r1/ckpts/grpo
"""
import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from model import MiniR1Model, ModelConfig, enable_grad_checkpointing

IM_START, IM_END = 4, 5
# ★ 提问模板里**不能出现可以被照抄的字面量**。
#   原来的写法是 "... within \\boxed{{...}}." —— 那个带省略号的占位符
#   和格式奖励要查的子串长得一模一样, 于是模型找到了最省力的路径:
#   把提示词里的 \boxed{...} 原样抄出来, 白拿 0.1 分 (2026-10-14, 见 AGENTS.md §7 第 19 条)。
#   现在留的是**空盒子** \boxed{}: 它仍然告诉模型"要用这个命令包住答案",
#   但照抄它得 0 分 (空盒子不算答案), 光抄已经拿不到任何奖励了。
PROMPT_TMPL = (
    "<|im_start|>user\n{question}\n"
    "Please reason step by step, and put your final answer in a "
    "\\boxed{{}} command.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# GRPO 模板里 "<|im_start|>user\n" 和 "<|im_end|>" **之间**那段的原文。
# GSM8K-CoT 热身数据要用它拼 SFT 的 human 轮 —— 两边的输入必须逐字符一致,
# 否则模型在 SFT 见到的分布和 RL 阶段不一样, 等于偷偷换了个任务 (而且不会报错)。
PROMPT_SUFFIX = ("Please reason step by step, and put your final answer in a "
                 "\\boxed{} command.")

# 上面那个模板是给 .format() 用的, 所以字面量的花括号写成了 {{}}。
# 这份常量是为了让热身脚本能拿到**不带 ChatML 外壳**的同一段话。
# 两处一旦漂移, SFT 和 RL 的输入就对不上了 —— 加个断言当场炸, 别等训完 12 小时才发现。
assert PROMPT_TMPL.format(question="") == (
    "<|im_start|>user\n\n" + PROMPT_SUFFIX
    + "<|im_end|>\n<|im_start|>assistant\n"
), "PROMPT_TMPL 和 PROMPT_SUFFIX 漂移了 —— 先修这个, 否则 SFT 和 GRPO 的输入分布不一致"


# ---------------- 奖励函数: 规则化, 无外部模型 ----------------

# 盒子里出现这些内容 = "没装答案": 空盒子、省略号、各种省略号的写法。
# ⚠️ 这份名单是**安全底线**, 不是"答案白名单" —— 唯一的判据是:
#    **有没有一种不用真做出答案、就能让这个盒子出现的写法?** 有, 就得堵上。
#    2026-10-14 那次 hacking 就是漏了这一条 (提示词里印着 \boxed{...})。
_PLACEHOLDER = {"", "...", "…", ". . .", "。。。", "xx", "your answer"}


def is_placeholder(x: str) -> bool:
    """盒子里的内容是空的 / 是占位符 → 不算"装了答案"。"""
    n = normalize(x)
    return n is None or n in _PLACEHOLDER


def extract_answer(text: str) -> str | None:
    """提取最终答案: 优先最后一个**非空**的 \\boxed{...}, 否则取最后一个数值表达式。

    ★ 盒子里的内容要过一遍 `is_placeholder`: 否则模型写个 `\\boxed{...}`
      就会在这里被当成"答案 = ..."抠出来, 把后面真正的数字挡掉
      (原来用的是 `[^{}]+`, 它会匹配上省略号)。
    """
    m = [x for x in re.findall(r"\\boxed\{([^{}]*)\}", text) if not is_placeholder(x)]
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


def gold_answer(ans: str) -> str:
    """从 GSM8K 的 answer 字段里取出最终答案。

    ★ 这个函数是补出来的, 因为不写它的代价实测过 (2026-10-13): GSM8K 的 answer 是
      **整段解题过程**, 答案在最后一行 —— "Janet sells 16 - 3 - 4 = ... #### 18"。
      而模型输出里抽出的是 "18"。拿整段跟一个数字比, 永远不相等, 于是
      **奖励恒等于 0**: 训练时 GRPO 一个梯度都拿不到 (权重纹丝不动),
      评测时 pass@1 恒为 0。日志上只看到 r=0.0000, 极容易被误判成「模型太弱」,
      实际是裁判坏了。GSM8K 的约定是 "#### " 后面跟答案。
    """
    return ans.split("####")[-1].strip()


def math_reward(completion: str, gold: str) -> float:
    # 这里对 gold 再过一次 gold_answer: 调用方传整段解题过程也不会算错,
    # 裁判函数自己要能把「数据集字段」翻译成「答案」。
    return 1.0 if normalize(extract_answer(completion)) == normalize(gold_answer(gold)) else 0.0


def has_boxed_answer(completion: str) -> bool:
    """输出里有没有一个**真的装了东西**的 \\boxed{...}。

    ★ 不能只查 `"\\boxed{"` 这个子串 —— 那正是 2026-10-14 那次 reward hacking 的入口:
      提示词里印着 `\\boxed{...}`, 而奖励只认这 4 个字符, 于是模型学会**照抄占位符**
      白拿 0.1 分。实测含占位符的采样从 4.7% 涨到 66%, 而那 42 条"用了 \\boxed{"
      的贪心输出里**一条真答案都没有**。
      → 判据必须是「盒子里装着东西」, 不是「盒子的左半边出现过」。
    """
    return any(not is_placeholder(x)
               for x in re.findall(r"\\boxed\{([^{}]*)\}", completion))


def format_reward(completion: str) -> float:
    """格式奖励: 用 \\boxed{} 给出了**一个真实的答案**再加 0.1 分 (R1 式推理格式)。"""
    return 0.1 if has_boxed_answer(completion) else 0.0


# ---------------- 生成段的解码与掩码 ----------------
#
# ★ 这两个函数修的是同一件事: **EOS 之后的 token 不是回答**。
#
# `generate` 是**整批**都出 EOS 才提前返回 (`(next_tok == eos_id).all()`), 所以一批
# G 条里先停下来的那些, 后面还会被解码器继续推着生成 —— 解出来就是"EOS 之后的幻觉"。
# 2026-10-14 实测 (sft_math6, 160 条采样): 92.5% 的采样在**有效段内**就正常收尾,
# 有效长度中位 87 token (训练分布中位 83, 对得上); 但**不截断**时同一批里能看到
# 一条连写 8 个 `\boxed{14000}`。
#
# 危害在于 `extract_answer` 取的是**最后一个**盒子 (这个口径本身是对的 —— 训练数据的
# 最后一个盒子就是答案), 于是它常常抠到幻觉里的盒子, 把真正的答案盖掉。
# 提示词里抄占位符那次 hacking (§7 第 19 条) 是"奖励能被满足", 这次是
# **"奖励读错了地方"**: 同一个 bug 同时污染了评测口径和 GRPO 的 rollout 奖励。

def decode_completion(tok, gen_ids, eos_id=IM_END) -> str:
    """把一条生成解码成文本, **在第一个 <|im_end|> 处截断**。

    语义上生成本来就终止于 EOS —— 把 EOS 之后算进答案纯属 harness bug。
    旧写法是 `tok.decode(outs[j, prompt_len:].tolist())`, 三处都是 (rollout 奖励 /
    贪心评测 / eval_gsm8k), 已全部换成这个。
    """
    ids = gen_ids.tolist() if torch.is_tensor(gen_ids) else list(gen_ids)
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    return tok.decode(ids)


def answer_mask(outs, prompt_len, eos_id=IM_END):
    """(G, T_ans) 的 0/1 掩码: **保留 EOS 本身, 丢掉 EOS 之后**。

    和上面是同一个 bug 的另一半: 奖励已经按 EOS 截断了, 但如果损失里仍然包含
    EOS 之后的 token, 那么每个 token 都在被同一个序列级优势加权 —— 等于**教模型
    在下课铃响之后继续说话**。掩码把这段排除掉。

    EOS 本身要保留 (mask=1): 模型得学会**停下来**, 那是回答的一部分。
    一条到顶都没出 EOS 的序列, 掩码全 1, 行为和老写法一致。

    对齐关系: `logp_of` 返回 `logp[:, prompt_len-1:]`, 它第 i 列对应的目标 token 是
    `outs[:, prompt_len + i]` —— 所以这里直接切 `outs[:, prompt_len:]` 就是逐列对齐的。
    """
    tgt = outs[:, prompt_len:]
    is_eos = (tgt == eos_id).long()
    before = is_eos.cumsum(-1) - is_eos       # 这个位置**之前**出现过几次 EOS
    return (before == 0).float()


# ---------------- 采样与对数概率 ----------------

def save_atomic(obj, path):
    """先写 .tmp 再 os.replace。

    被抢占时如果正好在覆盖目标文件, 留下的是个半个文件, --resume 再也起不来。
    预训练和 SFT 都踩过这个坑 (AGENTS.md §7), 换到 GRPO 直接照搬。
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def sample_group(model, prompt_ids, G, max_new, temperature, eos_ids):
    """对 1 个 prompt 采样 G 条回答, 返回 (ids, 逐 token 旧策略对数概率)。"""
    p = prompt_ids.unsqueeze(0).repeat(G, 1)
    model.eval()
    outs = model.generate(p, max_new, temperature=temperature, eos_id=eos_ids[0])
    model.train()
    logps = logp_of(model, outs, prompt_len=prompt_ids.numel())
    return outs, logps


def logp_of(model, ids, prompt_len):
    """逐 token 对数概率 (只算回答部分)。返回 (G, T_ans)。

    ★ 这个函数**不能**挂 @torch.no_grad() —— 它是三处共用的:
      ① rollout 里算 π_θ_old   ② 参考模型的 logp   ③ 更新时的 π_θ (要反传)
      这里挂了装饰器, ③ 就没有计算图, loss.backward() 直接报
      「element 0 of tensors does not require grad」。梯度抑制是**调用方**的事,
      ①②两处自己已经包了 no_grad, 所以这里必须放开。
    """
    logits = model(ids)
    logp = F.log_softmax(logits.float(), -1)
    target = ids[:, 1:]
    logp = logp[:, :-1].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return logp[:, prompt_len - 1:]


# ---------------- GRPO 损失 ----------------

def grpo_loss(logps_new, logps_old, adv, ref_logps, beta, eps=0.2, mask=None):
    """per-token: surr = -min(ρ·A, clip(ρ)·A), kl = k3 估计。返回标量。

    mask: (G, T_ans) 0/1, 见 `answer_mask` —— 把 EOS 之后的 token 排除在损失外。
          None = 全 1 (老行为, 留着方便对照)。归一化用 mask 的和, 不是 T, 否则
          分母里混进被屏蔽的位置, 会把梯度整体缩小。
    """
    rho = (logps_new - logps_old).exp()
    surr = -torch.minimum(rho * adv, rho.clamp(1 - eps, 1 + eps) * adv)
    kl = (ref_logps - logps_new).exp() - (ref_logps - logps_new) - 1
    per_tok = surr + beta * kl
    if mask is None:
        return per_tok.mean()
    return (per_tok * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------- 数据 ----------------

def read_parquet(path):
    """直接读本地 parquet 成 {"question": [...], "answer": [...]}, 不经过 datasets。

    ★ 为什么不用 `load_dataset("parquet", data_files=...)`: 它要在 `~/.cache/huggingface`
      下落一份 arrow 缓存, 而那是**容器根盘** (多人共享的 overlay 7TB)。
      2026-10-14 根盘被占满 (可用 0 字节), datasets 于是对**没缓存过**的文件一律
      `OSError: Not enough disk space` 拒绝处理 —— 热身 SFT 就是这么一起步就死的,
      和显存、和代码都无关, 纯粹卡在一个缓存目录上。
      当时 GSM8K 还能读, 只因为它早就在缓存里了 —— 那是运气, 缓存一清就再也建不起来。
      pyarrow 直接读原文件、不落任何缓存, 顺带也绕开了 §7.3 记着的 datasets 那些坑
      (ragged list 被展平、要枚举数据集)。分词语料当初也是这么弃用 datasets 的。
    """
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pydict()


def head(ds, n):
    """取前 n 条。datasets.Dataset 和 read_parquet 返回的普通 dict 都能吃。"""
    if hasattr(ds, "select"):
        return ds.select(range(min(n, len(ds))))
    return {k: v[:n] for k, v in ds.items()}


def load_gsm8k(tokenizer_path, n_eval=256, gsm8k_dir=None):
    """GSM8K: 训练集做 rollout, 测试集前 n_eval 题做训练中评测。"""
    if gsm8k_dir:   # 本地 parquet, 直接读 (理由见 read_parquet)
        train = read_parquet(f"{gsm8k_dir}/train.parquet")
        test = read_parquet(f"{gsm8k_dir}/test.parquet")
    else:
        train = load_dataset("openai/gsm8k", "main", split="train")
        test = load_dataset("openai/gsm8k", "main", split="test")
    tok = Tokenizer.from_file(tokenizer_path)
    # 手动构造成 list[dict]: datasets.map 对 ragged list-of-lists 会展平成多行
    mk = lambda ds: [{"ids": tok.encode(PROMPT_TMPL.format(question=q)).ids, "gold": a}
                     for q, a in zip(ds["question"], ds["answer"])]
    train = mk(train)
    test = mk(head(test, n_eval))
    check_reward_fn(train[:200])
    return train, test, tok


def check_reward_fn(examples):
    """裁判自检。**两组用例, 缺一不可。**

    ① 「标准答案原文当模型输出」必须得满分 —— 被现实教出来的: gold 字段是整段
       解题过程而不是一个数字, 让奖励恒等于 0, 而 GRPO 照样跑得"很正常"
       (loss 是 0, 权重不动, 不报任何错)。没有这条, 只能靠人盯着 r=0.0000
       猜是"模型太弱"还是"裁判坏了"。

    ② ★ **占位符/空盒子必须得 0 分** —— 这是 2026-10-14 补的, 因为①**抓不住
       reward hacking**: 那次 hacking 里裁判本身没错 (标准答案照样得满分),
       错的是"有一条不用真做答案也能得分的路"。①只验了"对的能拿分",
       没验"**不去做也能拿分吗**"。判分函数要同时验这两件事。
    """
    bad = [ex for ex in examples if math_reward(ex["gold"], ex["gold"]) < 1.0]
    assert not bad, (
        f"奖励函数自检①失败: {len(bad)}/{len(examples)} 条标准答案自己都拿不到分 "
        f"(第一条: {bad[0]['gold'][-60:]!r}) —— 先修奖励函数, 别急着跑训练"
    )
    # 这些字符串都是"照抄提示词/空壳子", 一分都不该拿到
    free_points = [f"答案是 {p}" for p in
                   ["\\boxed{}", "\\boxed{...}", "\\boxed{ }", "\\boxed{…}"]
                   if format_reward(f"答案是 {p}") > 0]
    assert not free_points, (
        f"奖励函数自检②失败: 这些空壳子白拿了格式分 {free_points} —— "
        f"说明 format_reward 还是只看 \\boxed{{ 这个子串, 模型会去抄提示词里的占位符"
    )
    # 反面: 一个**真的装了答案**的盒子必须拿到格式分 (别把判据收得太紧)
    assert format_reward("答案是 \\boxed{18}") > 0, "奖励函数自检②失败: 真答案反而不给格式分"
    print(f"奖励函数自检: ① {len(examples)} 条标准答案全部判对 ✓  "
          f"② 空壳子/占位符一律 0 分 ✓")


@torch.no_grad()
def eval_pass1(model, eval_ds, tok, max_new=384):
    """贪心解码 pass@1 (温度 0 = top-1 采样近似)。"""
    correct = 0
    for ex in eval_ds:
        # 设备从模型参数上取, 不写死 "cuda": 多卡时 torchrun 不设 CUDA_VISIBLE_DEVICES,
        # 写死的 "cuda" 等于每张卡都往 cuda:0 上放张量。
        ids = torch.tensor(ex["ids"],
                           device=next(model.parameters()).device).unsqueeze(0)
        out = model.generate(ids, max_new, temperature=0.0, eos_id=IM_END)
        text = decode_completion(tok, out[0, len(ex["ids"]):])
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
    # 格式分权重。0 = 关掉格式奖励, 只留数学分 —— 奖励设计的消融臂用 (2026-10-14)。
    # ★ 实测下来每步 24 组里约 56% 有信号, 但其中大半只携带"格式"信息 (盒子有没有内容),
    #   真正带数学信号的约 3.75 组。这条臂把格式分拿掉之后信号组会掉到 ~15.6% (=pass@8),
    #   而**携带的数学信号密度不变** —— 于是它单独回答一个问题: 那些"只会格式不会做题"的
    #   梯度到底是在帮忙还是在挤占。(权重存进检查点的 args 里, 续训时会自动 diff 报警。)
    ap.add_argument("--format_w", type=float, default=1.0)
    ap.add_argument("--update_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=5)   # 续训检查点间隔 (步)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ---- 分布式: torchrun 拉起 N 张卡时, 每张卡只负责 1/N 的 prompt ----
    # 单卡直接跑 (run_grpo.sh 一贯的路子) 时 world=1, 下面每一处都退化成原来的行为。
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    if world > 1:
        dist.init_process_group("nccl")
        # 分不匀就会有卡一个样本都拿不到, 那个 rank 于是走不到 backward,
        # 其余卡在 all-reduce 上永远等它 —— 表现为"训练卡住", 不报错。
        assert args.rollout_batch % world == 0, (
            f"--rollout_batch={args.rollout_batch} 不能被卡数 {world} 整除, "
            f"会有卡分到 0 个样本然后整体挂死"
        )
    # ★ device 必须显式带 local_rank: torchrun 不会替你设 CUDA_VISIBLE_DEVICES,
    #   写 "cuda" 等于每张卡都去抢 cuda:0。单卡跑时 LOCAL_RANK=0, 行为和以前完全一样。
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_ds, eval_ds, tok = load_gsm8k(args.tokenizer, gsm8k_dir=args.gsm8k_dir)
    if is_main:
        print(f"GSM8K: 训练 {len(train_ds)} 题, 评测 {len(eval_ds)} 题 | "
              f"卡数 {world}, 每卡每步 {args.rollout_batch // world} 个 prompt | "
              f"format_w={args.format_w}")

    # ---- 策略模型 (待优化) 与参考模型 (冻结) ----
    # ★ 检查点用 map_location="cpu" 读: load_state_dict 是**复制**进参数的,
    #   传 device 的话那份 1.6 GB 的原件会一直挂到 main() 结束, 全程白占显存。
    #   这类"看不见的常驻副本"是显存账目里最容易漏掉的一笔 (实测 GRPO 就死在它上面)。
    policy = MiniR1Model(ModelConfig()).to(device)
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    policy.load_state_dict(ckpt["model"])
    ref = MiniR1Model(ModelConfig()).to(device)
    ref.load_state_dict(ckpt["model"])
    del ckpt
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    # 更新那一步要把 16 层的激活全存下来才能反传 (T≈500, G=8 时约 5 GB),
    # 而 rollout 本来就在 no_grad 下、存不存激活都一样 —— 所以只对"更新"开重算。
    # cache_passthrough=True 让 generate() 的 KV-Cache 路径绕过检查点 (见 model.py),
    # 否则同一个模型就没法既做带缓存的采样、又做省显存的更新。
    enable_grad_checkpointing(policy, cache_passthrough=True)
    # ★ DDP 只包住"更新"用的那份 policy; 所有**推理**路径 (rollout 采样 / 评测) 一律走 net,
    #   也就是绕开 DDP 的 forward。理由: DDP 每次 forward 都会在 reducer 上记一笔
    #   "这次前向等会儿要 all-reduce", 而 no_grad 下根本不会有 backward, 那一笔永远不销账,
    #   下一次 backward 会直接报 "Expected to have finished reduction in the prior iteration"。
    #   走 net 零代价 —— 同一份参数对象, 只是不做桶准备, 顺带省掉推理时的开销。
    net = policy
    if world > 1:
        policy = DDP(policy, device_ids=[local_rank])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # 自己建目录: 启动脚本里有 mkdir -p, 但换个 --out_dir 跑就会在这儿崩
    # (和 SFT 那次 torch.save 不建父目录是同一类坑, 与其报错不如自己兜住)
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(f"{args.out_dir}/grpo_log.jsonl", "a")

    # ---- 断点续训: 有检查点就自己接上 (和 run_sft.sh 一样, 不用加参数) ----
    latest = f"{args.out_dir}/grpo_latest.pt"
    start_step = 0
    if os.path.exists(latest):
        # ★ 同样走 CPU: 这份 dict 里除了 1.6 GB 权重还有 3.3 GB 优化器状态。
        #   优化器那部分 load_state_dict 是"就地采用"(dtype/device 已经对上, .to() 返回自己),
        #   但**权重那 1.6 GB 是复制进去的**, 原件必须自己放掉。
        ck = torch.load(latest, map_location="cpu")
        # optimizer 必须一起恢复: 只 load 权重的话 AdamW 的一二阶动量归零,
        # 续训头几百步会明显抖 (SFT 那次就是这么定的, 见 AGENTS.md §7.12)
        # ★ 必须 load 进 net (裸模型) 而不是 policy: 检查点里存的是 net.state_dict(),
        #   键名没有前缀; 而 DDP 的 `module` 是个子模块, 它的 load_state_dict 会去找
        #   `module.*`, 于是非前缀的键全部落进 "missing/unexpected" —— strict=True 直接 RuntimeError。
        #   (torch 官方文档只写了反方向 "DDP 的 state_dict 载进非 DDP 模型要先剥前缀",
        #    反过来的这一半没写, 但后果一样。)world=1 时 net is policy, 行为与以前完全一致。
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        old = ck.get("args", {})
        diff = [f"{k}: {old.get(k)} -> {v}" for k, v in vars(args).items()
                if k in old and old[k] != v and k not in ("out_dir", "ckpt_every")]
        if diff:
            print("⚠️ 超参和检查点里的不一样, 确认这是你要的: " + "; ".join(diff), flush=True)
        del ck
        print(f"★ 从 GRPO 检查点续训: step {start_step}/{args.max_steps}", flush=True)

    if start_step >= args.max_steps:
        if is_main:
            print(f"已训满 {args.max_steps} 步, 无事可做 (守护脚本据此退出)")
        if world > 1:
            dist.destroy_process_group()
        return

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        lr = args.lr * min(1.0, step / args.warmup_steps) if step < args.warmup_steps \
            else args.lr * 0.5 * (1 + math.cos(math.pi * step / args.max_steps))
        for g in opt.param_groups:
            g["lr"] = lr

        # ---- 1. Rollout: 采样 + 奖励 + 优势 ----
        samples = []      # (prompt_ids, answer_ids, logps_old, adv)
        # ★ 题目由「步号派生的种子」抽, 不用全局 RNG。为什么: 全局 RNG 会被 model.generate
        #   按各卡的生成量消耗掉, 于是同一个 (seed, step) 在**重启后会抽出另一批题** ——
        #   而续训只记"已训到第几步"。步数对得上、题目对不上, 不报任何错。
        #   (纠正一条早期写过头的话: 这里原来的注释说「各卡独立抽取会让 24 道题重复」。
        #    7473 里抽 24 个, 两次独立抽样撞车的概率只有 ~1.9%, 实际几乎不会重复;
        #    按 rank 派生种子、再连续切分其实也是对的。真正只有"可复现"这一条是硬要求。)
        #   同一种子 ⇒ 各卡得到**同一份** randperm ⇒ 再按 rank::world 交错切分, 天然不重不漏。
        gen = torch.Generator().manual_seed(args.seed * 1000003 + step)
        idxs = torch.randperm(len(train_ds), generator=gen)[:args.rollout_batch][rank::world]
        total_r, n_signal = 0.0, 0        # 原始奖励 / 组内奖励有差异的组数
        for i in idxs:
            ex = train_ds[int(i)]
            prompt = torch.tensor(ex["ids"], device=device)
            outs, logps = sample_group(net, prompt, args.G, args.max_new,
                                       args.temperature, eos_ids=[IM_END])
            # ★ 按 EOS 截断再解码: 不截断的话, 先停下来的采样后面那段"幻觉"
            #   会盖掉真答案 (见 decode_completion 的注释)。
            texts = [decode_completion(tok, outs[j, len(ex["ids"]):]) for j in range(args.G)]
            # format_w=1.0 时和以前逐位一致; 0 则只留数学分 (见 --format_w 的注释)
            rs = [math_reward(t, ex["gold"]) + args.format_w * format_reward(t) for t in texts]
            rs = torch.tensor(rs, device=device)
            # ★ 统计要在**归一化之前**做, 而且要盯着「有信号组」这个数:
            #   组内优势按定义均值恒为 0, 拿 adv.mean() 当指标等于每步都打印 0,
            #   根本看不出 GRPO 有没有在学。而 std=0 的组 (G 条奖励全一样, 比如
            #   全错、或全都会用 \boxed{} 但都算错) 优势全是 0, 这组数据白采。
            #   这个比例接近 100% 就说明奖励设计或 max_new 有问题, 别干等 300 步。
            total_r += rs.mean().item()
            n_signal += int(rs.std() > 1e-6)
            adv = (rs - rs.mean()) / (rs.std() + 1e-8)
            samples.append((prompt, outs, logps, adv))

        # ---- 2. 参考模型 logp (每步对这批回答算一次) ----
        # 注意: prompt_len 必须取该样本自己的, 不能用循环外的 ex —— ex 是 rollout 循环
        # 泄漏出来的最后一个 prompt, 拿它切片会让每条 ref_logps 的宽度都对不上
        # logps_new (宽度 max_new), grpo_loss 里相减直接 shape 报错。
        with torch.no_grad():
            ref_logps = [logp_of(ref, outs, prompt.numel()) for prompt, outs, _, _ in samples]

        # ---- 3. K 轮梯度更新 ----
        total_loss = 0.0
        for _ in range(args.update_epochs):
            for j, (prompt, outs, logps_old, adv) in enumerate(samples):
                full = outs
                logps_new = logp_of(policy, full, prompt.numel())
                loss = grpo_loss(logps_new, logps_old, adv[:, None],
                                 ref_logps[j], args.beta, args.clip_eps,
                                 mask=answer_mask(full, prompt.numel()))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()

        # 每张卡只统计了自己那 1/N 组, 打印前先汇总 —— 否则「有信号组 x/y」和 r
        # 会随卡数变化, 跟之前那 300 步的记录没法横向比。
        n_groups = len(samples)
        if world > 1:
            t = torch.tensor([total_r, float(n_signal)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_r, n_signal, n_groups = t[0].item(), int(t[1].item()), args.rollout_batch

        n = len(samples) * args.update_epochs
        r_mean = total_r / n_groups
        # ★ 峰值显存必须每步打出来。这次 GRPO 崩了 51 次, 日志上只有一行
        #   「torch.OutOfMemoryError」—— 看不出是"本来就快满了"还是"某一步突然涨上去",
        #   只能靠事后翻 traceback 里那次失败的 2.52 GiB 反推。
        #   peak 是**自进程启动以来的最大值**, 拿它跟 24G 卡的可用显存一比就知道还有多少余量;
        #   reset_peak 放在打印之后, 于是这个数就是"本步消耗了多少"。
        peak = 0.0
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
        if is_main:   # 多卡时只有 rank 0 打印/写日志, 否则一行日志会被印 N 遍
            print(f"step {step} | loss {total_loss/n:.4f} | r {r_mean:.4f} | "
                  f"有信号组 {n_signal}/{n_groups} | lr {lr:.2e} | "
                  f"{time.time()-t0:.0f}s | 峰值 {peak:.1f}G", flush=True)
            logf.write(json.dumps({"step": step, "loss": total_loss / n,
                                   "reward": r_mean, "signal_groups": n_signal,
                                   "lr": lr, "peak_gib": round(peak, 2)}) + "\n")
            logf.flush()

        # 定期存续训检查点 (约 4.9 GB: 权重 1.6G + AdamW 两个动量 3.2G)。
        # GRPO 和 SFT 不同, 没有「跳到第几个样本」的记账问题 —— 每步的 prompt 是
        # 当场 randperm 抽的, 续训时 RNG 接不上也无所谓, 抽到哪批本来就是随机的。
        # ★ 存检查点必须用 net 而不是 policy: DDP 包装后的 state_dict() 键名会带
        #   "module." 前缀, 存下去下次 --resume 就 load 不回来 (满屏 unexpected keys)。
        #   只有 rank 0 写盘; 各卡参数由 DDP 保证一致, 写谁的都一样。
        if is_main and (step + 1) % args.ckpt_every == 0:
            save_atomic({"step": step, "model": net.state_dict(),
                         "optimizer": opt.state_dict(), "args": vars(args)}, latest)

        if (step + 1) % args.eval_every == 0:
            # 评测只在 rank 0 跑 (256 题 × 采样, 4~5 分钟)。其余卡会先冲进下一步,
            # 然后在那个 backward 的 all-reduce 上等 rank 0 —— 不死锁, 但也帮不上忙,
            # 所以这一段的时间就是实打实的额外开销 (和单卡一样)。
            if is_main:
                acc = eval_pass1(net, eval_ds, tok)
                print(f"  >> GSM8K pass@1 = {acc:.4f}", flush=True)
                logf.write(json.dumps({"step": step, "eval_pass1": acc}) + "\n")
                logf.flush()
                save_atomic({"step": step, "model": net.state_dict()},
                            f"{args.out_dir}/grpo.pt")

    # 收尾也存一次: 不满 ckpt_every 的零头步数不进检查点的话, 续训会白跑一遍
    if is_main:
        save_atomic({"step": args.max_steps - 1, "model": net.state_dict(),
                     "optimizer": opt.state_dict(), "args": vars(args)}, latest)
        print("GRPO 训练完成")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
