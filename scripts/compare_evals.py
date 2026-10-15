#!/usr/bin/env python3
"""两个评测报告的**配对**比较 —— 判断「GRPO 到底有没有提升」用这个, 不要看两个 pass@1 的差。

    python scripts/compare_evals.py reports/gsm8k_sft_math6.json reports/gsm8k_grpo.json

为什么必须配对 (2026-10-14 的教训):
    256 题时 pass@1 的标准差 ≈ sqrt(256×0.015×0.985) ≈ 1.4 题, 也就是 ±0.6 个百分点。
    两个模型各跑一次, 差值要 >1.7 个百分点才算「超出噪声」—— 而我们整个效应量就在这个量级上。
    所以(A)不配对的做法只能得出"无显著差异", 这个结论是真的但**信息量很低**:
    它分不清"确实没提升"和"样本不够".

    但两个模型答的是**同一批题**。同一道题, 一个对一个错 —— 这种"不一致对"才是信息所在,
    而两个模型都答错的题 (理论上占大头) 对差值没有任何贡献。
    配对之后有效样本量从"题数"变成"不一致对数", 灵敏度可以高好几倍。

这里给两个检验:
    [1] pass@8 的 McNemar 精确检验 —— 每道题是"8 条里至少一条对"的 0/1, 配对的二分类。
    [2] 逐样本正确率的**配对 bootstrap** 置信区间 —— 按题重采样, 保留题内的相关性。

⚠️ 别把它当成"p<0.05 就是提升": 这一轮的效应量本来就小, 而且
   eval 的 n 与训练日志里的 n 是两个口径 (见 AGENTS.md §8 第 1 条)。
   这里的结论只对"这批题 + 这个采样口径"负责。
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_grpo import normalize      # noqa: E402  (路径要先插进去)


def load(path):
    d = json.load(open(path))
    det = d["details"]
    # 按题干对齐: 两次评测的题序理论上一样, 但按内容对齐才不依赖这个假设
    out = {}
    for x in det:
        out[x["q"]] = x
    return d, out


def per_q_correct(x):
    """这道题 8 条采样里对了几条 (0..8)。返回 (对数, 总条数)。"""
    g = normalize(x["gold"])
    ans = x["answers"]
    return sum(1 for a in ans if normalize(a) == g), len(ans)


def mcnemar_exact(b, c):
    """精确二项检验 (双侧), H0: 不一致对里两个方向等概率。b+c=0 时 p=1。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def boot_ci(diffs, n_boot=20000, seed=0):
    """按题重采样的配对 bootstrap: diffs[i] = 第 i 题上 B 的正确率 − A 的正确率。"""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return lo, hi


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pa, pb = sys.argv[1], sys.argv[2]
    da, A = load(pa)
    db, B = load(pb)

    # 按 A 里的原顺序取交集 —— dict 的键序就是插入序, 天然稳定
    common = [q for q in A if q in B]
    print(f"A = {pa}   (n={len(A)})")
    print(f"B = {pb}   (n={len(B)})")
    print(f"✓ 对齐到同一批题: {len(common)} 道"
          + ("" if len(common) == len(A) == len(B) else "   ⚠️ 两次的题集不一样, 只比交集"))

    if not common:
        sys.exit("没有共同题目, 无法配对")

    nA = sum(per_q_correct(A[q])[0] for q in common)
    nB = sum(per_q_correct(B[q])[0] for q in common)
    tot = sum(per_q_correct(A[q])[1] for q in common)
    a_any = sum(1 for q in common if per_q_correct(A[q])[0] > 0)
    b_any = sum(1 for q in common if per_q_correct(B[q])[0] > 0)
    N = len(common)

    print()
    print(f"  逐样本正确率   A {nA}/{tot} = {nA/tot:.4%}   B {nB}/{tot} = {nB/tot:.4%}")
    print(f"  pass@8 (至少一条对)  A {a_any}/{N} = {a_any/N:.4%}   B {b_any}/{N} = {b_any/N:.4%}")

    # --- [1] pass@8 的 McNemar (配对二分类) ---
    both = sum(1 for q in common if per_q_correct(A[q])[0] > 0 and per_q_correct(B[q])[0] > 0)
    only_a = sum(1 for q in common if per_q_correct(A[q])[0] > 0 and per_q_correct(B[q])[0] == 0)
    only_b = sum(1 for q in common if per_q_correct(A[q])[0] == 0 and per_q_correct(B[q])[0] > 0)
    neither = N - both - only_a - only_b
    p = mcnemar_exact(only_a, only_b)
    print()
    print("=== [1] pass@8 的 McNemar 精确检验 (配对) ===")
    print(f"  两种都答对 {both} | 只有 A 对 {only_a} | 只有 B 对 {only_b} | 两种都错 {neither}")
    print(f"  不一致对 {only_a + only_b} 对  →  p = {p:.4g}")
    if p < 0.05:
        better = "B" if only_b > only_a else "A"
        print(f"  ⇒ 差异显著 (p<0.05), {better} 更好: 净多 {abs(only_b - only_a)} 题 "
              f"= {abs(only_b - only_a)/N:.2%} (绝对值)")
    else:
        print(f"  ⇒ 不显著。注意**不显著 ≠ 没差异**: 不一致对只有 {only_a + only_b} 对, "
              f"这点样本量只能排除大约 ±{1.96*math.sqrt(max(only_a+only_b,1))/N:.1%} 以上的效应")

    # --- [2] 逐样本正确率的配对 bootstrap ---
    diffs = [(per_q_correct(B[q])[0] - per_q_correct(A[q])[0]) / per_q_correct(A[q])[1]
             for q in common]
    mean_d = sum(diffs) / N
    lo, hi = boot_ci(diffs, seed=0)
    print()
    print("=== [2] 逐样本正确率的配对 bootstrap (按题重采样 2 万次) ===")
    print(f"  B − A = {mean_d:+.4%}   95% CI [{lo:+.4%}, {hi:+.4%}]")
    print("  ⇒ " + ("区间跨 0, 该口径下没有可测差异"
                    if lo <= 0 <= hi else "区间不含 0, 该口径下有差异"))

    # --- 格式 (只在报告里有这个字段时提一句) ---
    for tag, d in (("A", da), ("B", db)):
        r = d.get("result", {})
        if "fmt_per_sample" in r:
            print(f"  {tag} 格式: 逐样本 {r['fmt_per_sample']:.4f}, "
                  f"组内有差异 {r.get('fmt_signal_groups', float('nan')):.4f}")
    print()
    print("⚠️ 结论只对这批题和这个采样口径负责。要更强的结论就把 n_eval 开到 0 (全量 1319)。")


if __name__ == "__main__":
    main()
