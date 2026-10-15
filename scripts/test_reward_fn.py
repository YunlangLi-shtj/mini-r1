#!/usr/bin/env python3
"""奖励函数的回归测试 (纯 CPU, 1 秒, 不抢卡)。

**为什么要有这个文件**: GRPO 那次跑完, 日志上每个指标都在变好 (r 逼近 0.1 上限、
format_ratio 涨到 1.00), 而模型其实是在**抄提示词里的 `\\boxed{...}` 占位符**白拿格式分。
裁判本身没坏 —— 标准答案喂进去照样得满分, 所以 `check_reward_fn` 的
「满分样本自检」**完全抓不住它**。

这类 bug 的判据只有一条, 就是下面这条 CORE:
    **把提示词原样当成模型的回答吐出来, 必须得 0 分。**

跑法:  CUDA_VISIBLE_DEVICES="" python scripts/test_reward_fn.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from train_grpo import (PROMPT_TMPL, math_reward, format_reward,
                        extract_answer, has_boxed_answer, gold_answer)

GOLD = "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 18"
fails = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'✓' if ok else '✗'}] {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


print("=" * 72)
print("① 正常路径: 该给分的要给分 (别把判据收得太紧)")
print("=" * 72)
check("gold_answer 取 #### 之后", gold_answer(GOLD), "18")
check("标准答案原文当输出 → math 满分", math_reward(GOLD, GOLD), 1.0)
check("\\boxed{18} 答对", math_reward("推理...\\boxed{18}", GOLD), 1.0)
check("\\boxed{18} 拿到格式分", format_reward("推理...\\boxed{18}"), 0.1)
check("1,234 / 1234 归一化后相等", math_reward("\\boxed{1,234}", "#### 1234"), 1.0)
check("分数形式", math_reward("\\boxed{\\frac{1}{2}}", "#### \\frac{1}{2}"), 1.0)

print()
print("=" * 72)
print("② ★ CORE: 抄提示词必须一分都拿不到 (2026-10-14 那次 hacking 的入口)")
print("=" * 72)
# 模型干的事: 把提示词里那个占位符原样吐出来
echo_prompt = PROMPT_TMPL.format(question="Janet has 16 eggs.")
check("提示词原文当输出 → 格式分 0", format_reward(echo_prompt), 0.0)
check("提示词原文当输出 → math 0", math_reward(echo_prompt, GOLD), 0.0)
# 实测的那条原始输出
check("实测 hack 输出 → 格式分 0",
      format_reward("4. cannot be true within \\boxed{...}\n5. \n"), 0.0)
for p in ["\\boxed{}", "\\boxed{ }", "\\boxed{...}", "\\boxed{…}", "\\boxed{. . .}"]:
    check(f"空壳子 {p} → 格式分 0", format_reward(f"所以答案是 {p}"), 0.0)

print()
print("=" * 72)
print("③ 提示词里不能有可照抄的**填好的**盒子")
print("=" * 72)
# ⚠️ 必须检查 **.format() 之后**的文本 —— 那才是模型看到的东西。
#    检查原始字面量是空过的: 源码里是 `\\boxed{{}}`, 正则匹配不到 `\boxed{` 后面跟 `}`,
#    findall 返回 [], `all(...)` 对空列表恒为 True —— 一条永远会通过的断言。
rendered = PROMPT_TMPL.format(question="Janet has 16 eggs.")
boxes = __import__("re").findall(r"\\boxed\{([^{}]*)\}", rendered)
print(f"  渲染后提示词里的盒子内容: {boxes!r}")
check("渲染后确实提到了 \\boxed (模型得知道要用它)", len(boxes) > 0, True)
check("提示词里的盒子全是空的", all(b.strip() == "" for b in boxes), True)
check("照抄整个提示词拿不到格式分", format_reward(rendered), 0.0)
check("照抄整个提示词拿不到 math 分", math_reward(rendered, GOLD), 0.0)

print()
print("=" * 72)
print("④ 兜底路径: 盒子没装东西时, 该退回到数字")
print("=" * 72)
# 这是 extract_answer 的一个真 bug: 原来用 [^{}]+ 会匹配上省略号,
# 于是 "\boxed{...} 答案是 18" 会被抠出 "..." 而不是 18
check("\\boxed{...} 不该挡住后面的真数字",
      extract_answer("\\boxed{...} 所以最终答案是 18"), "18")
check("同上, math 分要能拿到", math_reward("\\boxed{...} 所以最终答案是 18", GOLD), 1.0)
check("全是占位符且无数字 → None", extract_answer("答案是 \\boxed{...}"), None)
check("多个盒子取最后一个真的",
      extract_answer("\\boxed{...} 先猜 3, 再算 \\boxed{18}"), "18")

print()
print("=" * 72)
if fails:
    print(f"❌ {len(fails)} 项没过: {fails}")
    sys.exit(1)
print("✅ 奖励函数回归测试全部通过")
print("=" * 72)
