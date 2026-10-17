# /data/liying/ms-swift/yli/util_pys/pick-workflow/7-1-pick-intra-duplicate.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
duplicate：
对结果去重

功能：分别挑选sent和ctx任务内部的（输入，佳翻译，差翻译）对；
输入：一个jsonl文件；其中18条jsonl数据为一组，如：（每组包含8个ctx/sample，1个ctx/greedy，8个sent/sample，1个sent/greedy）;dcomet有时候是null，大部分时候是0~1的一个小数；
输入路径：/data/liying/ms-swift/yli/256ctx/90mix-sample-validset-merged/eval/merged-4ways.flat.clean.id.with_scores.win5.jsonl
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/greedy", "comet": 0.9214195609092712, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.511634111404419, "bleu": 21.200626759025184, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.8771191239356995, "bleu": 52.53819788848316, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.9214195609092712, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.8698757290840149, "bleu": 14.923729480049115, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.9214195609092712, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.9214195609092712, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.6172788739204407, "bleu": 11.868405219520975, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "ctx/sample", "comet": 0.8777967095375061, "bleu": 33.03164318013809, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/greedy", "comet": 0.9214195609092712, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.8698757290840149, "bleu": 14.923729480049115, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.6172788739204407, "bleu": 11.868405219520975, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.921419620513916, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.921419620513916, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.8947271704673767, "bleu": 52.53819788848316, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.921419620513916, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.921419620513916, "bleu": 43.167001068522545, "dcomet": null}
{"doc_id": 1, "src_sent_id": 0, "prompt": "..", "input":"...","predict":"...","task": "sent/sample", "comet": 0.921419620513916, "bleu": 43.167001068522545, "dcomet": null}

具体功能：把comet/bleu/dcomet或者这几个数据加权后的值（要能正确处理null，把null映射成dcomet平均值），得到一个优势值，对同一个(doc_id,src_sent_id,task.split('/')[0])的数据进行比较，得到（佳翻译，差翻译）；
挑选的规则有：
- predict部分单词数量
- 优势值的绝对范围（如果是bleu，绝对范围xx，如果是comet，绝对范围xxx..）
- 佳翻译、差翻译之间优势值的相对大小范围（如果是bleu，范围xx，如果是comet，范围xxx..）
- 其他的规则，请帮我想一想；
代码要求：
- 把根据什么进行挑选，做成可以切换的模块；
- 挑选的规则都是超参数，由命令行传入；
- 把筛掉扔掉的数据为什么不合格，分类目统计，并记录在最后输出的readme
- 把最后保留的数据，统计它的优势值之差落在什么区间，记录在最后输出的readme
输出：挑选后的数据pair的jsonl，
包含{y_win,y_lose,x_win(即prompt win),x_lose(即prompt lose),misson(是'intra'或者'cross'，这里，由命令行统一传入为intra),win_value(赢家的优势值),lose_value(输家的优势值),rule(优势值是'comet'/'bleu'/'dcomet'/'weighted')}
"""
"""
从混合 sent/ctx 翻译结果中筛选偏好学习 pair 的脚本。

功能概要：
- 读取 jsonl（每行一个样本），按 (doc_id, src_sent_id, task_prefix) 分组：
    task_prefix = task.split('/')[0]  # "sent" 或 "ctx"
- 对每个 group 内多条候选翻译计算“优势值”（score），然后：
    - 选 score 最高的为 winner
    - 选 score 最低的 K 个为 losers（bottom-k）
    - 对每个 (winner, loser) 应用一系列过滤规则
- 输出：
    - pair jsonl：{y_win, y_lose, x_win, x_lose, mission, win_value, lose_value, rule}
    - readme 文本：统计筛掉原因 + gap 直方图等

注意：
- dcomet 为 null 时，用全局平均 dcomet 代替。
- x_win / x_lose 使用原始的 prompt 字段（你的原始定义）。

python /data/liying/ms-swift/yli/util_pys/pick-workflow/7-1-pick-intra.py \
  --input /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/merged-4ways.flat.jsonl \
  --output /data/liying/beiqu-swift/yli/dataset/256intra-2wsft-v1.0/intra.dcomet-and-comet.jsonl \
  --readme /data/liying/beiqu-swift/yli/dataset/256intra-2wsft-v1.0/intra.dcomet-and-comet.jsonl.readme.txt \
  --rule weighted \
  --w-comet 0.5 \
  --w-bleu  0.0 \
  --w-dcomet 0.5 \
  --mission intra \
  --min-len 6 \
  --max-len 200 \
  --min-win-score 0.30 \
  --min-gap 0.002 \
  --max-gap 0.1 \
  --max-pairs-per-group 1 \
  --gap-bins "0,0.002,0.005,0.006,0.008,0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090"

"""

import argparse
import json
import math
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional
import sys
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select intra-task preference pairs from translation candidates."
    )

    # I/O
    parser.add_argument("--input", type=str, required=True,
                        help="Input jsonl file path.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output jsonl file path for preference pairs.")
    parser.add_argument("--readme", type=str, required=True,
                        help="Output readme txt/markdown file path.")
    parser.add_argument("--mission", type=str, default="intra",
                        help="Mission field in output json (default: intra).")

    # Score rule
    parser.add_argument("--rule", type=str, choices=["comet", "bleu", "dcomet", "weighted"],
                        default="comet", help="Which metric to use as advantage value.")
    parser.add_argument("--w-comet", type=float, default=1.0,
                        help="Weight for comet when rule=weighted.")
    parser.add_argument("--w-bleu", type=float, default=0.0,
                        help="Weight for bleu when rule=weighted.")
    parser.add_argument("--w-dcomet", type=float, default=0.0,
                        help="Weight for dcomet when rule=weighted.")

    # Length-related filters
    parser.add_argument("--min-len", type=int, default=1,
                        help="Minimum token count (after split on whitespace) "
                             "for both y_win and y_lose.")
    parser.add_argument("--max-len", type=int, default=None,
                        help="Maximum token count for y_win/y_lose (optional).")
    parser.add_argument("--max-len-ratio", type=float, default=None,
                        help="Maximum ratio max(len_win, len_lose) / max(1, min(len_win, len_lose)).")

    # Absolute score constraints
    parser.add_argument("--min-win-score", type=float, default=None,
                        help="Minimum allowed winner score.")
    parser.add_argument("--max-lose-score", type=float, default=None,
                        help="Maximum allowed loser score.")

    # Relative gap constraints
    parser.add_argument("--min-gap", type=float, default=None,
                        help="Minimum gap = win_score - lose_score.")
    parser.add_argument("--max-gap", type=float, default=None,
                        help="Maximum gap = win_score - lose_score.")

    # Per-group pair count
    parser.add_argument("--max-pairs-per-group", type=int, default=1,
                        help="Maximum number of (winner, loser) pairs per group. "
                             "By design: top-1 vs bottom-k.")

    # Gap histogram bins
    parser.add_argument(
        "--gap-bins",
        type=str,
        default="0,0.02,0.05,0.1,0.2,1.0",
        help="Comma-separated bin edges for gap histogram (last bin is [last, +inf)). "
             "Example for COMET: '0,0.02,0.05,0.1,0.2,1.0'"
    )

    return parser.parse_args()


def safe_float(x: Any) -> Optional[float]:
    """Convert to float, return None if impossible."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_global_dcomet_mean(input_path: str) -> float:
    """First pass: compute global mean of dcomet (ignoring null)."""
    total = 0.0
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dcomet = safe_float(obj.get("dcomet"))
            if dcomet is not None:
                total += dcomet
                count += 1
    if count == 0:
        # 没有任何 dcomet，有点反常，但仍然给一个默认值
        return 0.0
    return total / count


def compute_score(
    item: Dict[str, Any],
    rule: str,
    w_comet: float,
    w_bleu: float,
    w_dcomet: float,
    dcomet_mean: float,
) -> Optional[float]:
    """Compute advantage score for one item according to rule.

    Returns None if score is invalid (e.g., missing required metric).
    """
    comet = safe_float(item.get("comet"))
    bleu = safe_float(item.get("bleu"))
    raw_dcomet = safe_float(item.get("dcomet"))
    dcomet = raw_dcomet if raw_dcomet is not None else dcomet_mean

    if rule == "comet":
        if comet is None:
            return None
        return comet
    elif rule == "bleu":
        if bleu is None:
            return None
        return bleu
    elif rule == "dcomet":
        # 已经用平均值填补
        return dcomet
    elif rule == "weighted":
        # 缺失的指标当做 0 处理（除了 dcomet 已经填补）
        comet_val = comet if comet is not None else 0.0
        bleu_val = bleu if bleu is not None else 0.0
        score = w_comet * comet_val + w_bleu * bleu_val + w_dcomet * dcomet
        return score
    else:
        # 不应该发生
        return None


def tokenize_len(text: str) -> int:
    """简单的 token 统计：按空格分割。
    对于中文没那么准确，但实现简单且可复用。
    """
    if not text:
        return 0
    return len(text.strip().split())


def apply_pair_filters(
    winner: Dict[str, Any],
    loser: Dict[str, Any],
    min_len: int,
    max_len: Optional[int],
    max_len_ratio: Optional[float],
    min_win_score: Optional[float],
    max_lose_score: Optional[float],
    min_gap: Optional[float],
    max_gap: Optional[float],
) -> Tuple[bool, Optional[str], float]:
    """对一个 (winner, loser) 应用过滤规则。

    返回:
        passed (bool): 是否通过过滤
        reason (str | None): 若未通过，返回第一条触发的原因标签
        gap (float): win_score - lose_score
    """
    win_text = (winner.get("predict") or "").strip()
    lose_text = (loser.get("predict") or "").strip()
    win_score = float(winner["__score__"])
    lose_score = float(loser["__score__"])

    # identical predict
    if win_text == lose_text:
        return False, "identical_predict", win_score - lose_score

    # length filters
    len_win = tokenize_len(win_text)
    len_lose = tokenize_len(lose_text)
    if len_win < min_len or len_lose < min_len:
        return False, "too_short", win_score - lose_score

    if max_len is not None and (len_win > max_len or len_lose > max_len):
        return False, "too_long", win_score - lose_score

    if max_len_ratio is not None:
        min_l = max(1, min(len_win, len_lose))
        max_l = max(len_win, len_lose)
        if max_l / min_l > max_len_ratio:
            return False, "length_ratio_too_large", win_score - lose_score

    # gap constraints
    gap = win_score - lose_score
    # 理论上已经按分数降序排序，gap 应该 >= 0，但还是防一手：
    if gap < 0:
        return False, "negative_gap_unexpected", gap

    if min_gap is not None and gap < min_gap:
        return False, "gap_too_small", gap

    if max_gap is not None and gap > max_gap:
        return False, "gap_too_large", gap

    # absolute score constraints
    if min_win_score is not None and win_score < min_win_score:
        return False, "win_score_too_low", gap

    if max_lose_score is not None and lose_score > max_lose_score:
        return False, "lose_score_too_high", gap

    return True, None, gap


def parse_gap_bins(gap_bins_str: str) -> List[float]:
    edges = [float(x) for x in gap_bins_str.split(",") if x.strip() != ""]
    edges = sorted(edges)
    if len(edges) == 0:
        # fallback
        edges = [0.0, 1.0]
    return edges


def bin_gap(gap: float, bin_edges: List[float]) -> int:
    """返回 gap 落到的 bin index.

    定义：
        [edge[i], edge[i+1]) 对于 i < n-1
        [edge[n-1], +inf) 对于最后一个 bin
    """
    n = len(bin_edges)
    for i in range(n - 1):
        if bin_edges[i] <= gap < bin_edges[i + 1]:
            return i
    return n - 1


def main():
    args = parse_args()


    # mkdir -p for output/readme
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    readme_dir = os.path.dirname(args.readme)
    if readme_dir:
        os.makedirs(readme_dir, exist_ok=True)
    # 1) 预先计算 dcomet 全局平均值
    dcomet_mean = compute_global_dcomet_mean(args.input)

    # 2) 读入全部数据，按 group 分组
    groups: Dict[Tuple[Any, Any, str], List[Dict[str, Any]]] = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at line {line_idx}: {e}")

            task = obj.get("task", "")
            task_prefix = str(task).split("/")[0]  # "sent" or "ctx"
            doc_id = obj.get("doc_id")
            src_sent_id = obj.get("src_sent_id")
            key = (doc_id, src_sent_id, task_prefix)
            groups[key].append(obj)

    # 3) 遍历 group，构造 pair
    gap_bin_edges = parse_gap_bins(args.gap_bins)
    gap_hist_counts = [0 for _ in gap_bin_edges]  # 最后一个 bin 是 [last, +inf)

    reject_stats: Counter = Counter()
    total_pair_attempts = 0
    total_pairs_kept = 0
    total_groups = len(groups)
    groups_with_any_pair = 0
    # 统计保留 pair 的 task 分布：四种 task 原样计数
    kept_task_counter = Counter()

    with open(args.output, "w", encoding="utf-8") as fout:
        seen_objs=[]
        quchong=0
        for key, cand_list in groups.items():
            # key: (doc_id, src_sent_id, task_prefix)
            doc_id, src_sent_id, task_prefix = key

            # 为这个 group 中的每条候选计算 score
            scored_cands: List[Dict[str, Any]] = []
            for item in cand_list:
                score = compute_score(
                    item,
                    rule=args.rule,
                    w_comet=args.w_comet,
                    w_bleu=args.w_bleu,
                    w_dcomet=args.w_dcomet,
                    dcomet_mean=dcomet_mean,
                )
                if score is None or math.isnan(score):
                    reject_stats["invalid_score_candidate"] += 1
                    continue
                item = dict(item)  # 复制防止污染原数据
                item["__score__"] = float(score)
                scored_cands.append(item)

            if len(scored_cands) < 2:
                reject_stats["no_valid_candidate_in_group"] += 1
                continue

            # 按 score 降序排序
            scored_cands.sort(key=lambda x: x["__score__"], reverse=True)

            winner = scored_cands[0]
            losers_pool = scored_cands[1:]  # 从中选 bottom-k

            if len(losers_pool) == 0:
                reject_stats["no_loser_in_group"] += 1
                continue

            # bottom-k: 从尾部取最多 K 个
            k = max(1, args.max_pairs_per_group)
            if len(losers_pool) <= k:
                chosen_losers = losers_pool
            else:
                # 取得分最低的 k 个
                chosen_losers = losers_pool[-k:]

            group_kept_pairs = 0

            for loser in chosen_losers:
                total_pair_attempts += 1

                passed, reason, gap = apply_pair_filters(
                    winner=winner,
                    loser=loser,
                    min_len=args.min_len,
                    max_len=args.max_len,
                    max_len_ratio=args.max_len_ratio,
                    min_win_score=args.min_win_score,
                    max_lose_score=args.max_lose_score,
                    min_gap=args.min_gap,
                    max_gap=args.max_gap,
                )

                if not passed:
                    reject_stats[reason] += 1
                    continue

                # 通过所有过滤，写到输出 jsonl
                out_obj = {
                    "x_win": winner.get("prompt"),
                    "x_lose": loser.get("prompt"),
                    "y_win": winner.get("predict"),
                    "y_lose": loser.get("predict"),
                    # x_win / x_lose 按你的原始定义用 prompt
                    "mission": args.mission,
                    "win_value": winner["__score__"],
                    "lose_value": loser["__score__"],
                    "rule": args.rule,
                    # 你要的话也可以加一点上下文信息，方便 debug：
                    "meta": {
                        "doc_id": doc_id,
                        "src_sent_id": src_sent_id,
                        "task_prefix": task_prefix,
                        "task_win": winner.get("task"),
                        "task_lose": loser.get("task"),
                    }
                }
                if out_obj not in seen_objs:
                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    seen_objs.append(out_obj)
                    # 统计 gap 直方图
                    gap_idx = bin_gap(gap, gap_bin_edges)
                    gap_hist_counts[gap_idx] += 1

                    total_pairs_kept += 1
                    group_kept_pairs += 1
                    # 记录 winner/loser 的 task
                    task_win = winner.get("task", "")
                    # task_lose = loser.get("task", "")

                    kept_task_counter[task_win] += 1
                    # kept_task_counter[task_lose] += 1
                else:
                    quchong=quchong+1
            if group_kept_pairs == 0:
                reject_stats["no_valid_pair_in_group"] += 1
            else:
                groups_with_any_pair += 1

    # 4) 写 readme
    with open(args.readme, "w", encoding="utf-8") as f:
        f.write("# Preference Pair Selection Report\n\n")
        # === 自动记录完整运行命令 ===
        full_cmd = " ".join([sys.executable] + sys.argv)
        f.write("## Command Used\n\n")
        f.write("```\n")
        f.write(full_cmd + "\n")
        f.write("```\n\n")

        f.write("## Configuration\n\n")
        cfg = {
            "input": args.input,
            "output": args.output,
            "mission": args.mission,
            "rule": args.rule,
            "w_comet": args.w_comet,
            "w_bleu": args.w_bleu,
            "w_dcomet": args.w_dcomet,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "max_len_ratio": args.max_len_ratio,
            "min_win_score": args.min_win_score,
            "max_lose_score": args.max_lose_score,
            "min_gap": args.min_gap,
            "max_gap": args.max_gap,
            "max_pairs_per_group": args.max_pairs_per_group,
            "gap_bins": gap_bin_edges,
            "global_dcomet_mean": dcomet_mean,
        }
        for k, v in cfg.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Overall Statistics\n\n")
        f.write(f"- Total groups: {total_groups}\n")
        f.write(f"- Groups with at least one kept pair: {groups_with_any_pair}\n")
        f.write(f"- Total pair attempts (before filters): {total_pair_attempts}\n")
        f.write(f"- Total kept pairs: {total_pairs_kept}\n")
        keep_rate = (total_pairs_kept / total_pair_attempts) if total_pair_attempts > 0 else 0.0
        f.write(f"- Keep rate: {keep_rate:.4f}\n")
        f.write(f"- 去重: {quchong}条\n")

        f.write("\n## Rejection Reasons\n\n")
        total_rejects = sum(reject_stats.values())
        if total_rejects == 0:
            f.write("No rejected pairs.\n")
        else:
            for reason, cnt in reject_stats.most_common():
                frac = cnt / total_rejects if total_rejects > 0 else 0.0
                f.write(f"- {reason}: {cnt} ({frac:.4%})\n")
        #task
        f.write("\n## Task Distribution of Kept Pairs (only winner)\n\n")

        total_tasks_counted = sum(kept_task_counter.values())
        if total_tasks_counted == 0:
            f.write("No kept pairs, so no task distribution.\n\n")
        else:
            for task, cnt in kept_task_counter.most_common():
                frac = cnt / total_tasks_counted
                f.write(f"- {task}: {cnt} ({frac:.4%})\n")
        f.write("\n")

        f.write("\n## Gap Histogram (win_value - lose_value)\n\n")
        f.write(
            "Each bin is counted over kept pairs only. "
            "Definition: [edge[i], edge[i+1]) for i < n-1, and [edge[n-1], +inf) for the last bin.\n\n"
        )

        if total_pairs_kept == 0:
            f.write("No kept pairs. Histogram is empty.\n")
        else:
            for i, edge in enumerate(gap_bin_edges):
                cnt = gap_hist_counts[i]
                frac = cnt / total_pairs_kept if total_pairs_kept > 0 else 0.0
                if i < len(gap_bin_edges) - 1:
                    next_edge = gap_bin_edges[i + 1]
                    f.write(
                        f"- [{edge:.6g}, {next_edge:.6g}): {cnt} pairs ({frac:.4%})\n"
                    )
                else:
                    f.write(
                        f"- [{edge:.6g}, +inf): {cnt} pairs ({frac:.4%})\n"
                    )

        f.write("\n(End of report)\n")
    print(f"已完成： {args.readme}")

if __name__ == "__main__":
    main()
