# /data/liying/ms-swift/yli/util_pys/pick-workflow/7-1-pick-cross-new-duplicate.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
duplicate：
对结果去重,并且恢复了meta字段，调整了obj的字段的顺序跟intra一样；

k代表什么：
对于1个源端x，找到它的18个回答中最好的一个y1，y1是ctx的话，就把sent的9个构成group；group中选最差的k个y和最好的k个y，
分别和y1，构成2k个pair；然后去filter留下合格的；

new:
取 win任务的最top ，分别匹配 lose任务的top_k，bottom_k，构建k*2个pair，然后进行filter;

1.comet
  python /data/liying/ms-swift/yli/util_pys/pick-workflow/7-1-pick-cross-new.py \
  --input /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/merged-4ways.flat.jsonl \
  --output /data/liying/beiqu-swift/yli/dataset/256cross-2wsft-v1.0/cross.comet.jsonl \
  --readme /data/liying/beiqu-swift/yli/dataset/256cross-2wsft-v1.0/cross.comet.jsonl.readme.txt \
  --mission cross \
  --rule comet \
  --min-len 6 \
  --max-len 200 \
  --min-win-score 0.30 \
  --min-gap 0.002 \
  --max-gap 0.1 \
  --max-pairs-per-group 4 \
  --gap-bins "0,0.002,0.005,0.006,0.008,0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090"
2.dcomet
python /data/liying/ms-swift/yli/util_pys/pick-workflow/7-1-pick-cross-new.py \
  --input /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/merged-4ways.flat.jsonl \
  --output /data/liying/beiqu-swift/yli/dataset/256cross-2wsft-v1.0/cross.dcomet.jsonl \
  --readme /data/liying/beiqu-swift/yli/dataset/256cross-2wsft-v1.0/cross.dcomet.jsonl.readme.txt \
  --mission cross \
  --rule dcomet \
  --min-len 6 \
  --max-len 200 \
  --min-win-score 0.30 \
  --min-gap 0.002 \
  --max-gap 0.1 \
  --max-pairs-per-group 4 \
  --gap-bins "0,0.002,0.005,0.006,0.008,0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090"
3.weighted
python /data/liying/ms-swift/yli/util_pys/pick-workflow/7-1-pick-cross-new.py \
  --input /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/merged-4ways.flat.jsonl \
  --output /data/liying/beiqu-swift/yli/dataset/256cross-2wsft-v1.0/cross.dcomet-and-comet.jsonl \
  --readme /data/liying/beiqu-swift/yli/dataset/256cross-2wsft-v1.0/cross.dcomet-and-comet.jsonl.readme.txt \
  --mission cross \
  --rule weighted \
  --w-comet 0.5 \
  --w-bleu  0.0 \
  --w-dcomet 0.5 \
  --min-len 6 \
  --max-len 200 \
  --min-win-score 0.30 \
  --min-gap 0.002 \
  --max-gap 0.1 \
  --max-pairs-per-group 4 \
  --gap-bins "0,0.002,0.005,0.006,0.008,0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090"
从混合 sent/ctx 翻译结果中筛选 **cross-task** 偏好学习 pair 的脚本。
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
        description="Select cross-task preference pairs (sent vs ctx) from translation candidates."
    )

    # I/O
    parser.add_argument("--input", type=str, required=True,
                        help="Input jsonl file path.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output jsonl file path for preference pairs.")
    parser.add_argument("--readme", type=str, required=True,
                        help="Output readme txt/markdown file path.")
    parser.add_argument("--mission", type=str, default="cross",
                        help="Mission field in output json (default: cross).")

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
                             "By design: top-1 vs bottom-k among cross-task candidates.")

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
            # obj = json.loads(line)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                print("Problem line:", line)
                raise
            dcomet = safe_float(obj.get("dcomet"))
            if dcomet is not None:
                total += dcomet
                count += 1
    if count == 0:
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
    """Compute advantage score for one item according to rule."""
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
        return dcomet
    elif rule == "weighted":
        comet_val = comet if comet is not None else 0.0
        bleu_val = bleu if bleu is not None else 0.0
        score = w_comet * comet_val + w_bleu * bleu_val + w_dcomet * dcomet
        return score
    else:
        return None


def tokenize_len(text: str) -> int:
    """简单的 token 统计：按空格分割。"""
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
    """对一个 (winner, loser) 应用过滤规则。"""
    win_text = (winner.get("predict") or "").strip()
    lose_text = (loser.get("predict") or "").strip()
    win_score = float(winner["__score__"])
    lose_score = float(loser["__score__"])

    if win_text == lose_text:
        return False, "identical_predict", win_score - lose_score

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

    gap = win_score - lose_score
    if gap < 0:
        return False, "negative_gap_unexpected", gap

    if min_gap is not None and gap < min_gap:
        return False, "gap_too_small", gap

    if max_gap is not None and gap > max_gap:
        return False, "gap_too_large", gap

    if min_win_score is not None and win_score < min_win_score:
        return False, "win_score_too_low", gap

    if max_lose_score is not None and lose_score > max_lose_score:
        return False, "lose_score_too_high", gap

    return True, None, gap


def parse_gap_bins(gap_bins_str: str) -> List[float]:
    edges = [float(x) for x in gap_bins_str.split(",") if x.strip() != ""]
    edges = sorted(edges)
    if len(edges) == 0:
        edges = [0.0, 1.0]
    #edges=[0,0.01,0.02,0.05,0.1,0.2,1.0]
    return edges


def bin_gap(gap: float, bin_edges: List[float]) -> int:
    """返回 gap 落到的 bin index."""
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

    # 1) dcomet 全局平均值
    dcomet_mean = compute_global_dcomet_mean(args.input)

    # 2) 按 (doc_id, src_sent_id) 分组
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at line {line_idx}: {e}")

            doc_id = obj.get("doc_id")
            src_sent_id = obj.get("src_sent_id")
            key = (doc_id, src_sent_id)
            groups[key].append(obj)

    # 3) 遍历 group，构造 cross pair
    gap_bin_edges = parse_gap_bins(args.gap_bins)
    gap_hist_counts = [0 for _ in gap_bin_edges]

    reject_stats: Counter = Counter()
    total_pair_attempts = 0
    total_pairs_kept = 0
    total_groups = len(groups)
    groups_with_any_pair = 0
    kept_task_counter = Counter()
    kept_cross_combo_counter = Counter()

    with open(args.output, "w", encoding="utf-8") as fout:
        seen_objs=[]
        quchong=0
        for key, cand_list in groups.items():
            doc_id, src_sent_id = key

            scored_cands: List[Dict[str, Any]] = []
            task_prefix_set = set()

            for item in cand_list:
                task = str(item.get("task", ""))
                task_prefix = task.split("/")[0] if "/" in task else task

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

                new_item = dict(item)
                new_item["__score__"] = float(score)
                new_item["__task_prefix__"] = task_prefix
                scored_cands.append(new_item)
                task_prefix_set.add(task_prefix)

            if len(scored_cands) < 2:
                reject_stats["no_valid_candidate_in_group"] += 1
                continue

            if len(task_prefix_set) < 2:
                reject_stats["no_cross_candidate_in_group"] += 1
                continue

            scored_cands.sort(key=lambda x: x["__score__"], reverse=True)

            winner = scored_cands[0]
            winner_prefix = winner["__task_prefix__"]

            losers_pool = [
                c for c in scored_cands[1:]
                if c["__score__"] < winner["__score__"]
                and c["__task_prefix__"] != winner_prefix
            ]

            if len(losers_pool) == 0:
                reject_stats["no_cross_loser_for_winner"] += 1
                continue

            k = max(1, args.max_pairs_per_group)
            # 取 win任务的最top ，分别匹配 lose任务的top_k，bottom_k个
            losers_pool.sort(key=lambda x: x["__score__"])  # 升序
            chosen_losers = losers_pool[:k]+losers_pool[-k:]

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

                loser_prefix = loser["__task_prefix__"]

                # 输出字段顺序与 intra 保持一致，并加上 meta
                out_obj = {
                    "x_win": winner.get("prompt"),
                    "x_lose": loser.get("prompt"),
                    "y_win": winner.get("predict"),
                    "y_lose": loser.get("predict"),
                    "mission": args.mission,
                    "win_value": winner["__score__"],
                    "lose_value": loser["__score__"],
                    "rule": args.rule,
                    "meta": {
                        "doc_id": doc_id,
                        "src_sent_id": src_sent_id,
                        "task_prefix": winner_prefix,          # 和 intra 一样，用 winner 这边的前缀
                        "task_win": winner.get("task"),
                        "task_lose": loser.get("task"),
                    },
                }

                if out_obj not in seen_objs:
                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    seen_objs.append(out_obj)

                    gap_idx = bin_gap(gap, gap_bin_edges)
                    gap_hist_counts[gap_idx] += 1

                    total_pairs_kept += 1
                    group_kept_pairs += 1

                    task_win = winner.get("task", "")
                    kept_task_counter[task_win] += 1

                    combo_key = f"{winner_prefix}->{loser_prefix}"
                    kept_cross_combo_counter[combo_key] += 1
                else:
                    quchong += 1

            if group_kept_pairs == 0:
                reject_stats["no_valid_pair_in_group"] += 1
            else:
                groups_with_any_pair += 1

    # 4) 写 readme
    with open(args.readme, "w", encoding="utf-8") as f:
        f.write("# Cross-task Preference Pair Selection Report\n\n")

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
        f.write(f"- Groups with at least one kept cross pair: {groups_with_any_pair}\n")
        f.write(f"- Total cross pair attempts (before filters): {total_pair_attempts}\n")
        f.write(f"- Total kept cross pairs: {total_pairs_kept}\n")
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

        f.write("\n## Task Distribution of Kept Pairs (only winner)\n\n")
        total_tasks_counted = sum(kept_task_counter.values())
        if total_tasks_counted == 0:
            f.write("No kept pairs, so no task distribution.\n\n")
        else:
            for task, cnt in kept_task_counter.most_common():
                frac = cnt / total_tasks_counted
                f.write(f"- {task}: {cnt} ({frac:.4%})\n")
        f.write("\n")

        f.write("## Cross Direction Distribution (task_prefix)\n\n")
        total_combos = sum(kept_cross_combo_counter.values())
        if total_combos == 0:
            f.write("No kept pairs, so no cross direction distribution.\n\n")
        else:
            for combo, cnt in kept_cross_combo_counter.most_common():
                frac = cnt / total_combos
                f.write(f"- {combo}: {cnt} ({frac:.4%})\n")
        f.write("\n")

        f.write("## Gap Histogram (win_value - lose_value)\n\n")
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
