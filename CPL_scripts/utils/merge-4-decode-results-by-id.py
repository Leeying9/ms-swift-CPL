#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将ms-swift/yli/util_pys/3-fix-new-extract-function-sort.py生成的4个结果，
    每行格式类似：
        {
          "doc_id": ...,
          "src_sent_id": ...,
          "prompt": ...,
          "input": ...,
          "predict": ...,
          "label": ...,
          "task": "ctx/sample" | "ctx/greedy" | "sent/sample" | "sent/greedy"
        }

按照 (doc_id, src_sent_id)和task将行挨在一起排好；保持升序；

如果碰到某一个(doc_id, src_sent_id)只在其中一个文件有，那么丢弃这个条目，并打印信息；输出一个最终的jsonl；
=======================================================================  
用法示例：
/data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/4096/sentpart/sent/output/eval-output/eval.jsonl
/data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/4096/ctxpart/context/output/eval-output2/eval.jsonl
/data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/greedy-infer-validset/sent/output/eval-output/eval.jsonl
/data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/greedy-infer-validset/context/output/eval-output2/eval.jsonl

    python /data/liying/ms-swift/yli/util_pys/4-merge-4-decode-results-by-id.py \
        --greedy_sent /data/liying/ms-swift/yli/256ctx/90mix-sample-validset-2-greedy/90mix-pick-greedy-sentpart.clean.id.jsonl \
        --greedy_ctx  /data/liying/ms-swift/yli/256ctx/90mix-sample-validset-1-greedy/90mix-pick-greedy-ctxpart.clean.id.jsonl \
        --sample_sent /data/liying/ms-swift/yli/256ctx/90mix-sample-validset-2/clean-and-with_id-output/90mix-pick-sample-sentpart-merged.clean.id.jsonl \
        --sample_ctx  /data/liying/ms-swift/yli/256ctx/90mix-sample-validset-1/clean-and-with_id-output/90mix-pick-sample-ctxpart-merged.clean.id.jsonl \
        --output      /data/liying/ms-swift/yli/256ctx/90mix-sample-validset-merged/merged-4ways.flat.jsonl

    python /data/liying/ms-swift/yli/util_pys/pick-workflow/4-merge-4-decode-results-by-id.py \
        --greedy_sent /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/greedy-infer-validset/sent/output/eval-output/eval.jsonl \
        --greedy_ctx  /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/greedy-infer-validset/context/output/eval-output2/eval.jsonl \
        --sample_sent /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/4096/sentpart/sent/output/eval-output/eval.jsonl \
        --sample_ctx  /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/v0-20251205-205429/checkpoint-157/4096/ctxpart/context/output/eval-output2/eval.jsonl \
        --output      /data/liying/beiqu-swift/yli/sample-output/train.en-de.256ctx.90.message.head2w.mix/merged-4ways.flat.jsonl

# en-de 90 32rank sft:
    python /data/liying/ms-swift/yli/util_pys/pick-workflow/4-merge-4-decode-results-by-id.py \
        --greedy_sent /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.mix/v9-20251204-014620/checkpoint-937/greedy-infer-train10set/sent/output/eval-output/eval.jsonl \
        --greedy_ctx  /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.mix/v9-20251204-014620/checkpoint-937/greedy-infer-train10set/context/output/eval-output2/eval.jsonl \
        --sample_sent /data/liying/beiqu-swift/yli/sample-output/en-de-90-lora32/sentpart/sent/output/eval-output/eval.jsonl \
        --sample_ctx  /data/liying/beiqu-swift/yli/sample-output/en-de-90-lora32/ctxpart/context/output/eval-output2/eval.jsonl \
        --output      /data/liying/beiqu-swift/yli/sample-output/en-de-90-lora32/merged-4ways.flat.jsonl


"""

import argparse
import json
import os
from typing import Dict, Tuple, List

Key = Tuple[int, int]  # (doc_id, src_sent_id)


def load_jsonl_as_dict_of_lines(path: str, tag: str) -> Dict[Key, List[str]]:
    """
    把一个 jsonl 读成 dict：
        (doc_id, src_sent_id) -> [原始 json 行字符串, ...]

    注意：
      - 使用 json.loads 解析来读取 doc_id / src_sent_id；
      - 存储时保留原始行文本，写出时原样输出，不改变字段顺序等；
      - 如果同一个文件里出现重复的 (doc_id, src_sent_id)，
        会全部保留在列表中，不再覆盖、不再报警。
    """
    mapping: Dict[Key, List[str]] = {}
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"{tag} 第 {line_no} 行不是合法 JSON：{e}") from e

            if "doc_id" not in obj or "src_sent_id" not in obj:
                raise ValueError(f"{tag} 第 {line_no} 行缺少 doc_id 或 src_sent_id 字段。")

            key = (obj["doc_id"], obj["src_sent_id"])
            mapping.setdefault(key, []).append(line)  # 保留所有重复行

    print(f"[info] loaded {total} lines from {tag}, unique keys: {len(mapping)}")
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="按 (doc_id, src_sent_id) 对齐 4 个 jsonl，只保留 4 个文件都有的条目，并按 task 排序输出。"
    )
    parser.add_argument("--greedy_sent", required=True, help="greedy sentence-level jsonl")
    parser.add_argument("--greedy_ctx", required=True, help="greedy context-level jsonl")
    parser.add_argument("--sample_sent", required=True, help="sample sentence-level jsonl")
    parser.add_argument("--sample_ctx", required=True, help="sample context-level jsonl")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. 加载四个文件，映射到 (doc_id, src_sent_id) -> [原始行,...]
    g_sent = load_jsonl_as_dict_of_lines(args.greedy_sent, "greedy_sent")
    g_ctx = load_jsonl_as_dict_of_lines(args.greedy_ctx, "greedy_ctx")
    s_sent = load_jsonl_as_dict_of_lines(args.sample_sent, "sample_sent")
    s_ctx = load_jsonl_as_dict_of_lines(args.sample_ctx, "sample_ctx")

    # 2. 所有 key 的并集
    all_keys = set(g_sent.keys()) | set(g_ctx.keys()) | set(s_sent.keys()) | set(s_ctx.keys())
    print(f"[info] total unique keys across 4 files: {len(all_keys)}")

    kept = 0
    dropped = 0

    # 3. 按 (doc_id, src_sent_id) 排序
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], k[1]))

    with open(args.output, "w", encoding="utf-8") as fout:
        for doc_id, src_sent_id in sorted_keys:
            key = (doc_id, src_sent_id)
            missing_in = []

            if key not in g_sent:
                missing_in.append("greedy_sent")
            if key not in g_ctx:
                missing_in.append("greedy_ctx")
            if key not in s_sent:
                missing_in.append("sample_sent")
            if key not in s_ctx:
                missing_in.append("sample_ctx")

            # 只保留 4 个文件都有的 key
            if missing_in:
                dropped += 1
                print(
                    f"[warn] (doc_id={doc_id}, src_sent_id={src_sent_id}) "
                    f"在以下文件中缺失：{', '.join(missing_in)}，丢弃该 (doc_id, src_sent_id)。"
                )
                continue

            kept += 1

            # 收集该 key 的所有行，并按 task 排序
            lines_with_task: List[Tuple[str, str]] = []  # (task, line)

            for line in g_sent[key]:
                obj = json.loads(line)
                task = obj.get("task", "")
                lines_with_task.append((task, line))

            for line in g_ctx[key]:
                obj = json.loads(line)
                task = obj.get("task", "")
                lines_with_task.append((task, line))

            for line in s_sent[key]:
                obj = json.loads(line)
                task = obj.get("task", "")
                lines_with_task.append((task, line))

            for line in s_ctx[key]:
                obj = json.loads(line)
                task = obj.get("task", "")
                lines_with_task.append((task, line))

            # 按 task 字符串升序排序
            lines_with_task.sort(key=lambda x: x[0])

            # 按排序后的顺序写出原始行
            for _, line in lines_with_task:
                fout.write(line + "\n")

    print(f"[done] kept {kept} keys (each key 4个文件都有, 行按 task 排序), dropped {dropped} keys。")
    print(f"[success] merged jsonl written to {args.output}")


if __name__ == "__main__":
    main()