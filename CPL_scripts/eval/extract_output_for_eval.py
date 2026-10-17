#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jr-llama3:
兼容llama3的输出格式了；
只改了这里：
    # predict = clean_think(predict_raw)
    predict = clean_predict(predict_raw)
    
nojump：
之前是“如果某一行无法唯一定位，将直接跳过这一行；”
现在，不会跳过；
现在，要确保提供的swift_file和ref_file将会行数相等，严格按行对应；
然后，直接按行去对应上doc_id和src_sent_id，而不是用key去查找到doc_id和src_sent_id

sort:
增加了按照 (doc_id, src_sent_id) 升序排列的功能；

new-extract-function:
修改了extract_text函数，现在能匹配：原句子中就有三引号的情况；
另外增加了按顺序排列后输出的功能；

--------------------------------
fix:
如果某一行无法唯一定位，将直接跳过这一行；
------------------------------
脚本功能：clean and get id
输入一个swift_generated_predictions.jsonl
它是一个sample的结果，一行数据的格式是：
{"messages": [{"role": "user", "content": "[English]:end TB completely, Africa will need new and sustained financial commitments from the public and private sectors.\n And, finally, disease-control strategies must be designed to support the economic and health-care priorities of the RECs.\n At the moment, most state and regional health-care systems lack funding and human resources.\n African policymakers must therefore develop systems for prevention, diagnosis, and care that help governments share the disease burden and ensure that treatment protocols are consistent across regions.\n To be sure, there are some positive trends in the fight against TB.\n Global infection rates are falling by about 2% annually, and even African countries hit by TB still managed a 4% decline in infections from 2013 to 2017.\n And, with organizations like the World Health Organization and the African Union beginning to produce roadmaps for TB eradication, it is clear that momentum is steadily building.\n Still, sustaining this progress in Africa will require significant regional coordination, not to mention a lot more money.\n Most vaccines and medications are currently too expensive for the majority of Africans.\n But even those lucky enough to obtain treatment – usually a six- to eight-month course of powerful antibiotics – still face a 20% chance of relapse.\n  Given the provided context, translate the following English sentence to German.\n  [English]:The goal of eradicating TB is within reach; the recent gathering at the UNGA marked a critical turning point in the world’s war on the disease.", "loss": null}, {"role": "assistant", "content": "<think>\n\n</think>\n\nDas Ziel, TB auszurotten, ist in Reichweite; die jüngste Tagung des UNO-Generalrates markierte einen entscheidenden Wendepunkt im Kampf gegen die Krankheit.", "loss": null}], "id": "1e9448c549c2c2650702a07340452904"}
它每n行共享一个user的content（n可以从命令行传入）
它以n行为一个单位，首先，通过清洗user的content（可以用我上面提供的extract_text函数）得到真正的input，比对/data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.sent.json中的数据，判断它的 doc_id,src_sent_id；
其中
/data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.sent.json的格式如下：
{"doc_id": 1, "src_sent_id": 0, "src_sent": "Saudi Arabia’s Game of Thrones", "tgt_sent_id": 0, "tgt_sent": "Game of Thrones in Saudi-Arabien"}

确定每一条数据的doc_id,src_sent_id;

按doc_id,src_sent_id顺序，做成
{doc_id,src_sent_id,prompt,input,predict,label,task}的jsonl文件；prompt是原始user content，input是清洗后的，task是一个字符串，由命令行我来指定 
===============================================================================
功能：
    输入一个 swift_generated_predictions.jsonl（Qwen Swift 的预测结果），
    以及一个带 doc_id/src_sent_id/src_sent/tgt_sent 的参考集
    （例如：/data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.sent.json），
    输出一个 jsonl 文件，每行格式为：

        {
          "doc_id": ...,
          "src_sent_id": ...,
          "prompt": ...,   # 原始 user content
          "input": ...,    # 从 prompt 中清洗出的英文句子
          "predict": ...,  # 模型生成的德语翻译
          "label": ...,    # 参考翻译（tgt_sent）
          "task": ...      # 命令行指定的字符串,ctx/sample,ctx/greedy,sent/sample,sent/greedy
        }

    其中 doc_id/src_sent_id 通过“input 与参考集中 src_sent 匹配”得到。

    说明：
        - 文件 swift_generated_predictions.jsonl 的每一行类似：
          {
            "messages": [
              {"role": "user", "content": "...", "loss": null},
              {"role": "assistant", "content": "<think>...</think> 翻译", "loss": null}
            ],
            "id": "xxxx"
          }
        - 每 n 行共享同一个 user 的 content（n 通过 --group_size 指定），
          本脚本会按行读取、分别处理；group_size 目前只用于 sanity check，
          不影响输出结构。

用法示例：
    python /data/liying/ms-swift/yli/util_pys/pick-workflow/3-fix-new-extract-function-sort-nojump.py \
        --swift_file /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.mix/v9-20251204-014620/checkpoint-937/context/swift_generated_predictions.jsonl \
        --ref_file /data/liying/ms-swift/yli/util_pys/eval/dcomet/dataset_with_id/test.en-de.sft.sent_withid.jsonl \
        --output /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.mix/v9-20251204-014620/checkpoint-937/context/output/1-newnew.jsonl \
        --task "ctx/greedy" \
        --group_size 1

    python /data/liying/ms-swift/yli/util_pys/pick-workflow/3-fix-new-extract-function-sort-nojump.py \
        --swift_file /data/liying/beiqu-swift/yli/pred/sft/train.en-de.256ctx.90.message.mix/v9-20251204-014620/checkpoint-937/context/swift_generated_predictions.jsonl \
        --ref_file /data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.sent.json \
        --output /data/liying/ms-swift/yli/256ctx/90mix-sample-validset-1/clean-and-with_id-output/90mix-pick-sample-ctxpart-merged.clean.id.jsonl \
        --task "ctx/sample" \
        --group_size 8

参数说明：
    --swift_file    必填，swift_generated_predictions.jsonl
    --ref_file      必填，参考集 json（含 doc_id/src_sent_id/src_sent/tgt_sent）
    --output        必填，输出 jsonl 路径
    --task          必填，字符串，会写入每行的 "task"
    --group_size    可选，默认 1：表示每多少行共享一个 prompt，仅用于检查提示是否一致
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Optional
from itertools import zip_longest


# ========== 文本处理工具 ==========
LANG_DICT = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
}

def extract_text(content: str) -> str:
    """
    从 prompt（user content）中提取“真正的输入句子”：
      1) 优先匹配以 [Language]: 开头的最后一段
      2) 否则匹配三引号中的段落
    """
    lines = content.splitlines()

    # 1) 先走 [Language]: 分支（for 循环）
    for lang in LANG_DICT.values():
        tag = f"[{lang}]:"
        lang_lines = []

        for raw in lines:
            s = raw.lstrip()
            if s.startswith(tag):
                lang_lines.append(s[len(tag):].strip())

        if lang_lines:
            return lang_lines[-1]

    # 2) 再尝试三引号块：从第一个三引号到最后一个三引号
    for delimiter in ("'''", '"""'):
        start = content.find(delimiter)
        end = content.rfind(delimiter)
        if start != -1 and end != -1 and end > start:
            inner = content[start + len(delimiter): end]
            return inner.strip()

    # 3) 实在找不到就返回空串
    return ""

def clean_think(text: str) -> str:
    """移除 <think>...</think> 片段，并去除首尾空白"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def clean_predict(text: str) -> str:
    """
    统一清理模型生成结果：
    - 兼容 Qwen 系列的 <think>...</think>
    - 兼容 Llama3 的 chat template 泄漏 token
    - 不依赖模型类型，向后兼容
    """
    if text is None:
        return ""

    # ===== 1) Qwen: 去掉 <think>...</think> =====
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)

    # ===== 2) Llama3: 去掉 chat template 特殊 token =====
    text = re.sub(r"<\|begin_of_text\|>", "", text)
    text = re.sub(r"<\|end_of_text\|>", "", text)
    text = re.sub(r"<\|eot_id\|>", "", text)

    # 去掉所有 role header，例如：
    # <|start_header_id|>assistant<|end_header_id|>
    # <|start_header_id|>user<|end_header_id|>
    text = re.sub(
        r"<\|start_header_id\|>.*?<\|end_header_id\|>\s*",
        "",
        text,
        flags=re.S,
    )

    # ===== 3) 兜底：去掉常见的 Assistant: 前缀 =====
    text = re.sub(r"^\s*(Assistant|assistant)\s*:\s*", "", text)

    return text.strip()


def normalize_text(s: Optional[str]) -> str:
    """
    文本规范化用于对比：
      - 去掉首尾空白
      - 合并连续空白为单个空格
      - 统一直引号/弯引号
    """
    if s is None:
        return ""
    s = s.strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s


# ========== 主函数：按行一一对应 + 严格对齐检查 ==========

def main():
    parser = argparse.ArgumentParser(
        description="从 swift_generated_predictions.jsonl 构造 {doc_id,src_sent_id,prompt,input,predict,label,task} jsonl（swift/ref 按行一一对应且 input==src_sent 校验）。"
    )
    parser.add_argument(
        "--swift_file", required=True,
        help="Qwen Swift 的预测结果 jsonl（swift_generated_predictions.jsonl）"
    )
    parser.add_argument(
        "--ref_file", required=True,
        help="参考集 jsonl，包含 doc_id/src_sent_id/src_sent/tgt_sent（与 swift_file 行数严格相等，按行对应）"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出 jsonl 文件路径"
    )
    parser.add_argument(
        "--task", required=True,
        help="写入每一行的 task 字符串"
    )
    parser.add_argument(
        "--group_size", type=int, default=1,
        help="每多少行共享一个 user 的 content，仅用于 sanity check，默认 1"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total = 0
    results = []

    group_size = max(args.group_size, 1)
    group_prompts = []

    print(f"[info] loading swift_file: {args.swift_file}")
    print(f"[info] loading ref_file  : {args.ref_file}")
    print("[info] 假定二者行数严格相等，并按行一一对应；如果不等或 input/src_sent 不匹配将直接报错退出。")

    with open(args.swift_file, "r", encoding="utf-8") as fin_swift, \
         open(args.ref_file, "r", encoding="utf-8") as fin_ref:

        # 使用 zip_longest 保证行数一致；如果一边先耗尽就报错
        for lineno, (swift_line, ref_line) in enumerate(
            zip_longest(fin_swift, fin_ref),
            start=1
        ):
            # 行数不等，立刻报错
            if swift_line is None or ref_line is None:
                raise ValueError(
                    f"[error] swift_file 和 ref_file 行数不一致，在第 {lineno} 行发现一边已结束。\n"
                    f"       请确保两文件行数严格相等。"
                )

            swift_line = swift_line.strip()
            ref_line = ref_line.strip()

            if not swift_line:
                raise ValueError(f"[error] swift_file 第 {lineno} 行为空，请确保没有空行。")
            if not ref_line:
                raise ValueError(f"[error] ref_file 第 {lineno} 行为空，请确保没有空行。")

            total += 1

            # ========== 解析 swift 当前行 ==========
            data_swift = json.loads(swift_line)
            messages = data_swift.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)

            if user_msg is None or asst_msg is None:
                raise ValueError(f"[error] 第 {lineno} 行找不到 user 或 assistant 消息。")

            prompt = user_msg.get("content", "")
            group_prompts.append(prompt)
            # 每 group_size 行检查一下 prompt 是否完全一致
            if len(group_prompts) == group_size:
                if len(set(group_prompts)) != 1:
                    print(
                        f"[warn] 行 {lineno-group_size+1}~{lineno} 的 user content 不完全一致，"
                        f"请确认 group_size 是否正确。"
                    )
                group_prompts = []

            input_text = extract_text(prompt)
            if not input_text:
                input_text = prompt  # 兜底

            predict_raw = asst_msg.get("content", "")
            # predict = clean_think(predict_raw)
            predict = clean_predict(predict_raw)


            # ========== 解析 ref 当前行 ==========
            data_ref = json.loads(ref_line)
            doc_id = data_ref["doc_id"]
            src_sent_id = data_ref["src_sent_id"]
            src_sent = data_ref["src_sent"]
            tgt_sent = data_ref["tgt_sent"]

            # ========== 校验：input vs src_sent 是否一致（规范化后比较） ==========
            norm_input = normalize_text(input_text)
            norm_src = normalize_text(src_sent)
            if norm_input != norm_src:
                raise ValueError(
                    f"[error] 第 {lineno} 行 input 与 ref_file 的 src_sent 不相等（规范化后不同）。\n"
                    f"       input_text = {input_text}\n"
                    f"       src_sent   = {src_sent}\n"
                    f"       norm_input = {norm_input}\n"
                    f"       norm_src   = {norm_src}"
                )

            # ========== 构造输出对象 ==========
            out_obj = {
                "doc_id": doc_id,
                "src_sent_id": src_sent_id,
                "prompt": prompt,
                "input": input_text,
                "predict": predict,
                "label": tgt_sent,
                "task": args.task,
            }
            results.append(out_obj)

    # 仍然按 (doc_id, src_sent_id) 排序后输出
    results.sort(key=lambda x: (x["doc_id"], x["src_sent_id"]))

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in results:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[done] processed {total} lines (swift_file & ref_file line-to-line aligned, input==src_sent checked)")
    print(f"[success] output written to {args.output}")
    print(f"[info] final entries: {len(results)} (sorted by doc_id, src_sent_id)")


if __name__ == "__main__":
    main()
