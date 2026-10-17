#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1208修复:
用doc_id，src_sent_id对齐，而不是按records顺序或者eumerate的idx；而且每条记录都要尝试构建dcomet，不是只保留最后一条
（doc_id,src_sent_id）

window:
增加一个dcomet可以调控window大小；
-------------------------------------
评估脚本：对已经带 doc_id/src_sent_id 的 jsonl 同时计算 COMET、BLEU、dCOMET

输入 JSONL 每行形如：
    {
      "doc_id": 1,
      "src_sent_id": 0,
      "prompt": "...",
      "input": "Saudi Arabia’s Game of Thrones",
      "predict": "Saudi-Arabiens Game of Thrones",
      "label": "Game of Thrones in Saudi-Arabien",
      "task": "ctx/greedy"
    }

额外输入：
    testset_with_id（例如 valid.en-de.sent.json），包含：
        {"doc_id": 1, "src_sent_id": 0, "src_sent": "...", "tgt_sent": "..."}

功能：
    1）对每一行计算 sentence-level COMET（无上下文）和 BLEU；
    2）利用 testset_with_id 和 doc_id/src_sent_id 构造三句上下文，
       计算上下文版本的 dCOMET（使用 enable_context()）；
    3）将 comet / bleu / dcomet 附加到每行 JSON 后，写到新的 jsonl；
    4）将整体统计结果（平均 BLEU、平均 COMET、平均 dCOMET 等）
       写入一个 summary 文本文件（例如 comet-bleu-dcomet-output.txt）。
{"doc_id": 1, "src_sent_id": 4, "prompt": "  ar?", "input": "C", "predict": "n?", "label": "Können die multinationalen Unternehmen anderen bedürftigen Regionen wie etwa Schwarzafrika, wo das Erbe der Kolonialisierung, Apartheid und Misswirtschaft ein fundamental anderes Geschäftsumfeld hervorgebracht hat, ähnliche Vorteile bringen?", "task": "ctx/greedy"}

用法示例：
CUDA_VISIBLE_DEVICES=0,1,2,3 python ms-swift-CPL\CPL_scripts\eval\comet_bleu_dcomet.py \
  --input_jsonl  /data/liying/beiqu-swift/yli/sample-output/en-de-90-lora32/merged-4ways.flat.no.eval.jsonl \
  --testset_with_id /data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.sent.json \
  --output_jsonl /data/liying/beiqu-swift/yli/sample-output/en-de-90-lora32/merged-4ways.flat.new.eval.jsonl \
  --summary_file /data/liying/beiqu-swift/yli/sample-output/en-de-90-lora32/merged-4ways.flat.new.eval.win5.txt \
  --context_window 5 \
  --gpus 8 \
  --batch_size_comet 256 \
  --batch_size_dcomet 128
# window = 5

  
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any
from sacrebleu.metrics import BLEU
import torch
from comet.models.utils import Prediction

# 允许反序列化 COMET 的 Prediction 对象（解决 PyTorch 2.6 weights_only 的限制）
torch.serialization.add_safe_globals([Prediction])
import numpy as np
from sacrebleu.metrics import BLEU
from comet import load_from_checkpoint
from tqdm import tqdm

# ==================== 配置 ====================

COMET_CKPT = Path("/data/liying/models/Unbabel/wmt22-comet-da/checkpoints/model.ckpt")
BLEU_TOKENIZE = "flores200"
BLEU_SMOOTH = "floor"
DO_BOOTSTRAP = True
N_BOOT = 1000
SEED = 2024

np.random.seed(SEED)

# =============================================


# ---------- 一些小工具 ----------

def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        means.append(values[idx].mean())
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def is_chinese_char(c: str) -> bool:
    return '\u4e00' <= c <= '\u9fff'


def should_use_chinese_tokenizer(lines: List[dict]) -> bool:
    sample = lines[:3]
    total_chars = 0
    zh_chars = 0
    for item in sample:
        label = item.get("label", "")
        total_chars += len(label)
        zh_chars += sum(1 for ch in label if is_chinese_char(ch))
    return total_chars > 0 and (zh_chars / total_chars >= 0.5)


# ---------- 读取 testset_with_id，构造上下文索引 ----------

def load_testset_with_id(path: Path) -> Tuple[Dict[int, List[dict]], Dict[int, Dict[int, int]]]:
    """
    返回：
        doc_sents: doc_id -> [ {src_sent_id, src_sent, tgt_sent}, ... ]（按 src_sent_id 升序）
        doc_pos  : doc_id -> {src_sent_id -> index_in_doc_sents}
    """
    doc_sents: Dict[int, List[dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = data["doc_id"]
            src_sent_id = data["src_sent_id"]
            src_sent = data["src_sent"]
            tgt_sent = data["tgt_sent"]
            doc_sents.setdefault(doc_id, []).append(
                {"src_sent_id": src_sent_id, "src_sent": src_sent, "tgt_sent": tgt_sent}
            )
    #doc_sent从1开始，src_sent_id从0开始
    # doc_sents[1]==[{"src_sent_id": 0, "src_sent": xxx, "tgt_sent": ...},{"src_sent_id": 1, "src_sent": xxx, "tgt_sent": ...}..]

    # 排序并构建索引
    doc_pos: Dict[int, Dict[int, int]] = {}
    for doc_id, lst in doc_sents.items():
        lst.sort(key=lambda x: x["src_sent_id"])
        pos_map = {item["src_sent_id"]: idx for idx, item in enumerate(lst)}
        doc_pos[doc_id] = pos_map
    # doc_pos[1] == {
    #     0: 0,
    #     1: 1,
    #     2: 2,
    #     ...
    # }

    print(f"[INFO] loaded testset_with_id from {path}, docs: {len(doc_sents)}")
    return doc_sents, doc_pos


# ---------- 计算 COMET + BLEU ----------

def compute_comet_and_bleu(
    records: List[dict],
    comet_model,
    gpus: int,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray, BLEU]:
    """
    对每条记录计算：
        - sentence-level BLEU
        - sentence-level COMET（无上下文）

    返回：
        comet_scores: shape [N]
        bleu_scores : shape [N]
        bleu_metric : sacrebleu BLEU 对象（用于 corpus BLEU）
    """
    # 决定 BLEU 分词
    use_zh = should_use_chinese_tokenizer(records)
    bleu_tokenizer = "zh" if use_zh else BLEU_TOKENIZE
    print(f"[INFO] BLEU tokenize: {bleu_tokenizer}")
    bleu_metric = BLEU(tokenize=bleu_tokenizer, smooth_method=BLEU_SMOOTH, effective_order=True)

    # COMET 输入
    comet_inputs = [
        {"src": it["input"], "mt": it["predict"], "ref": it["label"]} for it in records
    ]

    # preds = comet_model.predict(
    #     comet_inputs,
    #     batch_size=batch_size,
    #     gpus=gpus if gpus >= 0 else None,
    #     progress_bar=True,
    #     num_workers=0
    # )
    preds = comet_model.predict(
        comet_inputs,
        batch_size=batch_size,
        gpus=1,
        progress_bar=True,
        num_workers=0
    )
    comet_scores = np.array(preds.scores, dtype=float)
    assert len(comet_scores) == len(records)

    # 逐句 BLEU
    bleu_scores = []
    for item in tqdm(records, desc="Computing sentence BLEU"):
        bleu = bleu_metric.sentence_score(item["predict"], [item["label"]]).score
        bleu_scores.append(bleu)

    bleu_scores = np.array(bleu_scores, dtype=float)
    return comet_scores, bleu_scores, bleu_metric


# ---------- 构造 dCOMET 上下文，计算 dCOMET ----------

def build_dcomet_items(
    records: List[dict],
    doc_sents: Dict[int, List[dict]],
    doc_pos: Dict[int, Dict[int, int]],
    context_window: int,
) -> Tuple[List[dict], List[int]]:
    """
    根据 testset_with_id 中的句子，构造 dCOMET 需要的上下文：
    
    每条记录都会尝试生成 dCOMET，即使 doc_id/src_sent_id 相同，也会处理每一行。

    对某个句子在 doc 中的位置为 j：
      - 若 j < context_window，则视为上下文不足，跳过；
      - 否则：
            prev_src = [src_sent[j-context_window], ..., src_sent[j-1]]
            prev_ref = [tgt_sent[j-context_window], ..., tgt_sent[j-1]]
            curr_src = src_sent[j]
            curr_ref = tgt_sent[j]
            curr_hyp = records[idx]["predict"]

        context_src = "\\n".join(prev_src) + " </s> " + curr_src
        context_hyp = "\\n".join(prev_ref) + " </s> " + curr_hyp
        context_ref = "\\n".join(prev_ref) + " </s> " + curr_ref

    返回：
        dcomet_items: [{src, mt, ref}, ...]
        idx_for_scores: 每条 dcomet_items 对应的 records 中的索引
    """
    if context_window <= 0:
        raise ValueError(f"context_window 必须为正整数，当前为 {context_window}")

    dcomet_items: List[dict] = []
    idx_for_scores: List[int] = []

    # 统计原因
    stats = {
        "missing_doc_id": 0,
        "missing_in_testset": 0,
        "insufficient_context": 0,
        "empty_fields": 0,
        "kept": 0,
    }

    for idx, rec in enumerate(records):
        doc_id = rec.get("doc_id")
        src_sent_id = rec.get("src_sent_id")

        if doc_id is None or src_sent_id is None:
            stats["missing_doc_id"] += 1
            continue

        if doc_id not in doc_sents or doc_id not in doc_pos or src_sent_id not in doc_pos[doc_id]:
            stats["missing_in_testset"] += 1
            continue

        j = doc_pos[doc_id][src_sent_id]
        if j < context_window:
            stats["insufficient_context"] += 1
            continue

        doc_list = doc_sents[doc_id]
        prev_slice = doc_list[j - context_window : j]
        curr = doc_list[j]

        prev_src = [it["src_sent"] for it in prev_slice]
        prev_ref = [it["tgt_sent"] for it in prev_slice]
        s_curr = curr["src_sent"]
        r_curr = curr["tgt_sent"]
        h_curr = rec.get("predict", None)

        if any(x in ("", None) for x in prev_src + prev_ref + [s_curr, r_curr, h_curr]):
            stats["empty_fields"] += 1
            continue

        context_src = "\n".join(prev_src) + f" </s> {s_curr}"
        context_hyp = "\n".join(prev_ref) + f" </s> {h_curr}"
        context_ref = "\n".join(prev_ref) + f" </s> {r_curr}"

        dcomet_items.append({
            "src": context_src,
            "mt": context_hyp,
            "ref": context_ref,
        })
        idx_for_scores.append(idx)
        stats["kept"] += 1

    print(f"[INFO] dCOMET 构建统计:")
    print(f"  total records         : {len(records)}")
    print(f"  missing doc_id/src_sent_id : {stats['missing_doc_id']}")
    print(f"  missing in testset         : {stats['missing_in_testset']}")
    print(f"  insufficient context       : {stats['insufficient_context']}")
    print(f"  empty fields               : {stats['empty_fields']}")
    print(f"  kept for dCOMET            : {stats['kept']}")
    print(f"[INFO] built {len(dcomet_items)} dCOMET items (context window={context_window})")

    return dcomet_items, idx_for_scores

def compute_dcomet(
    records: List[dict],
    doc_sents: Dict[int, List[dict]],
    doc_pos: Dict[int, Dict[int, int]],
    dcomet_model,
    gpus: int,
    batch_size: int,
    context_window: int,
) -> np.ndarray:
    """
    计算 dCOMET 分数，并把每条记录的 dcomet 写到 records[idx]["dcomet"] 中。

    返回：
        dcomet_scores: shape [N]，某些句子可能没有上下文，dcomet 为 np.nan。
    """
    dcomet_items, idx_for_scores = build_dcomet_items(
        records, doc_sents, doc_pos, context_window=context_window
    )
    if not dcomet_items:
        print("[WARN] 没有可用于 dCOMET 的样本（上下文不足），所有 dcomet=NaN")
        return np.full(len(records), np.nan, dtype=float)

    # 建议这里还是强制单卡，避免多次 DDP 卡死
    # if gpus is None or gpus <= 0:
    #     dcomet_gpus = 0
    # else:
    #     dcomet_gpus = 1
    dcomet_gpus=gpus
    dcomet_model.enable_context()
    print(f"[INFO] computing dCOMET on {len(dcomet_items)} items with window={context_window} ...")
    preds = dcomet_model.predict(
        dcomet_items,
        batch_size=batch_size,
        gpus=dcomet_gpus if dcomet_gpus > 0 else None,
        progress_bar=True,
        num_workers=0,
    )

    seg_scores = np.array(preds.scores, dtype=float)
    assert len(seg_scores) == len(idx_for_scores)

    dcomet_scores = np.full(len(records), np.nan, dtype=float)
    for idx, score in zip(idx_for_scores, seg_scores):
        dcomet_scores[idx] = score

    return dcomet_scores

# ---------- 主逻辑 ----------

def main():
    parser = argparse.ArgumentParser(
        description="对带 doc_id/src_sent_id 的 jsonl 计算 COMET / BLEU / dCOMET。"
    )
    parser.add_argument("--input_jsonl", type=str, required=True, help="输入 jsonl（含 doc_id/src_sent_id/input/predict/label）")
    parser.add_argument("--testset_with_id", type=str, required=True, help="参考集 jsonl（含 doc_id/src_sent_id/src_sent/tgt_sent）")
    parser.add_argument("--output_jsonl", type=str, required=True, help="输出打分后的 jsonl 路径")
    parser.add_argument("--summary_file", type=str, required=True, help="整体统计结果输出路径，如 comet-bleu-dcomet-output.txt")
    # 新增：可控 dCOMET window（单位：句子数）
    parser.add_argument(
        "--context_window", type=int, default=2,
        help="dCOMET 上下文窗口大小：使用前 N 句作为上下文（默认 2，即使用 i-2,i-1,i 三句）"
    )
    parser.add_argument("--gpus", type=int, default=1, help="用于 COMET / dCOMET 的 GPU 数（或 0/负数表示 CPU）")
    parser.add_argument("--batch_size_comet", type=int, default=1024, help="COMET batch size")
    parser.add_argument("--batch_size_dcomet", type=int, default=256, help="dCOMET batch size")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    testset_path = Path(args.testset_with_id)
    output_jsonl = Path(args.output_jsonl)
    summary_file = Path(args.summary_file)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    # 环境变量（可选：离线模式）
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 1. 读取输入 jsonl
    records: List[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    print(f"[INFO] loaded {len(records)} records from {input_path}")

    # 2. 读取 testset_with_id（用于 dCOMET 上下文）
    doc_sents, doc_pos = load_testset_with_id(testset_path)

    # 3. 加载 COMET 模型（一个无上下文，一个用于 dCOMET）
    print(f"[INFO] loading COMET model from {COMET_CKPT}")
    comet_model = load_from_checkpoint(str(COMET_CKPT), reload_hparams=True)
    print(f"[INFO] COMET models loaded. Encoder: {comet_model.hparams.encoder_model}")

    print(f"[INFO] loading dCOMET model from {COMET_CKPT}")
    dcomet_model = load_from_checkpoint(str(COMET_CKPT), reload_hparams=True)

    print(f"[INFO] begin bleu and COMET compute: ")
    # 4. COMET + BLEU
    comet_scores, bleu_scores, bleu_metric = compute_comet_and_bleu(
        records, comet_model, gpus=args.gpus, batch_size=args.batch_size_comet
    )
    print(f"[INFO] Success! bleu and COMET compute end")

    print(f"[INFO] begin dCOMET compute :")
    # 5. dCOMET
    dcomet_scores = compute_dcomet(
        records,
        doc_sents,
        doc_pos,
        dcomet_model,
        gpus=args.gpus,
        batch_size=args.batch_size_dcomet,
        context_window=args.context_window,
    )
    print(f"[INFO] Success! dCOMET compute end")

    # 6. 把分数写回记录，并输出新的 jsonl
    for rec, c, b, dc in zip(records, comet_scores, bleu_scores, dcomet_scores):
        rec["comet"] = float(c)
        rec["bleu"] = float(b)
        # 用 NaN 表示没有上下文的句子；json 里可以写成 null，或者保留 NaN 字符串
        # 这里选择 null：方便后续处理
        rec["dcomet"] = None if np.isnan(dc) else float(dc)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] scored jsonl written to {output_jsonl}")

    # 7. 统计整体指标并写 summary
    hyps = [r["predict"] for r in records]
    refs = [r["label"] for r in records]

    avg_sent_bleu = float(np.mean(bleu_scores))
    corpus_bleu = bleu_metric.corpus_score(hyps, [refs]).score

    avg_comet = float(np.mean(comet_scores))
    std_comet = float(np.std(comet_scores, ddof=1))

    # dCOMET 只统计有上下文的那些句子
    valid_d = dcomet_scores[~np.isnan(dcomet_scores)]
    if len(valid_d) > 0:
        avg_dcomet = float(np.mean(valid_d))
        std_dcomet = float(np.std(valid_d, ddof=1))
    else:
        avg_dcomet = float("nan")
        std_dcomet = float("nan")

    lines_out = [
        f"[{input_path.name}] 统计结果：",
        f"  avg_sentence_bleu : {avg_sent_bleu:.2f}",
        f"  corpus_bleu       : {corpus_bleu:.2f}",
        f"  avg_comet         : {avg_comet:.4f}",
        f"  std_comet         : {std_comet:.4f}",
        f"  avg_dcomet        : {avg_dcomet:.4f}  (仅有上下文的句子)",
        f"  std_dcomet        : {std_dcomet:.4f}",
        f"  写在语雀：bleu/comet/dcomet : "
        f"{corpus_bleu:.2f}/{(avg_comet * 100):.2f}/{(avg_dcomet * 100 if np.isfinite(avg_dcomet) else float('nan')):.2f}",
    ]

    if DO_BOOTSTRAP and len(comet_scores) > 1:
        ci_low_c, ci_high_c = bootstrap_mean_ci(comet_scores, n_boot=N_BOOT, alpha=0.05)
        lines_out.append(f"  COMET 95% CI (bootstrap):   [{ci_low_c:.4f}, {ci_high_c:.4f}]")

    if DO_BOOTSTRAP and len(valid_d) > 1:
        ci_low_d, ci_high_d = bootstrap_mean_ci(valid_d, n_boot=N_BOOT, alpha=0.05)
        lines_out.append(f"  dCOMET 95% CI (bootstrap):  [{ci_low_d:.4f}, {ci_high_d:.4f}]")

    result_text = "\n".join(lines_out)
    print(result_text)

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(result_text + "\n")
    print(f"[✓] 汇总结果写入: {summary_file}")


if __name__ == "__main__":
    main()
