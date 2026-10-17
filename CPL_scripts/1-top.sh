#!/bin/bash
# /data/liying/exp_queue/universal-scripts/sft/[top示例]39-qwen3-4b-all-langs_sft-and-mycpoA.sh
model_path="/data/liying/models/Qwen3-4B/qwen3-4b"
model_type=qwen3
extract_py_path="/data/liying/ms-swift/yli/util_pys/pick-workflow/3-fix-new-extract-function-sort-nojump.py"
LOG_BASE="/data/liying/exp_queue/universal-scripts/othermodel-sft-mycpo/logs/$(basename $model_path)"
# gpus_arg="4,5,6,7"
gpus_arg="0,1,2,3"
#en-nl en-ru
for lang_pair in en-de en-es en-fr en-it; do
  log_lang_dir="$LOG_BASE/$lang_pair"
  mkdir -p "$log_lang_dir"
  #==================开始sft=========================
  #------------------测试集---------------------------
  if [[ $lang_pair == "en-de" ]];then 
      sent_testset=/data/liying/beiqu-swift/yli/dataset/en-de-testset/test.en-de.sft.sent.json
      context_testset=/data/liying/beiqu-swift/yli/dataset/en-de-testset-256ctx/test.en-de.256ctx.msg.jsonl
      testset_with_id=/data/liying/ms-swift/yli/util_pys/eval/dcomet/dataset_with_id/test.en-de.sft.sent_withid.jsonl
      dataset_sent=/data/liying/beiqu-swift/yli/dataset/256-sft-right/sent-final-train/train.en-de.sent.90.message.head2w.jsonl
      dataset_ctx=/data/liying/beiqu-swift/yli/dataset/256-sft-right/ctx-final-train/train.en-de.256ctx.90.message.head2w.jsonl
      dataset_mix=/data/liying/beiqu-swift/yli/dataset/256-sft-right/mix/train.en-de.256ctx.90.message.head2w.mix.jsonl
  else
      BASE_DIR="/data/liying/beiqu-swift/yli/dataset/sft"
      sent_testset="$BASE_DIR/$lang_pair/test.$lang_pair.withid.msg.jsonl"
      context_testset="$BASE_DIR/$lang_pair/test.$lang_pair.256ctx.msg.jsonl"
      testset_with_id="$BASE_DIR/$lang_pair/withid/test.$lang_pair.withid.jsonl"
      dataset_sent="$BASE_DIR/$lang_pair/train.$lang_pair.2w.sent.msg.jsonl"
      dataset_ctx="$BASE_DIR/$lang_pair/train.$lang_pair.2w.256ctx.msg.jsonl"
      dataset_mix="$BASE_DIR/$lang_pair/train.$lang_pair.2w.mix.msg.jsonl"
  fi
  #------------------测试集end---------------------------
  for sfttype in sent ctx mix; do
    dataset_name="sft.$lang_pair.2w.${sfttype}"
    if [[ $sfttype == "sent" ]]; then
      dataset="$dataset_sent"
    elif [[ $sfttype == "ctx" ]]; then
      dataset="$dataset_ctx"
    else
      dataset="$dataset_mix"
    fi
    timestamp=$(date +"%Y%m%d-%H%M%S")
    log_file="$log_lang_dir/sft.${sfttype}.$timestamp.log"
    echo "====================start================================"
    echo "【$(date +"%Y-%m-%d %H:%M:%S")】开始跑实验：语言对: $lang_pair , sft ,sft类型: $sfttype"
    echo "写日志到:  $log_file"
    bash CPL_scripts/1-sft.sh "$model_path" "$model_type" "$extract_py_path" "$sent_testset" "$context_testset" "$testset_with_id" "$dataset" "$dataset_name" "$gpus_arg" > "$log_file" 2>&1
    echo "====================end=================================="
  done

done 
















