/data/liying/exp_queue/queueA/done_queue/44-sample-llama3-langs-A.sh
#!/bin/bash
model_path="/data/liying/models/llama3-8b-instruct"
model_type=llama3
extract_py_path="/data/liying/ms-swift/yli/util_pys/pick-workflow/3-fix-new-extract-function-sort-langs-jr-llama3.py"
LOG_BASE="/data/liying/exp_queue/universal-scripts/othermodel-sft-mycpo/logs/$(basename $model_path)"
# gpus_arg="4,5,6,7"
gpus_arg="0,1,2,3"

# for lang_pair in en-de en-es en-fr; do
for lang_pair in en-de en-es en-fr; do
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

  #####==================跑采样=========================
  #得到train_adapter路径end===================###
  sft_dataset_name="sft.$lang_pair.2w.mix"
  A="/data/liying/beiqu-swift/yli/sft-ckpts/$(basename $model_path)/$sft_dataset_name/bs256"
  # ================ 最新version ================
  latest_dir=$(
  find "$A" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
  | awk '/^v[0-9]+/ {print $0 "\t" substr($0,2)+0}' \
  | sort -k2,2n -k1,1 | tail -n1 | cut -f1
  )
  [ -n "$latest_dir" ] || { echo "No vN* directory under $A" >&2; continue; }
  echo "[INFO] Latest version dir: $latest_dir"
  # adapter_root=/data/ly/ms-swift-main/ms-swift-main/yli/cpo-checkpoints/mycpo4/mycpo4_v2.0/v0-20251012-232615
  adapter_root="$A/$latest_dir"    
  # ================ 最新ckpt ================
  latest_ckpt=$(
  find "$adapter_root" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
  | sed -E 's@.*/checkpoint-([0-9]+)@\1@' \
  | sort -n | tail -n1
  )
  [ -n "$latest_ckpt" ] || { echo "No checkpoint-* dirs under $adapter_root" >&2; continue; }
  # ================ 最新ckpt end================
  train_adapter="${adapter_root}/checkpoint-${latest_ckpt}"
  #如果没有adapter，就直接写为空字符串就行
  # train_adapter=""
  #得到train_adapter路径end===================###
  #====================================开始推理sample===========================
    adapter_name="$sft_dataset_name"
    if [[ $lang_pair == "en-de" ]];then 
        pickset_sent=/data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/sent_msg_valid.jsonl
        pickset_ctx=/data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.256ctx.msg.jsonl
        pickset_with_id=/data/liying/ms-swift/yli/dataset/256ctx/pickset/valid/valid.en-de.sent.json
    else
        pickset_sent=/data/liying/beiqu-swift/yli/dataset/sft/$lang_pair/valid.$lang_pair.withid.msg.jsonl
        pickset_ctx=/data/liying/beiqu-swift/yli/dataset/sft/$lang_pair/valid.$lang_pair.256ctx.msg.jsonl
        pickset_with_id=/data/liying/beiqu-swift/yli/dataset/sft/$lang_pair/withid/valid.$lang_pair.withid.jsonl
    fi
    args=(
    "$model_path"
    "$model_type"
    "$extract_py_path"
    "$train_adapter"
    "$adapter_name"
    "$pickset_sent"
    "$pickset_ctx"
    "$pickset_with_id"
    "$gpus_arg"
    )

    log_path="$log_lang_dir/sample.$timestamp.log"
    echo "==================【info】===========================" | tee -a "${log_path}"
    echo "Start: $(date)" | tee -a "${log_path}"
    echo "【sample】" | tee -a "${log_path}"
    echo "Log file: ${log_path}" | tee -a "${log_path}"
    echo "==================================================" | tee -a "${log_path}"

    bash ms-swift-CPL/CPL_scripts/3-sample.sh "${args[@]}" >> "${log_path}" 2>&1
  
done 
