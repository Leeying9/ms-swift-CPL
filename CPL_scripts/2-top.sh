#!/bin/bash
model_path="/data/liying/models/Qwen3-4B/qwen3-4b"
model_type=qwen3
extract_py_path="/data/liying/ms-swift/yli/util_pys/pick-workflow/3-fix-new-extract-function-sort-nojump.py"
LOG_BASE="/data/liying/exp_queue/universal-scripts/othermodel-sft-mycpo/logs/$(basename $model_path)"
# gpus_arg="4,5,6,7"
gpus_arg="0,1,2,3"
#en-es en-fr en-it en-nl en-ru
for lang_pair in en-de; do
  log_lang_dir="$LOG_BASE/$lang_pair"
  mkdir -p "$log_lang_dir"
  #==================开始sft=========================
  #------------------测试集---------------------------
  sent_testset=/data/liying/beiqu-swift/yli/dataset/en-de-testset/test.en-de.sft.sent.json
  context_testset=/data/liying/beiqu-swift/yli/dataset/en-de-testset-256ctx/test.en-de.256ctx.msg.jsonl
  testset_with_id=/data/liying/ms-swift/yli/util_pys/eval/dcomet/dataset_with_id/test.en-de.sft.sent_withid.jsonl
  dataset_sent=/data/liying/beiqu-swift/yli/dataset/256-sft-right/sent-final-train/train.en-de.sent.90.message.head2w.jsonl
  dataset_ctx=/data/liying/beiqu-swift/yli/dataset/256-sft-right/ctx-final-train/train.en-de.256ctx.90.message.head2w.jsonl
  dataset_mix=/data/liying/beiqu-swift/yli/dataset/256-sft-right/mix/train.en-de.256ctx.90.message.head2w.mix.jsonl
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
    bash /data/liying/exp_queue/universal-scripts/sft/train-and-infer-sft-2w-moreparam.sh "$model_path" "$model_type" "$extract_py_path" "$sent_testset" "$context_testset" "$testset_with_id" "$dataset" "$dataset_name" "$gpus_arg" > "$log_file" 2>&1
    echo "====================end=================================="
  done
  #==================sft结束=========================
  #####==================跑mycpo的方法=========================
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

  for datatype in intra intra-and-cross cross;do
      dataset_name_mycpo=${lang_pair}-2w-weighted-k1-${datatype}
      lambda_intra=1.0
      lambda_cross=1.0
      lambda_bc=1.0
      lr=5e-5
      global_bs=128
      epoch=2
      timestamp=$(date +"%Y%m%d-%H%M%S")
      log_path="$log_lang_dir/mycpo.${datatype}.$timestamp.log"
      args=(
        "$model_path"
        "$model_type"
        "$extract_py_path"
        "$train_adapter"
        "$dataset_name_mycpo"
        "$sent_testset"
        "$context_testset"
        "$testset_with_id"
        "$lambda_intra"
        "$lambda_cross"
        "$lambda_bc"
        "$lr"
        "$global_bs"
        "$epoch"
        "$gpus_arg"
      )

        mycpo_bash="ms-swift-CPL/CPL_scripts/2-train_and_infer.sh"

      echo "==================【info】===========================" | tee -a "${log_path}"
      echo "Start: $(date)" | tee -a "${log_path}"
      echo "【mycpo训练：${datatype}】lambda_cross=${lambda_cross}, lr=${lr}, global_bs=${global_bs}" | tee -a "${log_path}"
      echo "Log file: ${log_path}" | tee -a "${log_path}"
      echo "==================================================" | tee -a "${log_path}"

      bash "$mycpo_bash" "${args[@]}" >> "${log_path}" 2>&1
  done
done 
















