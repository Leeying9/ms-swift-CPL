#!/bin/bash
# adapter、
# pick_set(两个)、
# pickset_with_id（1个，必须是全量的，即使set是10%这里也要给100%的withid文件）
# ，去sample和greedy采样并且评测
# 参考 $ROOT_PATH/256ctx/bash/90mix-right/1205-2w-ende-newparam/sample-mix-2w/greedy-mix-2w.sh
# 基座模型
model_path=$1
model_type=$2
extract_py_path=$3
adapter=$4
adapter_name=$5
pickset_sent=$6
pickset_ctx=$7
pickset_with_id=$8

ROOT_PATH="YOUR_PATH/ms-swift-CPL"
cd $ROOT_PATH
output_dir="sample-output/$(basename "$model_path")/${adapter_name}/all"
mkdir -p "$output_dir"
source /data/liying/miniconda/etc/profile.d/conda.sh
conda activate mycpo
export OMP_NUM_THREADS=16
gpus_arg=${9:-"0,1,2,3"}
# gpus_arg="4,5,6,7"
IFS=',' read -ra GPUS <<< "$gpus_arg"
GPU_N=${#GPUS[@]}
export GPU_N
export CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${GPUS[*]}")"

#====echo area====
echo "接收到参数model_path：$model_path"
echo "接收到参数model_type：$model_type"
echo "接收到参数extract_py_path：$extract_py_path"
echo "接收到参数adapter：$adapter"
echo "接收到参数adapter_name：$adapter_name"
echo "接收到参数pickset_sent: $pickset_sent"
echo "接收到参数pickset_ctx $pickset_ctx"
echo "接收到参数pickset_with_id $pickset_with_id"
echo "接收到参数gpus_arg $gpus_arg"
#adapter_name的格式是"sft.$lang_pair.2w.mix"，从中分离出lang_pair
# lang_pair=$(echo "$adapter_name" | sed -E 's/^sft\.([^.]+)\.2w\.mix$/\1/')
lang_pair=$(echo "$adapter_name" | sed -nE 's/^sft\.([^.]+)\.2w\.mix$/\1/p')
if [[ -z "$lang_pair" ]]; then
  echo "ERROR: adapter_name format unexpected 导致无法分离出langpair: $adapter_name" >&2
  exit 1
fi
echo "lang_pair=$lang_pair"
# ================保险睡眠区域==============
# GPUS 有一张不空就睡10min
while true; do
    busy=0

    # 遍历数组中的所有 GPU id
    for id in "${GPUS[@]}"; do
        count=$(nvidia-smi -i "$id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                | sed '/^$/d' | wc -l)
        busy=$((busy + count))
    done

    if [ "$busy" -eq 0 ]; then
        echo "GPU ${GPUS[*]} 全部空闲，退出睡眠。"
        break
    else
        echo "GPU ${GPUS[*]} 共 $busy 个进程，睡眠 10 分钟..."
        sleep 600
    fi
done
# # ==============================================
# ==========开始寻找可用port[随机起始点]==============
BASE_PORT=29600
MAX_PORT=29700  # 你可以改大一点

# 计算范围
RANGE=$((MAX_PORT - BASE_PORT + 1))
# 随机起点
start_port=$((BASE_PORT + RANDOM % RANGE))
try_port=$start_port

echo "Random start port: $start_port"

while ss -ltn | awk '{print $4}' | grep -q ":$try_port$"; do
  # 当前端口被占用，换下一个
  try_port=$((try_port + 1))

  # 超过最大端口就从头开始
  if [ "$try_port" -gt "$MAX_PORT" ]; then
    try_port=$BASE_PORT
  fi

  # 如果兜了一圈又回到起点，说明整个区间都没空
  if [ "$try_port" -eq "$start_port" ]; then
    echo "No free port found in [$BASE_PORT, $MAX_PORT]" >&2
    exit 1
  fi
done

export MASTER_PORT="$try_port"
echo "Using MASTER_PORT=$MASTER_PORT"
# ==========结束port设置（固定不用改）==============
OUT_PREFIX=sentpart 
OUT_DIR="sample-output/$(basename "$model_path")/${adapter_name}/sample/${OUT_PREFIX}"
mkdir -p "$OUT_DIR"

for ((i=0; i<GPU_N; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
  swift sample \
    --model "$model_path" \
    --adapters "$adapter" \
    --model_type "$model_type" \
    --sampler_engine vllm \
    --num_return_sequences 8 \
    --num_sampling_per_gpu_batch_size 32 \
    --dataset "$pickset_sent" \
    --output_dir "$OUT_DIR" \
    --output_file "${OUT_PREFIX}-proc${i}.jsonl" \
    --override_exist_file True \
    --temperature 0.7 \
    --use_hf False \
    --max_length 6000 \
    --max_new_tokens 4096 \
    --data_range $i $GPU_N \
    > "$OUT_DIR/${OUT_PREFIX}-proc${i}.log" 2>&1 &
done
wait

echo "合并所有proc{i}.jsonl到${OUT_PREFIX}-merged.jsonl"
#合并所有proc{i}.jsonl到${OUT_PREFIX}-merged.jsonl
MERGED_JSONL="$OUT_DIR/${OUT_PREFIX}-merged.jsonl"
: > "$MERGED_JSONL"   # 先清空/创建
for ((i=0; i<GPU_N; i++)); do
  f="$OUT_DIR/${OUT_PREFIX}-proc${i}.jsonl"
  [[ -s "$f" ]] || { echo "ERROR: missing/empty $f" >&2; exit 1; }
done

for ((i=0; i<GPU_N; i++)); do
    echo "merge ${OUT_PREFIX}-proc${i}.jsonl"
    cat "$OUT_DIR/${OUT_PREFIX}-proc${i}.jsonl" >> "$MERGED_JSONL"
done
merged1=$MERGED_JSONL
our_dir1=$OUT_DIR
# 结束合并；
echo "All sent-part sampling done."


############################################################################################################################
OUT_PREFIX=ctxpart 
OUT_DIR="sample-output/$(basename "$model_path")/${adapter_name}/sample/${OUT_PREFIX}"
mkdir -p "$OUT_DIR"

for ((i=0; i<GPU_N; i++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
  swift sample \
    --model "$model_path" \
    --adapters "$adapter" \
    --model_type "$model_type" \
    --sampler_engine vllm \
    --num_return_sequences 8 \
    --num_sampling_per_gpu_batch_size 32 \
    --dataset "$pickset_ctx" \
    --output_dir "$OUT_DIR" \
    --output_file "${OUT_PREFIX}-proc${i}.jsonl" \
    --override_exist_file True \
    --temperature 0.7 \
    --use_hf False \
    --max_length 6000 \
    --max_new_tokens 4096 \
    --data_range $i $GPU_N \
    > "$OUT_DIR/${OUT_PREFIX}-proc${i}.log" 2>&1 &
done
wait

echo "合并所有proc{i}.jsonl到${OUT_PREFIX}-merged.jsonl"
#合并所有proc{i}.jsonl到${OUT_PREFIX}-merged.jsonl
MERGED_JSONL="$OUT_DIR/${OUT_PREFIX}-merged.jsonl"
: > "$MERGED_JSONL"   # 先清空/创建
for ((i=0; i<GPU_N; i++)); do
  f="$OUT_DIR/${OUT_PREFIX}-proc${i}.jsonl"
  [[ -s "$f" ]] || { echo "ERROR: missing/empty $f" >&2; exit 1; }
done

for ((i=0; i<GPU_N; i++)); do
    echo "merge ${OUT_PREFIX}-proc${i}.jsonl"
    cat "$OUT_DIR/${OUT_PREFIX}-proc${i}.jsonl" >> "$MERGED_JSONL"
done
# 结束合并；
echo "All ctx-part sampling done."
############################################################################################################################

# extract sent
python $extract_py_path \
    --swift_file $merged1 \
    --ref_file $pickset_with_id \
    --output $our_dir1/sentpart-merged.clean.id.jsonl \
    --task "sent/sample" \
    --group_size 8

python $extract_py_path \
    --swift_file $MERGED_JSONL \
    --ref_file $pickset_with_id \
    --output $OUT_DIR/ctxpart-merged.clean.id.jsonl \
    --task "ctx/sample" \
    --group_size 8

  conda activate comet
  output_file1=$our_dir1/sentpart-merged.clean.id.jsonl
  output_file2=$OUT_DIR/ctxpart-merged.clean.id.jsonl
  eval_dir1="${our_dir1}/sent/output/eval-output"
  eval_dir2="${OUT_DIR}/context/output/eval-output2"

  mkdir -p "$(dirname "$output_file1")" "$(dirname "$output_file2")" "$eval_dir1" "$eval_dir2"

  
  echo "[FINAL][POST]  (sent) "
  python $ROOT_PATH/CPL_scripts/eval/comet_bleu_dcomet.py \
  --input_jsonl  $output_file1 \
  --testset_with_id $pickset_with_id \
  --output_jsonl $eval_dir1/eval.jsonl \
  --summary_file $eval_dir1/eval.out.txt \
  --context_window 5 \
  --gpus $GPU_N \
  --batch_size_comet 64 \
  --batch_size_dcomet 64

  echo "[FINAL][POST] (ctx) "
  python $ROOT_PATH/CPL_scripts/eval/comet_bleu_dcomet.py \
  --input_jsonl  $output_file2 \
  --testset_with_id $pickset_with_id \
  --output_jsonl $eval_dir2/eval.jsonl \
  --summary_file $eval_dir2/eval.out.txt \
  --context_window 5 \
  --gpus $GPU_N \
  --batch_size_comet 64 \
  --batch_size_dcomet 64

sample_sent=$eval_dir1/eval.jsonl
sample_ctx=$eval_dir2/eval.jsonl

###########################################################
echo "【info】sample部分结束，开始greedy部分"
cd /data/liying/beiqu-swift
# conda 初始化与环境
conda activate mycpo

  base_pred_dir="sample-output/$(basename "$model_path")/${adapter_name}/greedy"

  pred_file1="${base_pred_dir}/sent/swift_generated_predictions.jsonl"
  pred_file2="${base_pred_dir}/context/swift_generated_predictions.jsonl"

  mkdir -p "$(dirname "$pred_file1")" "$(dirname "$pred_file2")"

  # 重新确保在推理环境
  conda activate mycpo

  # -------- sent 推理 --------
  NPROC_PER_NODE=$GPU_N swift infer \
    --model "$model_path" \
    --adapters "$adapter" \
    --model_type $model_type \
    --infer_backend vllm \
    --torch_dtype float16 \
    --seed 42 \
    --temperature 0.0 --top_p 1 --top_k 1 \
    --result_path "$pred_file1" \
    --val_dataset "$pickset_sent" \
    --max_batch_size 16 \
    --max_new_tokens 4096 \
    --vllm_tensor_parallel_size $GPU_N \
    --vllm_max_model_len 6000 \
    --vllm_max_num_seqs 64 \
    --vllm_gpu_memory_utilization 0.88 \
    --vllm_enforce_eager True \
    --dataset_num_proc 8 

  # -------- context 推理 --------
  NPROC_PER_NODE=$GPU_N swift infer \
    --model "$model_path" \
    --adapters "$adapter" \
    --model_type "$model_type" \
    --infer_backend vllm \
    --torch_dtype float16 \
    --seed 42 \
    --temperature 0.0 --top_p 1 --top_k 1 \
    --result_path "$pred_file2" \
    --val_dataset "$pickset_ctx" \
    --max_batch_size 16 \
    --max_new_tokens 4096 \
    --vllm_tensor_parallel_size $GPU_N \
    --vllm_max_model_len 6000 \
    --vllm_max_num_seqs 64 \
    --vllm_gpu_memory_utilization 0.88 \
    --vllm_enforce_eager True \
    --dataset_num_proc 8 


  if [[ ! -f "$pred_file1" || ! -f "$pred_file2" ]]; then
    echo "ERROR: 缺少 greedy 的pred 文件 "
    exit 1
  fi

  # ========= 后处理 & 评测 =========
  conda activate comet

  output_file1="${base_pred_dir}/sent/output/1-new.jsonl"
  output_file2="${base_pred_dir}/context/output/1-new.jsonl"
  eval_dir1="${base_pred_dir}/sent/output/eval-output"
  eval_dir2="${base_pred_dir}/context/output/eval-output2"

  output_file1_dcomet="${base_pred_dir}/sent/output/sent_for_dcomet.jsonl"
  output_file2_dcomet="${base_pred_dir}/context/output/context_for_dcomet.jsonl"

  mkdir -p "$(dirname "$output_file1")" "$(dirname "$output_file2")" "$eval_dir1" "$eval_dir2"

  echo "[FINAL][POST] extract (sent) "
  python $extract_py_path \
    --swift_file $pred_file1 \
    --ref_file $pickset_with_id \
    --output  $output_file1\
    --task "sent/greedy" \
    --group_size 1

  python $ROOT_PATH/CPL_scripts/eval/comet_bleu_dcomet.py \
  --input_jsonl  $output_file1 \
  --testset_with_id $pickset_with_id \
  --output_jsonl $eval_dir1/eval.jsonl \
  --summary_file $eval_dir1/eval.out.txt \
  --context_window 5 \
  --gpus $GPU_N \
  --batch_size_comet 64 \
  --batch_size_dcomet 64

  echo "[FINAL][POST] extract (ctx) "
  python $extract_py_path \
    --swift_file $pred_file2 \
    --ref_file $pickset_with_id \
    --output  $output_file2 \
    --task "ctx/greedy" \
    --group_size 1

  python $ROOT_PATH/CPL_scripts/eval/comet_bleu_dcomet.py \
  --input_jsonl  $output_file2 \
  --testset_with_id $pickset_with_id \
  --output_jsonl $eval_dir2/eval.jsonl \
  --summary_file $eval_dir2/eval.out.txt \
  --context_window 5 \
  --gpus $GPU_N \
  --batch_size_comet 64 \
  --batch_size_dcomet 64

greedy_sent=$eval_dir1/eval.jsonl
greedy_ctx=$eval_dir2/eval.jsonl

echo "【info】greedy部分结束，开始四路归一"

# 4路归一 
merged_4ways_jsonl=$output_dir/${model_type}-${lang_pair}-merged-4ways.jsonl
python $ROOT_PATH/CPL_scripts/utils/merge-4-decode-results-by-id.py \
    --greedy_sent $greedy_sent \
    --greedy_ctx  $greedy_ctx \
    --sample_sent $sample_sent \
    --sample_ctx  $sample_ctx \
    --output      $merged_4ways_jsonl

echo "开始挑选数据（全量）"
    python $ROOT_PATH/CPL_scripts/utils/pick-intra-duplicate.py \
      --input $merged_4ways_jsonl \
      --output $ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/intra.jsonl \
      --readme $ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/intra.jsonl.readme \
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

    python $ROOT_PATH/CPL_scripts/utils/pick-cross-duplicate.py \
      --input $merged_4ways_jsonl \
      --output $ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/cross.jsonl \
      --readme $ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/cross.jsonl.readme \
      --rule weighted \
      --w-comet 0.5 \
      --w-bleu  0.0 \
      --w-dcomet 0.5 \
      --mission cross \
      --min-len 6 \
      --max-len 200 \
      --min-win-score 0.30 \
      --min-gap 0.002 \
      --max-gap 0.1 \
      --max-pairs-per-group 1 \
      --gap-bins "0,0.002,0.005,0.006,0.008,0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090"
    
    cat "$ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/intra.jsonl" "$ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/cross.jsonl" > "$ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/intra-and-cross.jsonl"
    
echo "【info】挑选全量-数据结束，开始生成全量-数据集注册内容，输出到$ROOT_PATH/dataset/register边（全量）"
output_file="dataset/register-2w-mycpo-quanliang-langs/${model_type}/${adapter_name}.out"
mkdir -p "$(dirname "$output_file")"
> "$output_file"  # 清空已有文件内容

cat <<EOF >> "$output_file"
#---------------------
#全量 2w-weighted-k1-${lang_pair}-${model_type}
register_dataset(
    DatasetMeta(
        dataset_name='${model_type}-${lang_pair}-2w-weighted-k1-intra',
        dataset_path='$ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/intra.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))

register_dataset(
    DatasetMeta(
        dataset_name='${model_type}-${lang_pair}-2w-weighted-k1-cross',
        dataset_path='$ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/cross.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))


register_dataset(
    DatasetMeta(
        dataset_name='${model_type}-${lang_pair}-2w-weighted-k1-intra-and-cross',
        dataset_path='$ROOT_PATH/dataset/2w-mycpo-quanliang-langs/${model_type}/${adapter_name}/weighted/intra-and-cross.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))
#-----------------------------
EOF


echo "已写入到 $output_file"
#==============================================