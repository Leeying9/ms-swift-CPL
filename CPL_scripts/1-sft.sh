#!/bin/bash
#增加了modelpath等param
ROOT_PATH="/data/liying/ms-swift-CPL"
cd $ROOT_PATH
source ~/miniconda/etc/profile.d/conda.sh
conda activate mycpo
export OMP_NUM_THREADS=16
model_path=$1
model_type=$2
extract_py_path=$3
#--------如果要用adapter就改下面两行-------------
train_adapter=$4
sent_testset=$5
context_testset=$6
testset_with_id=$7
dataset=$8
dataset_name=$9
gpus_arg=${10:-"0,1,2,3"}
# gpus_arg="4,5,6,7"
IFS=',' read -ra GPUS <<< "$gpus_arg"
GPU_N=${#GPUS[@]}
export GPU_N
export CUDA_VISIBLE_DEVICES="$(IFS=,; printf '%s' "${GPUS[*]}")"
#echo区域#
echo "接收到参数model_path=${model_path}"
echo "接受到参数model_type=$model_type"
echo "接受到参数extract_py_path=$extract_py_path"
echo "接收到train_adapter是：$train_adapter"
echo "接受到参数sent_testset=$sent_testset"
echo "接受到参数context_testset=$context_testset"
echo "接受到参数testset_with_id=$testset_with_id"
echo "接受到参数dataset=$dataset"
echo "接受到参数dataset_name=$dataset_name"
echo "接受到参数gpus_arg=$gpus_arg"
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

OUT_DIR=$ROOT_PATH/sft-ckpts/$(basename $model_path)/$dataset_name/bs256
train_name=sft
mkdir -p "$OUT_DIR"
log_name=train_$(date +'%Y%m%d_%H%M%S').log
echo "log 放到： $OUT_DIR/$log_name"

# -------- train_adapter 可选：存在才传给 swift rlhf --------
adapter_args=()
if [[ -n "$train_adapter" ]]; then
  if [[ -e "$train_adapter" ]]; then
    echo "[INFO] 使用 train_adapter: $train_adapter"
    adapter_args=(--adapters "$train_adapter")
  else
    echo "[WARN] train_adapter 路径不存在：$train_adapter （将忽略，不传 --adapters）" >&2
  fi
else
  echo "[INFO] 未提供 train_adapter，不传 --adapters"
fi
# ------------------------------------------------------

NPROC_PER_NODE=$GPU_N \
swift sft \
    --model $model_path \
    "${adapter_args[@]}" \
    --train_type lora \
    --dataset $dataset \
    --gradient_checkpointing \
    --model_type $model_type \
    --save_strategy epoch \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --output_dir $OUT_DIR \
    --dataloader_num_workers 8 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --deepspeed zero2 \
    --dataset_num_proc 8 \
  > "$OUT_DIR/$log_name" 2>&1 

if [ $? -eq 0 ]; then
    echo 'train done'
else
    echo 'train failed' >&2
    exit 1
fi

# ================= 基础环境与路径 =================
echo 'infer begin '
# conda 初始化与环境
source ~/miniconda/etc/profile.d/conda.sh
conda activate mycpo
# adapter 根目录（每个 ckpt 在这里找 checkpoint-XXX）
A=$OUT_DIR
latest_dir=$(
  find "$A" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
  | awk '/^v[0-9]+/ {print $0 "\t" substr($0,2)+0}' \
  | sort -k2,2n -k1,1 | tail -n1 | cut -f1
)
[ -n "$latest_dir" ] || { echo "No vN* directory under $A" >&2; exit 1; }
echo "[INFO] Latest version dir: $latest_dir"
# adapter_root=/data/ly/ms-swift-main/ms-swift-main/yli/cpo-checkpoints/mycpo4/mycpo4_v2.0/v0-20251012-232615
adapter_root="$A/$latest_dir"

# ================ 最新ckpt ================

latest_ckpt=$(
  find "$adapter_root" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
  | sed -E 's@.*/checkpoint-([0-9]+)@\1@' \
  | sort -n | tail -n1
)
[ -n "$latest_ckpt" ] || { echo "No checkpoint-* dirs under $adapter_root" >&2; exit 1; }
ckpt_list=("$latest_ckpt")

# ==========================================================

for ck in "${ckpt_list[@]}"; do
  adapter="${adapter_root}/checkpoint-${ck}"
  ckpt_name="checkpoint-${ck}"

  if [[ ! -d "$adapter" ]]; then
    echo "跳过 ${ckpt_name}: 目录不存在 -> ${adapter}"
    continue
  fi

  echo ">>>> 开始处理 ${ckpt_name}"

  base_pred_dir="$ROOT_PATH/pred/${train_name}/$(basename $model_path)/${dataset_name}/${latest_dir}/${ckpt_name}"
  pred_file1="${base_pred_dir}/sent/swift_generated_predictions.jsonl"
  pred_file2="${base_pred_dir}/context/swift_generated_predictions.jsonl"

  mkdir -p "$(dirname "$pred_file1")" "$(dirname "$pred_file2")"

  # 重新确保在推理环境
  conda activate mycpo

#   -------- sent 推理 --------
  NPROC_PER_NODE=$GPU_N swift infer \
    --model "$model_path" \
    --adapters "$adapter" \
    --model_type $model_type \
    --infer_backend vllm \
    --torch_dtype float16 \
    --seed 42 \
    --temperature 0.0 --top_p 1 --top_k 1 \
    --result_path "$pred_file1" \
    --val_dataset "$sent_testset" \
    --max_batch_size 16 \
    --max_new_tokens 4096 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 6000 \
    --vllm_max_num_seqs 64 \
    --vllm_gpu_memory_utilization 0.88 \
    --vllm_enforce_eager True \
    --dataset_num_proc 4 

#   -------- context 推理 --------
  NPROC_PER_NODE=$GPU_N swift infer \
    --model "$model_path" \
    --adapters "$adapter" \
    --model_type $model_type \
    --infer_backend vllm \
    --torch_dtype float16 \
    --seed 42 \
    --temperature 0.0 --top_p 1 --top_k 1 \
    --result_path "$pred_file2" \
    --val_dataset "$context_testset" \
    --max_batch_size 16 \
    --max_new_tokens 4096 \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 6000 \
    --vllm_max_num_seqs 64 \
    --vllm_gpu_memory_utilization 0.88 \
    --vllm_enforce_eager True \
    --dataset_num_proc 4 

  if [[ ! -f "$pred_file1" || ! -f "$pred_file2" ]]; then
    echo "跳过 ${ckpt_name}: 缺少 pred 文件"
    continue
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
    #记得改了测试集的话也要改这里

  echo "[FINAL][POST] eval (sent) "
  python $extract_py_path \
    --swift_file $pred_file1 \
    --ref_file $testset_with_id \
    --output  $output_file1\
    --task "sent/greedy" \
    --group_size 1

  python CPL_scripts/eval/comet_bleu_dcomet.py \
  --input_jsonl  $output_file1 \
  --testset_with_id $testset_with_id \
  --output_jsonl $eval_dir1/eval.jsonl \
  --summary_file $eval_dir1/eval.out.txt \
  --context_window 5 \
  --gpus $GPU_N \
  --batch_size_comet 64 \
  --batch_size_dcomet 64

  echo "[FINAL][POST] eval (ctx) "
  python $extract_py_path \
    --swift_file $pred_file2 \
    --ref_file $testset_with_id \
    --output  $output_file2 \
    --task "ctx/greedy" \
    --group_size 1

  python CPL_scripts/eval/comet_bleu_dcomet.py \
  --input_jsonl  $output_file2 \
  --testset_with_id $testset_with_id \
  --output_jsonl $eval_dir2/eval.jsonl \
  --summary_file $eval_dir2/eval.out.txt \
  --context_window 5 \
  --gpus $GPU_N \
  --batch_size_comet 64 \
  --batch_size_dcomet 64

  echo "<<<< 完成 ${ckpt_name}"
done

echo "全部指定 ckpt 处理完成。"

