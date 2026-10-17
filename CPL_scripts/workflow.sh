#!/bin/bash
cd your_path/ms-swift-CPL
# 1.sft训练：（或者用ms-swift-CPL\CPL_scripts\1-sft.sh与CPL_scripts\1-top.sh，但手动修改其中所有绝对路径）
model_path="your_path/models/qwen3-8B"
model_type="qwen3"
dataset="dataset/sft/en-de/train.en-de.256ctx.90.head2w.msg.mix.jsonl"  # 根据实际情况修改
OUT_DIR="your_path/outputs/sft_ckpt"
GPU_N=4  # 根据实际GPU情况修改
NPROC_PER_NODE=$GPU_N \
swift sft \
    --model $model_path \
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
    --dataset_num_proc 8 
#如果要sent-level和ctx-aware两个任务单独sft，则把上面dataset改成相应的train.en-de.sent.90.head2w.msg.jsonl和train.en-de.256ctx.90.head2w.msg.jsonl；

# 2.利用sft得到的模型，sample得到所需数据,并做成CPL数据集,生成数据集注册信息
model_path="your_path/models/qwen3-8B"
model_type="qwen3"
extract_py_path="CPL_scripts/eval/extract_output_for_eval.py"
train_adapter="your_path/outputs/sft_ckpt"
adapter_name="your_sft_ckpt_name"  # 根据实际情况修改，自定义即可
pickset_sent="dataset/sft/en-de/valid.en-de.sent.jsonl"  # 用valid集生成sample数据
pickset_ctx="dataset/sft/en-de/valid.en-de.256ctx.msg.jsonl"  # 用valid集生成sample数据
pickset_with_id="dataset/sft/en-de/valid.en-de.sent_withid.jsonl"  # 用valid集生成sample数据
gpus_arg="0,1,2,3"  # 根据实际GPU情况修改
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
#要修改ms-swift-CPL/CPL_scripts/3-sample.sh中ROOT_PATH为实际路径
bash ms-swift-CPL/CPL_scripts/3-sample.sh "${args[@]}" 
#跑完后，数据集注册信息路径生成在：dataset/register-2w-mycpo-quanliang-langs/${model_type}/${adapter_name}.out:

# 3. 把刚刚的生成的注册数据集表单 追加到自定义数据集文件dataset/custom_cpo_dataset.py
cat  dataset/register-2w-mycpo-quanliang-langs/${model_type}/${adapter_name}.out >> dataset/custom_cpo_dataset.py

# 4.CPL训练+推理+评测：
model_path="your_path/models/qwen3-8B-or-llama3" # 根据实际情况修改
model_type="qwen3/llama" # 根据实际情况修改
extract_py_path="CPL_scripts/eval/extract_output_for_eval.py" #固定
train_adapter="your_path/outputs/sft_ckpt"  # 使用sft训练得到的adapter
dataset_name_mycpo="en-de-2w-weighted-k1-intra-and-cross" # 根据dataset/custom_cpo_dataset.py中注册数据集的实际名称修改
sent_testset="dataset/sft/en-de/test.en-de.sft.sent.jsonl"  
context_testset="dataset/sft/en-de/test.en-de.256ctx.msg.jsonl" 
testset_with_id="dataset/sft/en-de/test.en-de.sft.sent_withid.jsonl" 
lambda_intra=1.0 #固定超参数
lambda_cross=1.0 #固定超参数
lambda_bc=1.0 #固定超参数
lr=5e-5 #固定超参数
global_bs=128 #固定超参数
epoch=2 #固定超参数
log_path="your_path/xx.log"
gpus_arg="0,1,2,3"  # 根据实际GPU情况修改
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
mycpo_bash="CPL_scripts/2-train_and_infer.sh"
bash "$mycpo_bash" "${args[@]}" >> "${log_path}" 2>&1




