# ms-swift-CPL

**English** | [中文](README_CN.md)

This project implements a complete **CPL (Cross-Preference Learning)** training workflow based on **ms-swift**, covering:

- SFT training
- Data sampling
- CPL dataset construction
- CPL training / inference / evaluation

Multi-GPU training and DeepSpeed are supported.

---

## 1. Environment Setup

```bash
# Create and activate a Conda environment
conda create -n mycpo python=3.10 -y
conda activate mycpo

# Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Basic dependencies
conda install -c conda-forge pyarrow sentencepiece -y
pip install --upgrade pip

# Install ms-swift-CPL
cd ms-swift-CPL
pip install -e .

# Additional dependencies
conda install -c conda-forge sentence -y
pip install deepspeed liger-kernel
pip install "vllm==0.8.5.post1"
```

## 2. Overall Workflow (ms-swift-CPL/CPL_scripts/workflow.sh)

### 1. SFT (Supervised Fine-Tuning)

Perform SFT on a base model (such as Qwen3 / LLaMA3) using LoRA to obtain an adapter:

```bash
swift sft \
  --model your_path/models/qwen3-8B \
  --train_type lora \
  --dataset $dataset \
  --model_type qwen3 \
  --output_dir your_path/outputs/sft_ckpt \
  ...
```

This SFT adapter will be reused in the subsequent **sampling and CPL training** stages.

------

### 2. Sample from the SFT Model to Build a CPL Dataset

Use the SFT-trained model to sample from the validation set and generate the data required for CPL:

```bash
bash ms-swift-CPL/CPL_scripts/3-sample.sh "${args[@]}"
```

After sampling, **dataset registration code snippets** are generated automatically at:

```text
dataset/register-2w-mycpo-quanliang-langs/{model_type}/{adapter_name}.out
```

------

### 3. Register Custom CPL Datasets

Append the generated registration code to the custom dataset file:

```bash
cat dataset/register-2w-mycpo-quanliang-langs/.../*.out \
  >> dataset/custom_cpo_dataset.py
```

------

### 4. CPL Training / Inference / Evaluation

Use the SFT adapter and the registered CPL dataset for training and evaluation:

```bash
bash ms-swift-CPL/CPL_scripts/2-train_and_infer.sh "${args[@]}"
```

Logs and evaluation results are output and saved automatically.

------

## 3. CPL Dataset Registration Format

CPL datasets are registered using `register_dataset` and `DatasetMeta`.
Each dataset corresponds to a **JSONL file** in the **CPL quadruple format**, preprocessed by `MyCPOQuadruplePreprocessor`.

------

### 1. Full CPL (Intra + Cross)

```python
# CPL
register_dataset(
    DatasetMeta(
        dataset_name='en-de-2w-weighted-intra-and-cross',
        dataset_path='dataset/cpl/en-de/intra-and-cross.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))
```

------

### 2. Intra Preference Only (Without Cross)

```python
# CPL w/o cross
register_dataset(
    DatasetMeta(
        dataset_name='en-de-2w-weighted-intra',
        dataset_path='dataset/cpl/en-de/intra.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))
```

------

### 3. Cross Preference Only (Without Intra)

```python
# CPL w/o intra
register_dataset(
    DatasetMeta(
        dataset_name='en-de-2w-weighted-cross',
        dataset_path='dataset/cpl/en-de/cross.jsonl',
        preprocess_func=MyCPOQuadruplePreprocessor(),
        tags=['rlhf', 'cpo', 'custom'],
    ))
```

------

### Field Descriptions

- **dataset_name**
  The unique dataset identifier used to reference the dataset in training scripts
- **dataset_path**
  The path to the CPL dataset JSONL file

### Data Format

CPL datasets are stored in **JSONL** format. Each line represents a **CPL quadruple sample** used for preference modeling in contrastive learning.

#### Example Data

```json
{
  "x_win": "[English]:Saudi Arabia’s Game of Thrones\n  Given the provided context, translate the following English sentence to German.\n  [English]:PRINCETON – Saudi Arabia’s King Salman has just replaced the 57-year-old Muhammad bin Nayif with his 31-year-old son, Mohammed bin Salman, as crown prince, completing a process of power centralization that began with Salman’s accession to the throne in January 2015.",
  "x_lose": "[English]:Saudi Arabia’s Game of Thrones\n  Given the provided context, translate the following English sentence to German.\n  [English]:PRINCETON – Saudi Arabia’s King Salman has just replaced the 57-year-old Muhammad bin Nayif with his 31-year-old son, Mohammed bin Salman, as crown prince, completing a process of power centralization that began with Salman’s accession to the throne in January 2015.",
  "y_win": "PRINCETON – König Salman von Saudi-Arabien hat den 57-jährigen Muhammad bin Nayif gerade durch seinen 31-jährigen Sohn, Mohammed bin Salman, als Thronfolger ersetzt, was den Prozess der Machtzentralisierung beendet, der mit Salmans Thronbesteigung im Januar 2015 begonnen hat.",
  "y_lose": "PRINCETON – Der saudische König Salman hat den 57-jährigen Muhammad bin Nayif gerade mit seinem 31-jährigen Sohn, Mohammed bin Salman, als Thronerbe ersetzt, was das Abschließen eines Prozesses der Machtzentralisierung darstellt, der 2015 mit Salmans Thronbesteigung begonnen hat.",
  "mission": "intra",
  "win_value": 0.8598819552795172,
  "lose_value": 0.8308219193355322,
  "rule": "weighted",
  "meta": {
    "doc_id": 1,
    "src_sent_id": 1,
    "task_prefix": "ctx",
    "task_win": "ctx/sample",
    "task_lose": "ctx/sample"
  }
}
```

### CPL w/o cross (Intra Preference Only)

When training CPL w/o cross, the `"task_win"` and `"task_lose"` fields should both be `ctx/xxx` or both be `sent/xxx`, meaning they come from the same task.

The `"x_win"` and `"x_lose"` fields have identical contents, corresponding to $`x`$ in Equation (2) of the paper; `"y_win"` corresponds to $`y^+`$, and `"y_lose"` corresponds to $`y^-`$.

```math
\begin{aligned}
\mathcal{L}_{\text{intra}}(\theta)
&= \mathbb{E}_{(y^+, y^-)\sim P_s}
   \mathcal{L}_{\text{CPO}}(x, y^+, y^-) \\
&\quad + \mathbb{E}_{(y^+, y^-)\sim P_c}
   \mathcal{L}_{\text{CPO}}((x,c), y^+, y^-)
\end{aligned}
\tag{2}
```

### CPL w/o intra (Cross Preference Only)

When training CPL w/o intra, one of the `"task_win"` and `"task_lose"` fields should be `ctx/xxx` and the other `sent/xxx`, meaning they come from different tasks.

The `"x_win"` field corresponds to $`x_w^+`$ in Equation (3) of the paper, and `"x_lose"` corresponds to $`x_l^+`$; `"y_win"` corresponds to $`y^+`$, and `"y_lose"` corresponds to $`y^-`$.

```math
\begin{aligned}
\mathcal{L}_{\text{cross}}(\theta)
&= \mathbb{E}_{(y_w^+, y_l^+)}
   \mathcal{L}_{\text{C-CPO}}(\{x_w^+, x_l^+\}, y_w^+, y_l^+) \\
&\quad + \mathbb{E}_{(y_w^+, y_l^-)}
   \mathcal{L}_{\text{C-CPO}}(\{x_w^+, x_l^-\}, y_w^+, y_l^-)
\end{aligned}
\tag{3}
```

### Full CPL

When training full CPL, the `"task_win"` and `"task_lose"` fields may come from either the same task or different tasks, and are substituted into Equation (5) of the paper:

```math
\begin{aligned}
\mathcal{L}_{\text{CPL}}(\theta)
&= \mathcal{L}_{\text{intra}}(\theta)
- \mathcal{L}_{\text{cross}}(\theta)
\end{aligned}
\tag{5}
```

------

## 4. Notes

- **Absolute paths must be updated manually** in all scripts
- CPL-related hyperparameters (such as `lambda_intra`, `lambda_cross`, `lr`, and `global_bs`) are fixed in the scripts
- Multi-GPU training with DeepSpeed ZeRO-2 is supported
- Use the same tokenizer and model version during sampling and training

------
