# ReMoT: Reinforcement Learning with Motion Contrast Triplets

This repository provides the **dataset**, **model**, and **evaluation code** accompanying our paper:

> **"[[ReMoT: Reinforcement Learning with Motion Contrast Triplets\]](https://arxiv.org/abs/2603.00461)"**
>  *Accepted as a Highlight Paper at CVPR 2026*

- 🔗 **Paper:** [arXiv:2603.00461](https://arxiv.org/abs/2603.00461)
- 🤖 **Model:** [ReMoT_model on ModelScope](https://www.modelscope.cn/models/Tonycreate1/ReMoT_model)
- 📂 **Dataset:** [ReMoT_data on ModelScope](https://www.modelscope.cn/datasets/Tonycreate1/ReMoT_data)

------

## Overview

### Motivation

![image-20260708175107544](C:\Users\1\AppData\Roaming\Typora\typora-user-images\image-20260708175107544.png)

### Overview of the Triplet Motion Contrasts pipeline

![59de6c06-805e-45db-93d2-6046bb1f7836](D:\wechat_files\xwechat_files\wxid_sfmm5vt3c1th22_35b2\temp\InputTemp\59de6c06-805e-45db-93d2-6046bb1f7836.png)

### Benchmarks

![feaf9c6a-bad7-4dd2-9245-475643140809](D:\wechat_files\xwechat_files\wxid_sfmm5vt3c1th22_35b2\temp\InputTemp\feaf9c6a-bad7-4dd2-9245-475643140809.png)

![3b964ada-65e3-4883-b995-cd0ca5320e18](D:\wechat_files\xwechat_files\wxid_sfmm5vt3c1th22_35b2\temp\InputTemp\3b964ada-65e3-4883-b995-cd0ca5320e18.png)

## 📊 Dataset Description

### Data Format

All datasets use JSONL format, with each data entry containing the following fields:

| Field | Type | Description |
|------|------|------|
| `id` | string | Unique data identifier |
| `images` | list | List of image paths (supports multi-image input) |
| `messages` | list | List of conversation messages |
| `solution` | string | Ground truth answer in format `<answer>ABC</answer>` |

### Sample Data

```json
{
  "images": [
    "/data/guozeyu/DATA/agibot/resize_images_4/anchor/a_grab_release_656686_000150_35883.png",
    "/data/guozeyu/DATA/agibot/resize_images_4/positive/p_grab_release_656686_000508_35883.png",
    "/data/guozeyu/DATA/agibot/resize_images_4/negative/n_grab_release_656686_000039_35883.png"
  ],
  "messages": [{
    "role": "user",
    "content": "The image showed to you is what the robot seen by its eyes.\nIn the image, the robotic arm on the left is the robot's left arm, and the robotic arm on the right is the robot's right arm.\nFocus only on robot arm/gripper motion across the three images.\nPlease select from the following options whether the left gripper is opened or closed from Image 1 to Image 2? A: Opened, B: No movement, C: Closed.\nPlease select from the following options whether the left gripper is opened or closed from Image 1 to Image 3? A: Opened, B: No movement, C: Closed.\nPlease select from the following options whether the left gripper is opened or closed from Image 2 to Image 3? A: Closed, B: No movement, C: Opened.\nAnswer all three questions above in order. Only return the correct option A, B,or C for each of the three questions in order inside <answer></answer>, e.g., <answer>CAB</answer>"
  }],
  "solution": "<answer>CAC</answer>",
  "id": "idx_resize_4_test_0"
}
```

### Dataset Details

| Dataset | Task Type | Description |
|--------|----------|------|
| **agibot** | Robot gripper and arm state recognition | Determine gripper and arm state changes between different frames |
| **scannet** | Camera viewpoint change recognition | Determine camera viewpoint changes between different frames |
| **generaldata** | General visual reasoning | Includes relative position (mm), object detection (om), counting (vl), and other tasks |

### Image Path Tracing Instructions

Since only JSONL files are open-sourced without direct image files, original images can be traced based on filename conventions. Below are the image path naming conventions for each dataset.

#### 1. **Agibot**

- Example filename: `agibot/resize_images_4/anchor/a_grab_release_656686_000150_35883.png`
- Filename structure:
- `a_move_complex`: Task type (e.g., `grab_release`, `move_complex`, etc.)
- `652288`: Scene or task number in the Agibot dataset
- `000450`: Video frame number
<!-- - `8184`: Internal index or unique image ID -->

Through these fields, the image can be traced back to a specific scene and frame sequence in the Agibot dataset.

#### 2. **ScanNet**

- Example filename:
`scannet/crops3/scene0086_02_000300_A.png`
- Filename structure:
- `scene0086_02`: Scene number `scene0086_02` from the ScanNet dataset
- `crops3`: Indicates the 3rd crop version
- `000300`: Frame number
- `A`: Original frame image
- `B`: Cropped image simulating camera movement in the opposite direction

With these identifiers, corresponding images can be located in the original ScanNet dataset using path mapping.

---

## 🚀 Evaluation Code Usage

### 1. Single Dataset Evaluation

Use `index_eval.py` for multi-GPU parallel evaluation of a single dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_codes/index_eval.py \
    --model /path/to/your/model \
    --input /data1/guozeyu/DATA/CVPR2026/data_upload/agibot/test.jsonl \
    --output /path/to/output/result.jsonl \
    --check /path/to/checkpoint.jsonl \
    --max-new-tokens 10000 \
    --num-gpus 8
```

### 2. Batch Evaluation

Use `infer_remot_bench.py` to evaluate all datasets at once:

```bash
# First modify configuration parameters in the script (MODEL_PATH, CKPT_TAG, CHECK_FILE, etc.)
python eval_codes/infer_remot_bench.py
```

### 3. Result Aggregation

Use `summary.py` to generate evaluation reports:

```bash
# Modify configuration parameters in the script (BASE_FOLDER, MODEL_NAME, etc.) before running
python eval_codes/summary.py
```

### ⚙️ Parameter Description

#### index_eval.py Parameters

| Parameter | Type | Required | Description |
|------|------|------|------|
| `--model` | string | ✅ | HuggingFace model path or name |
| `--input` | string | ✅ | Input test set JSONL file path |
| `--output` | string | ✅ | Output result file path |
| `--check` | string | ✅ | Checkpoint file path (for resuming from breakpoint; can use any placeholder path for first run) |
| `--max-new-tokens` | int | ❌ | Maximum number of generated tokens, default 128 |
| `--num-gpus` | int | ❌ | Number of GPUs, default 8 |

#### infer_remot_bench.py Configuration

Modify the following configurations at the beginning of the script:

| Configuration | Description |
|--------|------|
| `MODEL_PATH` | Model path |
| `CKPT_TAG` | Output filename tag |
| `CHECK_FILE` | Checkpoint file path |
| `BASE_OUTPUT_DIR` | Output directory |
| `GPUS` | GPU list, e.g., `"0,1,2,3"` |
| `MAX_NEW_TOKENS` | Maximum number of generated tokens |
| `NUM_GPUS` | Number of GPUs |

#### summary.py Configuration

| Configuration | Description |
|--------|------|
| `AGIBOT_INDEX_PATH` | AGIBOT dataset index path |
| `GENERALDATA_INDEX_PATH` | GeneralData dataset index path |
| `BASE_FOLDER` | Evaluation results directory |
| `MODEL_NAME` | Model name (for matching result files) |
| `TASK_NAMES` | Task list `["agibot", "generaldata", "scannet"]` |

### 📈 Evaluation Metrics

#### Metric Descriptions

| Metric | Description |
|------|------|
| **Exact Match (Ov)** | Exact match accuracy, prediction completely matches ground truth |
| **Per-Char Match (Par)** | Character-level match accuracy, calculates correct ratio at each position |
| **Avg Response Words** | Average number of response words |

#### Dataset-Specific Metrics

##### AGIBOT
- **Gripper-Move**: Gripper movement tasks (move_lr + lift_lower)
- **Gripper-State**: Gripper state tasks (grab_release)
- **Composite**: Composite tasks (move_complex)

##### ScanNet

- **Camera Ov**: Camera viewpoint exact match (average of partial matches across viewpoints)
- **Camera Par**: Camera viewpoint character-level match

##### GeneralData
- **Rel-Pos**: Relative position tasks (mm class)
- **Grounding**: Object detection tasks (om class)
- **Counting**: Counting tasks (vl class)

### 🔄 Resume from Checkpoint

The evaluation script supports resuming from checkpoint:

1. When evaluation is interrupted, completed results are saved in temporary files (`{output}_gpu{id}.jsonl`)
2. When rerunning, specify the previous output file using the `--check` parameter
3. The script automatically skips processed data and continues from the checkpoint

### 📝 Output Format

#### Inference Results (JSONL)

```json
{
  "id": "idx_resize_4_test_0",
  "gpu_id": 0,
  "response_word_count": 45,
  "model_response": "Based on image analysis...<answer>CAC</answer>",
  "pred_answer": "CAC",
  "solution": "<answer>CAC</answer>",
  "gt_answer": "CAC",
  "correct_count": 3
}
```

#### Summary Report (Excel)

The generated Excel file contains two worksheets:

| Worksheet | Content |
|--------|------|
| `Summary_Indicators` | Core metric summary for each task (Sheet1) |
| `Detailed_Metrics` | Detailed classification accuracy |

### 📖 Usage Workflow

- Prepare model → Download or train a vision-language model
- Run batch evaluation → python infer_remot_bench.py
- Generate summary report → python summary.py
- View results → Open the generated Excel file