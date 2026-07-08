import os
import json
import re
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from multiprocessing import Process, Value, Lock

# 正则解析 <answer> 标签 或 </think> 后的文本
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_END_PATTERN = re.compile(r"</think>(.*)", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    """提取答案内容"""
    m = ANSWER_TAG_PATTERN.search(text or "")
    if m:
        return m.group(1).strip().upper()
    m2 = THINK_END_PATTERN.search(text or "")
    if m2:
        return m2.group(1).strip().upper()
    return ""


def build_messages(images: List[str], user_text: str) -> List[Dict[str, Any]]:
    """按多图多模态聊天模板构建 message"""
    content = []
    for p in images:
        abs_p = os.path.abspath(p)
        if not os.path.exists(abs_p):
            raise FileNotFoundError(f"Image not found: {p}")
        content.append({"type": "image", "image": abs_p})
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on JSONL testset using multi-GPU parallelism with resume support.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--check", required=True, type=str, help="Checkpoint JSONL file to determine resume position by last 'id'")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-gpus", type=int, default=8, help="How many GPUs to use for parallel inference")
    return parser.parse_args()


def count_correct_chars(pred: str, gt: str) -> int:
    """比对模型输出与标准答案中共有多少个相同选项。"""
    pred_set = set(pred.strip().upper())
    gt_set = set(gt.strip().upper())
    return len(pred_set & gt_set)


def count_words(text: str) -> int:
    """统计输出文本中的英文单词数量"""
    words = re.findall(r"\b\w+\b", text)
    return len(words)


def process_chunk(
    chunk_id: int,
    items: list,
    args: argparse.Namespace,
    model_name: str,
    progress_count: Value,
    progress_lock: Lock,
    total_items: int
):
    """单个进程工作函数 (每个 GPU 一份)"""
    torch.cuda.set_device(chunk_id)
    device = f"cuda:{chunk_id}"

    print(f"[GPU-{chunk_id}] Loading model on {device}")
    model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    output_file = f"{os.path.splitext(args.output)[0]}_gpu{chunk_id}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[GPU-{chunk_id}] Writing to {output_file}")

    total_words = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(total=len(items), desc=f"GPU {chunk_id}", position=chunk_id, leave=False)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in items:
            # 获取 id（用于后续输出）
            data_id = item.get("id", "unknown")

            images = item.get("images", [])
            messages_in = item.get("messages", [])
            solution = item.get("solution", "")

            if not images or not isinstance(images, list) or not messages_in:
                pbar.update(1)
                with progress_lock:
                    progress_count.value += 1
                continue

            # 提取 user_text
            user_text = ""
            for m in messages_in:
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str) and c.strip():
                        user_text = c.strip()
                        break
                    elif isinstance(c, list):
                        for seg in c:
                            if seg.get("type") == "text" and seg.get("text", "").strip():
                                user_text = seg["text"].strip()
                                break
                        if user_text:
                            break

            # 构建输入
            messages = build_messages(images, user_text)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # === 统计 ===
            resp_words = count_words(response_text)
            pred_answer = extract_answer(response_text)
            gt_answer = extract_answer(solution)
            correct_count = count_correct_chars(pred_answer, gt_answer)

            total_words += resp_words
            total_correct += correct_count
            total_samples += 1

            # ✅ 关键：加入 id 字段
            rec = {
                "id": data_id,  # 保留原始 id
                "gpu_id": chunk_id,
                "response_word_count": resp_words,
                "model_response": response_text,
                "pred_answer": pred_answer,
                "solution": solution,
                "gt_answer": gt_answer,
                "correct_count": correct_count,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # 更新进度
            pbar.update(1)
            with progress_lock:
                progress_count.value += 1

        pbar.close()

        summary = {
            "gpu_id": chunk_id,
            "summary": True,
            "num_samples": total_samples,
            "avg_words": (total_words / total_samples) if total_samples > 0 else 0,
            "avg_correct_options": (total_correct / total_samples) if total_samples > 0 else 0,
        }
        f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"[GPU-{chunk_id}] Done. Samples: {total_samples}")


def merge_outputs_short(num_gpus: int, output: str):
    """
    合并多个 GPU 的结果，并确保每个 GPU 贡献相同条数（取最少的那一份的行数）。
    保留交错顺序：gpu0[0], gpu1[0], ..., gpuN[0], gpu0[1], gpu1[1], ...
    过滤掉 summary 行。
    """
    base, ext = os.path.splitext(output)
    part_files = [f"{base}_gpu{i}{ext}" for i in range(num_gpus)]
    part_records: List[List[str]] = []

    # 1️⃣ 读取所有 GPU 文件
    for file_path in part_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件缺失：{file_path}，该 GPU 跳过。")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            # 过滤 summary 行
            lines = [line for line in lines if '"summary": true' not in line.lower()]
            part_records.append(lines)
            print(f"{os.path.basename(file_path)}: {len(lines)} 条有效记录")

    if not part_records:
        raise RuntimeError("没有找到有效的 GPU 结果文件！")

    # 2️⃣ 取最短长度，保证公平采样
    min_len = min(len(lines) for lines in part_records)
    total_output = min_len * len(part_records)
    print(f"🚀 每个 GPU 取前 {min_len} 条，共 {total_output} 条输出")

    # 3️⃣ 创建输出目录并写入交错顺序结果
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fout:
        for i in range(min_len):
            for lines in part_records:
                fout.write(lines[i] + "\n")

    print(f"✅ 合并完成，最终结果输出到：{output}")

def merge_outputs(num_gpus: int, output: str):
    """
    合并多个 GPU 的结果，但保留交错逻辑，
    不再以最短文件为准——用 zip_longest 填充。
    """
    import itertools

    base, ext = os.path.splitext(output)
    part_files = [f"{base}_gpu{i}{ext}" for i in range(num_gpus)]

    part_records = []
    for fp in part_files:
        if not os.path.exists(fp):
            print(f"⚠️ 文件缺失：{fp}，跳过。")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and '"summary": true' not in line.lower()]
            part_records.append(lines)
            print(f"{os.path.basename(fp)}: {len(lines)} 条有效记录")

    if not part_records:
        raise RuntimeError("🚨 未找到任何有效 GPU 文件！")

    # 使用 zip_longest 迭代所有行
    from itertools import zip_longest
    total_written = 0
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fout:
        for row_bundle in zip_longest(*part_records, fillvalue=None):
            for line in row_bundle:
                if line is not None:
                    fout.write(line + "\n")
                    total_written += 1

    print(f"✅ 合并完成，共写入 {total_written} 条。输出文件：{output}")


def split_data(data: List[Dict[str, Any]], num_parts: int) -> List[List[Dict[str, Any]]]:
    """将数据均匀分成 num_parts 份"""
    return [data[i::num_parts] for i in range(num_parts)]


def get_last_id_from_check(check_file: str) -> str:
    """
    从 check 文件中读取最后一行的有效数据（非 summary），提取其 id。
    如果文件不存在或为空，返回 None。
    """

    if not os.path.exists(check_file):
        print(f"🔍 Checkpoint 文件未找到: {check_file} → 将从头开始处理。")
        return None

    with open(check_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"🔍 Checkpoint 文件为空 → 将从头开始处理。")
        return None

    # 逆序查找第一个不是 summary 的记录
    for line in reversed(lines):
        if '"summary": true' not in line.lower():
            try:
                item = json.loads(line)
                last_id = item.get("id")
                if last_id is not None:
                    print(f"✅ 在 checkpoint 文件中找到最后处理的 id: {last_id}")
                    return last_id
            except Exception as e:
                continue

    print(f"🔍 未能从 checkpoint 文件中解析出有效 id → 将从头开始处理。")
    return None


import re


# 在 filter_input_data_by_id 函数中替换原有的 ID 解析逻辑
def extract_numeric_part(item_id: str) -> int:
    """
    从给定的 item_id 中提取出其数字部分，并返回为整数。
    支持 idx_scannet_test_3_2, idx_resize_4_test_0 等格式。
    """
    match = re.search(r'\d+', item_id[::-1])  # 反转字符串后查找最后一个出现的数字序列
    if match:
        return int(match.group()[::-1])  # 将找到的数字反转回来，并转换成整数
    else:
        raise ValueError(f"无法从 id {item_id} 中提取有效的数字部分")


def filter_input_data_by_id(data: List[Dict[str, Any]], last_id: str) -> List[Dict[str, Any]]:
    """
    从 input 数据中筛选出 id > last_id 的条目。
    使用 extract_numeric_part 提取数字部分进行比较。
    """
    if last_id is None:
        return data

    try:
        last_idx = extract_numeric_part(last_id)
    except ValueError as e:
        print(str(e))
        return data

    filtered = []
    for item in data:
        item_id = item.get("id")
        if item_id is None:
            filtered.append(item)
            continue
        try:
            item_idx = extract_numeric_part(item_id)
            if item_idx > last_idx:
                filtered.append(item)
        except ValueError:
            # 如果某个 id 无法解析，保守起见仍处理
            filtered.append(item)

    print(f"📋 原始数据 {len(data)} 条，从 id {last_id} 之后筛选出 {len(filtered)} 条待处理。")
    return filtered




def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    # 1. 读取 input 数据
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 2. 从 check 文件获取 last_id，并过滤 input 数据
    last_id = get_last_id_from_check(args.check)
    data = filter_input_data_by_id(data, last_id)

    if not data:
        print("🔚 所有数据均已处理完毕，无需继续执行。")
        return

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    chunks = split_data(data, num_gpus)
    total_items = len(data)

    print(f"总样本数 {total_items}，分配到 {num_gpus} 个 GPU")

    # ==== 共享进度条 ====
    progress_count = Value("i", 0)
    progress_lock = Lock()
    global_pbar = tqdm(total=total_items, desc="总进度", position=0)

    processes = []
    for gid in range(num_gpus):
        p = Process(
            target=process_chunk,
            args=(gid, chunks[gid], args, args.model, progress_count, progress_lock, total_items),
        )
        p.start()
        processes.append(p)

    # 主进程动态刷新进度
    last_val = 0
    while any(p.is_alive() for p in processes):
        with progress_lock:
            current = progress_count.value
        global_pbar.update(current - last_val)
        last_val = current
        torch.cuda._sleep(int(1e8))  # ~0.1s

    # 最后一刷
    with progress_lock:
        current = progress_count.value
    global_pbar.update(current - last_val)
    global_pbar.close()

    for p in processes:
        p.join()

    merge_outputs(num_gpus, args.output)
    print("✅ 全部完成！")


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # ✅ 关键修复点
    main()