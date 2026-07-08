import json
import os
from typing import List, Dict
import pandas as pd

# =========================================================
# User Configuration
# =========================================================
AGIBOT_INDEX_PATH = "/data1/guozeyu/DATA/CVPR2026/data_upload/agibot/test.jsonl"
GENERALDATA_INDEX_PATH = "/data1/guozeyu/DATA/CVPR2026/data_upload/generaldata/test.jsonl"

BASE_FOLDER = "/data3/guozeyu/codes/pami_eval_result/remot_bench/base_cross4epoch2/"
MODEL_NAME = "ckpt_agibot_scannet_grpo"
TASK_NAMES = ["agibot", "generaldata", "scannet"]
SCANNET_INDICES_LIST = [[0, 4, 8], [2, 6, 10], [1, 5, 9], [3, 7, 11]]

# =========================================================
# Utility Functions
# =========================================================
def per_sample_accuracy(pred: str, gt: str) -> float:
    if not isinstance(pred, str) or not isinstance(gt, str):
        return 0.0
    return 1.0 if pred.strip() == gt.strip() else 0.0

def per_char_accuracy(pred: str, gt: str) -> float:
    if not isinstance(pred, str) or not isinstance(gt, str):
        return 0.0
    pred, gt = pred.strip(), gt.strip()
    if len(gt) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(pred, gt) if p == g)
    return correct / len(gt)

def scannet_partial_match(pred: str, gt: str, indices: List[int]) -> float:
    if not isinstance(pred, str) or not isinstance(gt, str):
        return 0.0
    pred_chars = "".join(pred[i] for i in indices if i < len(pred))
    gt_chars = "".join(gt[i] for i in indices if i < len(gt))
    return 1.0 if pred_chars == gt_chars else 0.0

def count_words(text: str) -> int:
    return len(text.strip().split()) if isinstance(text, str) else 0

# =========================================================
# Index Mapping Loaders
# =========================================================
def load_agibot_index(path: str) -> Dict[str, str]:
    VALID_CLASSES = {"grab_release", "lift_lower", "move_complex", "move_lr"}
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                id_ = obj.get("id")
                imgs = obj.get("images", [])
                if not imgs:
                    continue
                base = os.path.basename(imgs[0])
                for c in VALID_CLASSES:
                    if c in base:
                        mapping[id_] = c
                        break
            except:
                continue
    return mapping

def load_generaldata_index(path: str) -> Dict[str, str]:
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                id_ = obj["id"]
                imgs = obj["images"]
                parts = imgs[0].split("/")
                if "generaldata_resize" in parts and "images" in parts:
                    idx1, idx2 = parts.index("generaldata_resize"), parts.index("images")
                    cls = parts[idx1 + 1] if idx2 - idx1 >= 2 else "unknown"
                else:
                    cls = "unknown"
                mapping[id_] = cls
            except:
                continue
    return mapping

# =========================================================
# Core Evaluation Functions
# =========================================================
def evaluate(data, func, mapping=None, indices=None):
    mapping = mapping or {}
    totals, per_cls = 0, {}
    for obj in data:
        pred, gt = str(obj.get("pred_answer", "")), str(obj.get("gt_answer", ""))
        acc = func(pred, gt) if indices is None else func(pred, gt, indices)
        totals += acc
        if mapping:
            cls = mapping.get(obj.get("id", ""), "unknown")
            stat = per_cls.setdefault(cls, {"n": 0, "acc": 0})
            stat["n"] += 1
            stat["acc"] += acc
    n = len(data)
    overall = totals / n if n else 0
    by_cls = {c: (v["acc"] / v["n"] if v["n"] else 0) for c, v in per_cls.items()}
    return overall, by_cls

def calculate_avg_words(data) -> float:
    words, count = 0, 0
    for obj in data:
        resp = obj.get("model_response", "")
        if resp:
            words += count_words(resp)
            count += 1
    return words / count if count else 0.0

def process_jsonl(path: str):
    name = os.path.basename(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "pred_answer" in obj:
                    data.append(obj)
            except:
                continue
    if not data:
        return None

    task_name = os.path.basename(os.path.dirname(path))
    is_agibot, is_general, is_scannet = (task_name == "agibot"), (task_name == "generaldata"), (task_name == "scannet")

    if is_agibot:
        mapping = load_agibot_index(AGIBOT_INDEX_PATH)
    elif is_general:
        mapping = load_generaldata_index(GENERALDATA_INDEX_PATH)
    else:
        mapping = {}

    results = []
    avg_words = calculate_avg_words(data)
    results.append(("Avg Response Words", avg_words, {}))
    overall, by_cls = evaluate(data, per_sample_accuracy, mapping)
    results.append(("Exact Match", overall, by_cls))
    if is_scannet:
        partial_accs = []
        for idxs in SCANNET_INDICES_LIST:
            o, _ = evaluate(data, scannet_partial_match, mapping, indices=idxs)
            results.append((f"Partial {idxs}", o, {}))
            partial_accs.append(o)
        if partial_accs:
            avg_partial = sum(partial_accs) / len(partial_accs)
            results.append(("Avg Partial Match", avg_partial, {}))
        o, _ = evaluate(data, per_char_accuracy, mapping)
        results.append(("Per-Char Match", o, {}))
    else:
        o, by_cls = evaluate(data, per_char_accuracy, mapping)
        results.append(("Per-Char Match", o, by_cls))
    return task_name, results

# =========================================================
# Custom Metric Computation
# =========================================================
def compute_custom_metrics(task: str, results):
    rows = []
    metric_dict = {mode: (overall, by_cls) for mode, overall, by_cls in results}

    if task == "scannet":
        partial_vals = [overall for mode, overall, _ in results if mode.startswith("Partial [")]
        camera_ov = sum(partial_vals) / len(partial_vals) if partial_vals else 0
        camera_par = metric_dict.get("Per-Char Match", (0, {}))[0]
        rows.append({"Task": task, "Indicator": "Camera Ov", "Accuracy": round(camera_ov, 4)})
        rows.append({"Task": task, "Indicator": "Camera Par", "Accuracy": round(camera_par, 4)})

    elif task == "generaldata":
        em = metric_dict.get("Exact Match", (0, {}))[1]
        char = metric_dict.get("Per-Char Match", (0, {}))[1]

        def max_acc(cls):
            return max(em.get(cls, 0), char.get(cls, 0))

        rows += [
            {"Task": task, "Indicator": "Rel-Pos Ov", "Accuracy": round(max_acc("mm"), 4)},
            {"Task": task, "Indicator": "Grounding Ov", "Accuracy": round(max_acc("om"), 4)},
            {"Task": task, "Indicator": "Counting Ov", "Accuracy": round(max_acc("vl"), 4)},
        ]

    elif task == "agibot":
        em = metric_dict.get("Exact Match", (0, {}))[1]
        char = metric_dict.get("Per-Char Match", (0, {}))[1]
        gm_ov = (em.get("move_lr", 0) + em.get("lift_lower", 0)) / 2
        gm_par = (char.get("move_lr", 0) + char.get("lift_lower", 0)) / 2
        gs_ov = em.get("grab_release", 0)
        gs_par = char.get("grab_release", 0)
        comp_ov = em.get("move_complex", 0)
        comp_par = char.get("move_complex", 0)
        rows += [
            {"Task": task, "Indicator": "Gripper-Move Ov", "Accuracy": round(gm_ov, 4)},
            {"Task": task, "Indicator": "Gripper-Move Par", "Accuracy": round(gm_par, 4)},
            {"Task": task, "Indicator": "Gripper-State Ov", "Accuracy": round(gs_ov, 4)},
            {"Task": task, "Indicator": "Gripper-State Par", "Accuracy": round(gs_par, 4)},
            {"Task": task, "Indicator": "Composite Ov", "Accuracy": round(comp_ov, 4)},
            {"Task": task, "Indicator": "Composite Par", "Accuracy": round(comp_par, 4)},
        ]
    return rows

# =========================================================
# Main Routine
# =========================================================
def main():
    jsonl_paths = []
    for task in TASK_NAMES:
        path = os.path.join(BASE_FOLDER, task, f"{MODEL_NAME}.jsonl")
        if os.path.exists(path):
            jsonl_paths.append((task, path))
        else:
            print(f"[Warn] File not found: {path}")

    if not jsonl_paths:
        print("[Error] No valid JSONL files found.")
        return

    print("\nProcessing the following files:")
    for task, p in jsonl_paths:
        print(f"  - ({task}) {p}")

    output_excel_path = os.path.join(BASE_FOLDER, f"eval_{MODEL_NAME}.xlsx")
    writer = pd.ExcelWriter(output_excel_path, engine="openpyxl")

    all_rows = []
    summary_rows = []
    mode_acc_for_avg = {"Exact Match": [], "Per-Char Match": []}
    total_avg_words = []

    # Process each task file
    for task, path in jsonl_paths:
        print(f"\nProcessing task: {task}")
        result = process_jsonl(path)
        if not result:
            print(f"  [Skip] No valid data: {path}")
            continue

        _, results = result

        for mode, overall, by_cls in results:
            if mode == "Avg Response Words":
                all_rows.append({
                    "File": task,
                    "Mode": mode,
                    "Class": "Overall",
                    "Accuracy": round(overall, 2)
                })
                total_avg_words.append(overall)
                continue

            all_rows.append({
                "File": task,
                "Mode": mode,
                "Class": "Overall",
                "Accuracy": round(overall, 4)
            })
            for cls, acc in by_cls.items():
                all_rows.append({
                    "File": task,
                    "Mode": mode,
                    "Class": cls,
                    "Accuracy": round(acc, 4)
                })

            if mode in mode_acc_for_avg:
                mode_acc_for_avg[mode].append(overall)

        summary_rows.extend(compute_custom_metrics(task, results))

    # Compute overall averages
    avg_ov = None
    avg_par = None
    if mode_acc_for_avg["Exact Match"]:
        avg_ov = sum(mode_acc_for_avg["Exact Match"]) / len(mode_acc_for_avg["Exact Match"])
    if mode_acc_for_avg["Per-Char Match"]:
        avg_par = sum(mode_acc_for_avg["Per-Char Match"]) / len(mode_acc_for_avg["Per-Char Match"])

    # Create global rows for the summary sheet (Avg Ov and Avg Par)
    if avg_ov is not None:
        summary_rows.append({"Task": "Avg", "Indicator": "Avg. Ov.", "Accuracy": round(avg_ov, 4)})
    if avg_par is not None:
        summary_rows.append({"Task": "Avg", "Indicator": "Avg. Par.", "Accuracy": round(avg_par, 4)})

    # Prepare final DataFrames
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_summary = pd.DataFrame(summary_rows)
        # Summary indicators as Sheet1
        df_summary.to_excel(writer, sheet_name="Summary_Indicators", index=False)
        df_all.to_excel(writer, sheet_name="Detailed_Metrics", index=False)
        print("\nSuccessfully generated two sheets: Summary_Indicators (Sheet1) and Detailed_Metrics.")

    writer.close()
    print(f"\nResults have been saved to: {output_excel_path}")


if __name__ == "__main__":
    main()