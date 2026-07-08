import subprocess
import os

def run_command(cmd):
    """Execute a command and print output to the console"""
    print(f"\n🚀 Executing command: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print("✅ Task completed successfully.\n")
    else:
        print(f"❌ Command execution failed (return code {result.returncode}), stopped.")
        exit(result.returncode)


if __name__ == "__main__":
    # ==================================================
    # 👇 Modify only this section
    MODEL_PATH = "/data/gzy/model/grpo/v0-20251107-203923/checkpoint-85/"
    CKPT_TAG = "ckpt_grpo"     # Used for the output filename
    CHECK_FILE = "/data/321123.jsonl" # If a checkpoint file exists, specify its path; otherwise, use any non-existent path as a placeholder
    # ==================================================

    BASE_OUTPUT_DIR = "/data/gzy/eval_result/remot_bench/grpo/"

    # Dataset information (keys are subfolder names)
    DATASETS = {
        "agibot": "/data1/guozeyu/DATA/CVPR2026/data_upload/agibot/test.jsonl",
        "scannet": "/data1/guozeyu/DATA/CVPR2026/data_upload/scannet/test.jsonl",
        "generaldata": "/data1/guozeyu/DATA/CVPR2026/data_upload/generaldata/test.jsonl"
    }

    # GPU configuration
    GPUS = "0,1,2,3,4,5,6,7"
    MAX_NEW_TOKENS = 10000
    NUM_GPUS = 8

    # ==================================================
    # Automatically construct 3 commands based on the above configuration
    # ==================================================
    for name, input_path in DATASETS.items():
        # Output directory
        output_dir = os.path.join(BASE_OUTPUT_DIR, name)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{CKPT_TAG}.jsonl")

        cmd = f"""
        CUDA_VISIBLE_DEVICES={GPUS} python /data1/guozeyu/DATA/CVPR2026/data_upload/eval_codes/index_eval.py \
        --model {MODEL_PATH} \
        --check {CHECK_FILE} \
        --input {input_path} \
        --output {output_file} \
        --max-new-tokens {MAX_NEW_TOKENS} \
        --num-gpus {NUM_GPUS}
        """

        run_command(cmd.strip())

    print("🎯 All tasks have been executed successfully!")