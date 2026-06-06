import os
import sys
from pathlib import Path
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --------------- LOAD DATA --------------- #
import datasets
from collections import defaultdict
from datasets import Image

RESULTS_SAVE_DIR = Path("/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/Qwen")

def add_full_path(example):
    example["image"] = os.path.join(image_root, example["image"])
    return example

def resize_dataset(example):
    # No resizing applied
    image = example["image"]
    example["image"] = image
    return example

print("Loading V-STAR dataset...")
image_root = "/data1/yunkai/VIG_Group/dataset/vstar_bench"
dataset = datasets.load_dataset(
    "json", 
    data_files="/data1/yunkai/VIG_Group/dataset/vstar_bench/test_questions.jsonl",
    )
test_dataset = dataset["train"]

# Add full path to images and load them
test_dataset = test_dataset.map(add_full_path, writer_batch_size=100, batch_size=100)
test_dataset = test_dataset.cast_column("image", Image(decode=True))
test_dataset = test_dataset.map(resize_dataset, writer_batch_size = 100, batch_size =100)
print(f"Loaded {len(test_dataset)} samples")
print(test_dataset[0])

# --------------- LOAD processor --------------- #
# trained_model_id = "/home/dangyunkai/yunkai/VLM/VIG-Group/model/Qwen2.5-VL-7B-Instruct"
# trained_model_id = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiacheng/251014-HURT3/MGPO/MGPO"
trained_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    trained_model_id,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()  # Set model to evaluation mode for deterministic inference
processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

# --------------- conversation --------------- #
import re
import torch
import json
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    # "You are a helpful assistant. "
    # "You will be given an image and a question. "
    # "First, identify the key image area relevant to solving the problem. "
    # "Then, provide the final answer (A, B, C, or D)."
    "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
)

def generate_with_reasoning(problem, image):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Question:" + " " + problem + " " +\
                # "First, identify and output the coordinates of the key image area relevant to solving the problem. "
                # "Carefully analyze both the original image and the key image area to solve the question step by step. "
                # "Present your reasoning clearly, and provide the final answer (A, B, C, or D)."
                # "Provide only the final answer (A, B, C, or D)."
                "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
                },
            ],
        },
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    
    print("---text", text)
    
    # Vision packing
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("===output_text", output_text)
    
    # ------------------------ Parse output ------------------------ #
    m = None
    m2 = re.search(r"\b([A-D])\b", output_text[0], flags=re.IGNORECASE)
    if m2:
        m = m2.group(1).upper()
    
    return m

# --------------- EVALUATION --------------- #
RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = Path(trained_model_id).name or "model"
bench_name = "vstar"
run_dir_name = f"{model_name}-{bench_name}-{timestamp}"
run_dir = RESULTS_SAVE_DIR / run_dir_name
run_dir.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {run_dir}")

stats = defaultdict(lambda: {"count": 0, "correct": 0})

for i in range(len(test_dataset)):  
    print(f"\n{'='*50}")
    print(f"Processing {i+1}/{len(test_dataset)}")
    print(f"{'='*50}")
    
    generated_answer = generate_with_reasoning(
        test_dataset[i]["text"], 
        test_dataset[i]["image"]
    )
    
    gt = test_dataset[i]["label"]
    category = test_dataset[i]["category"]

    key = f"{category}"
    
    print(f"\nGenerated: {generated_answer}, Ground Truth: {gt}")
    
    stats[key]["count"] += 1
    if generated_answer and generated_answer == gt:
        stats[key]["correct"] += 1
        print("✓ Correct")
    else:
        print("✗ Incorrect")

# --------------- PRINT RESULTS --------------- #
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count if count > 0 else 0
    print(f"{key}: total={count}, correct={correct}, accuracy={acc:.2%}")

# --------------- SAVE RESULTS --------------- #
import json

result = {}
for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count if count > 0 else 0
    result[key] = {
        "total": count,
        "correct": correct,
        "accuracy": f"{acc:.2%}"
    }

total_correct = sum(val["correct"] for val in stats.values())
total_count = sum(val["count"] for val in stats.values())
average_accuracy = total_correct / total_count if total_count > 0 else 0

print("="*50)
print(f"OVERALL: Total={total_count}, Correct={total_correct}, Accuracy={average_accuracy:.2%}")
print("="*50)

result["Overall"] = {
    "total_samples": total_count,
    "total_correct": total_correct,
    "overall_accuracy": f"{average_accuracy:.2%}"
}

summary_path = run_dir / "summary_report.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"\nResults saved to: {summary_path}")