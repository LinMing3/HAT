import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Allow overriding from bash without editing this file.
trained_model_id = os.environ.get(
    "TRAINED_MODEL_ID",
    "/home/yangjiacheng/data/jiarui/HARI-sampling/HARI/checkpoint-19000",
)

print(f"Using trained_model_id: {trained_model_id}")
# RESULT_JSON_PATH = os.environ.get("MME_RESULT_JSON_PATH", "result.json")
EVAL_VERBOSE = os.environ.get("EVAL_VERBOSE", "0") == "1"


ckpt_name = os.path.basename(trained_model_id.rstrip("/"))
step_part = ckpt_name.split("-")[-1] if "checkpoint-" in ckpt_name else "unknown"
print(f"checkpoint name: {ckpt_name}, step part: {step_part}")


result_save_path = os.environ.get(
    "RESULT_SAVE_PATH",
    "/home/yangjiacheng/data/jiarui/HARI-random/EVAL-all/MME",
)

# --------------- LOAD DATA --------------- #
import datasets
from collections import defaultdict
from datasets import load_dataset, Image

MME_IMAGE_ROOT = os.environ.get(
    "MME_IMAGE_ROOT",
    "/data1/yangjiacheng/MME-RealWorld-Lite/imgs",
)
MME_JSON_PATH = os.environ.get(
    "MME_JSON_PATH",
    "/data1/yangjiacheng/MME-RealWorld-Lite/MME-RealWorld-Lite.json",
)
MME_SKIP_MISSING = os.environ.get("MME_SKIP_MISSING", "0") == "1"

def add_full_path(example):
    example["Image"] = os.path.join(image_root, example["Image"])
    example["ImagePath"] =  example["Image"]
    return example

def image_exists(example):
    return os.path.exists(example["Image"])

def process_dataset(dataset_split):
    dataset_split =  dataset_split.map(
        add_full_path, num_proc=32)
    if MME_SKIP_MISSING:
        dataset_split = dataset_split.filter(image_exists, num_proc=32)
    dataset_split = dataset_split.cast_column("Image", Image(decode=True))
    return dataset_split

image_root = MME_IMAGE_ROOT
if not os.path.isdir(image_root):
    raise FileNotFoundError(
        f"MME image_root not found: {image_root}. "
        f"Set env MME_IMAGE_ROOT to the correct directory."
    )
dataset = load_dataset(
    "json", 
    data_files=MME_JSON_PATH
    )
dataset = dataset['train']

test_dataset = process_dataset(dataset)
print(test_dataset)
print(test_dataset[0])

# --------------- LOAD processor --------------- #

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     trained_model_id,
#     torch_dtype="auto",
#     device_map="auto",
# )


from peft import LoraConfig, get_peft_model
import torch

from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=trained_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

# --------------- conversation --------------- #
import re
import torch
try:
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError:
    def process_vision_info(messages):
        images, videos = [], []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image" and "image" in part:
                    images.append(part["image"])
                if part.get("type") == "video" and "video" in part:
                    videos.append(part["video"])
        return images, videos

SYSTEM_PROMPT = (
   "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
)

def generate_with_reasoning(problem, image, answer_choices):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Question:" + " " + problem + " "  +\
                "The choices are listed below:" + " " + ' '.join(answer_choices) + " " +\
                "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
                },
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    if EVAL_VERBOSE:
        print("Prompt:", prompt)
    
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
    output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    if EVAL_VERBOSE:
        print("output_text", output_text)
    
    new_ansewr = re.search(r'\\boxed\{([A-E])\}', output_text[0])
    if EVAL_VERBOSE:
        print("Extracted answer:", new_ansewr, new_ansewr.group(1) if new_ansewr else None)
    if new_ansewr:
        return new_ansewr.group(1)
    else:
        return None
    
stats = defaultdict(lambda: {"count": 0, "correct": 0})
for i in range(len(test_dataset)):  
    print(i, "-", len(test_dataset))
    generated_text= generate_with_reasoning(
        test_dataset[i]["Text"], 
        test_dataset[i]["Image"],
        test_dataset[i]["Answer choices"]
        )
    gt = test_dataset[i]["Ground truth"]
    task = test_dataset[i]["Task"]
    subtask = test_dataset[i]["Subtask"]

    key = f"{task}-{subtask}"
    print(generated_text, gt)
    stats[key]["count"] += 1
    if generated_text and generated_text == gt:
        stats[key]["correct"] += 1
        print("correct")
        

for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count
    print(f"{key}: total={count}, correct={correct}, accuracy={acc:.2%}")
    
import json
from pathlib import Path

max_json_path = Path("/home/yangjiacheng/data/jiarui/EVAL/max-mme.json")
with max_json_path.open(encoding="utf-8") as f:
    max_scores = json.load(f)

# 计算perception和reasoning加权准确率
def get_mme_perception_reasoning(data):
    perception_keys = [
        "Perception-Remote Sensing",
        "Perception-Monitoring",
        "Perception-Autonomous_Driving",
        "Perception-OCR with Complex Context",
        "Perception-Diagram and Table"
    ]
    reasoning_keys = [
        "Reasoning-Monitoring",
        "Reasoning-Autonomous_Driving",
        "Reasoning-OCR with Complex Context",
        "Reasoning-Diagram and Table"
    ]
    def calc(keys):
        total = sum(data[k]["total"] for k in keys if k in data)
        correct = sum(data[k]["correct"] for k in keys if k in data)
        return correct / total * 100 if total > 0 else None
    return calc(perception_keys), calc(reasoning_keys)

result = {}
outperform_count = 0
for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count
    baseline_entry = max_scores.get(key, {})
    baseline_acc = baseline_entry.get("accuracy", 0.0)
    outperform = acc >= baseline_acc
    if outperform:
        outperform_count += 1
    result[key] = {
        "total": count,
        "correct": correct,
        "accuracy": f"{acc:.2%}",
        "outperform_max": outperform
    }
    
total_correct = sum(val["correct"] for val in stats.values())
total_count = sum(val["count"] for val in stats.values())
average_reward = total_correct / total_count

print("=" * 50)
print(f"OVERALL: Outperform count = {outperform_count}, Total={total_count}, Correct={total_correct}, Accuracy={average_reward:.2%}")
print("=" * 50)

result["Average Reward"] = {
    "outperform_count": outperform_count,
    "total_samples": total_count,
    "overall_accuracy": f"{average_reward:.2%}"
}

# 计算perception和reasoning加权准确率并写入result
perception_acc, reasoning_acc = get_mme_perception_reasoning(result)
result["Average Reward"]["perception_accuracy"] = perception_acc
result["Average Reward"]["reasoning_accuracy"] = reasoning_acc


ckpt_name = os.path.basename(trained_model_id.rstrip("/"))
step_part = ckpt_name.split("-")[-1] if "checkpoint-" in ckpt_name else "unknown"

os.makedirs(result_save_path, exist_ok=True)
result_file = os.path.join(
    result_save_path,
    f"MME-checkpoint-{step_part}-{average_reward:.2%}-{outperform_count}.json"
)

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
