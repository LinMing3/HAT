import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Allow overriding from bash without editing this file.
trained_model_id = os.environ.get(
    "TRAINED_MODEL_ID",
    "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/HARI-halfretry-more/HARI-strict-reward/checkpoint-19000",
)
RESULT_JSON_PATH = os.environ.get("MME_RESULT_JSON_PATH", "result.json")
DETAIL_JSON_PATH = os.environ.get("MME_DETAIL_JSON_PATH", "output-HARI.json")
EVAL_VERBOSE = os.environ.get("EVAL_VERBOSE", "0") == "1"

# --------------- LOAD DATA --------------- #
import json
import datasets
from collections import defaultdict
from datasets import Image

MME_IMAGE_ROOT = os.environ.get(
    "MME_IMAGE_ROOT",
    "/data1/yunkai/VIG_Group/dataset/MME-RealWorld-Lite/imgs",
)
MME_JSON_PATH = os.environ.get(
    "MME_JSON_PATH",
    "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/visual/hari-test.json",
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
with open(MME_JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if isinstance(raw_data, dict):
    if "data" in raw_data and isinstance(raw_data["data"], list):
        raw_data = raw_data["data"]
    else:
        raise ValueError("JSON root object must contain a 'data' list or be a list of samples.")
elif not isinstance(raw_data, list):
    raise ValueError("Unsupported JSON structure: expected list of samples or {'data': [...]}.")

dataset = datasets.Dataset.from_list(raw_data)

test_dataset = process_dataset(dataset)
print(test_dataset)
print(test_dataset[0])

# --------------- LOAD processor --------------- #

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    trained_model_id,
    torch_dtype="auto",
    device_map="auto",
)
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
detailed_records = []
for i in range(len(test_dataset)):
    print(i, "-", len(test_dataset))
    sample = test_dataset[i]
    generated_text= generate_with_reasoning(
        sample["Text"], 
        sample["Image"],
        sample["Answer choices"]
        )
    gt = sample["Ground truth"]
    task = sample["Task"]
    subtask = sample["Subtask"]

    key = f"{task}-{subtask}"
    print(generated_text, gt)
    stats[key]["count"] += 1
    if generated_text and generated_text == gt:
        stats[key]["correct"] += 1
        print("correct")

    record = {
        "index": i,
        "task": task,
        "subtask": subtask,
        "question": sample["Text"],
        "answer_choices": sample["Answer choices"],
        "ground_truth": gt,
        "predicted_answer": generated_text,
        "is_correct": bool(generated_text and generated_text == gt),
        "image_path": sample.get("ImagePath")
    }
    if record["image_path"] is None:
        record["image_path"] = str(sample.get("Image"))
    detailed_records.append(record)
        

for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count
    print(f"{key}: total={count}, correct={correct}, accuracy={acc:.2%}")
    
result = {}
for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count
    result[key] = {
        "total": count,
        "correct": correct,
        "accuracy": f"{acc:.2%}"
    }
    
total_correct = sum(val["correct"] for val in stats.values())
total_count = sum(val["count"] for val in stats.values())
average_reward = total_correct / total_count
print(f"Average Reward (Overall Accuracy): {average_reward:.2%}")

result["Average Reward"] = {
    "total_samples": total_count,
    "overall_accuracy": f"{average_reward:.2%}"
}

acc_for_filename = f"{average_reward * 100:.2f}"
model_id_name = os.path.basename(os.path.normpath(trained_model_id))
output_filename = f"MME-{acc_for_filename}-{model_id_name}.json"

result_path_hint = os.path.expanduser(RESULT_JSON_PATH)
if result_path_hint.endswith(os.sep) or (
    os.path.isdir(result_path_hint) and os.path.splitext(result_path_hint)[1] == ""
):
    output_dir = result_path_hint.rstrip(os.sep) or "."
else:
    base, ext = os.path.splitext(result_path_hint)
    if ext:
        output_dir = os.path.dirname(result_path_hint) or "."
    else:
        output_dir = result_path_hint or "."

os.makedirs(output_dir, exist_ok=True)
final_result_path = os.path.join(output_dir, output_filename)

with open(final_result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"Saved MME results to {final_result_path}")

detail_output_path = os.path.expanduser(DETAIL_JSON_PATH)
detail_dir = os.path.dirname(detail_output_path) or "."
os.makedirs(detail_dir, exist_ok=True)
with open(detail_output_path, "w", encoding="utf-8") as f:
    json.dump(detailed_records, f, ensure_ascii=False, indent=4)

print(f"Saved per-question outputs to {detail_output_path}")
