import os
import sys
from pathlib import Path
from datetime import datetime
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --------------- LOAD DATA --------------- #
import pandas as pd
import base64
from io import BytesIO
from PIL import Image as PILImage
from collections import defaultdict

RESULTS_SAVE_DIR = Path(
    os.environ.get(
        "RESULT_SAVE_PATH",
        "/home/yangjiacheng/data/jiarui/GRPO/EVAL-all/TreeBench",
    )
)
treebench_tsv_path = "/data1/yangjiacheng/TreeBench/TreeBench.tsv"

trained_model_id = os.environ.get(
    "TRAINED_MODEL_ID",
    "/home/yangjiacheng/data/jiarui/HARI-75retry/HARI/checkpoint-30000",
)
EVAL_VERBOSE = os.environ.get("EVAL_VERBOSE", "0") == "1"
TREEBENCH_RUN_NAME = os.environ.get("TREEBENCH_RUN_NAME")

def base64_to_PIL(base64_string):
    """Decode base64 image string to PIL Image object"""
    image_data = base64.b64decode(base64_string)
    image = PILImage.open(BytesIO(image_data))
    return image

def parse_target_instances(target_instances_str):
    """Parse target_instances from string to proper data structure"""  
    if not target_instances_str or target_instances_str == '' or target_instances_str == 'nan':
        return None
    
    try:
        # Handle string representation of list/dict
        if isinstance(target_instances_str, str):
            import ast
            return ast.literal_eval(target_instances_str)
        else:
            return target_instances_str
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse target_instances: {target_instances_str}, error: {e}")
        return None

def load_treebench_dataset():
    """Load and process TreeBench TSV dataset"""
    df = pd.read_csv(treebench_tsv_path, sep='\t')
    processed_data = []
    for _, row in df.iterrows():
        # Decode base64 image to PIL Image
        image = base64_to_PIL(row['image'])
        
        processed_data.append({
            'index': row['index'],
            'image': image,
            'question': row['question'],
            'answer': row['answer'],
            'multi_choice_options': row['multi-choice options'],
            'category': row['category'],
            'l2_category': row['l2-category'],
            'target_instances': parse_target_instances(row['target_instances'])
        })
    return processed_data

print("Loading TreeBench dataset...")
test_dataset = load_treebench_dataset()
print(f"Loaded {len(test_dataset)} samples")
print(test_dataset[0])

# --------------- LOAD processor --------------- #

# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import torch
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=trained_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# trained_model_id="/home/yangjiacheng/data/jiarui/GRPO/TRAIN/GRPO-more/checkpoint-48000"
print(f"trained model id :{trained_model_id}")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     trained_model_id,
#     torch_dtype="auto",
#     device_map="auto",
# )


# processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

# --------------- conversation --------------- #
import re
import torch
import json
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
from string import ascii_uppercase
# from utils.dataset_image_process import smart_resize

# Image processing parameters
image_factor = 28
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

def parse_lettered_options(options_str, max_options=6):
    """
    Parse line-by-line lettered options like:
    A. text
    B) text
    C: text
    Returns (parsed_list, count) where parsed_list is [{'label':'A','text':'...'}, ...]
    """
    if not options_str:
        return [], 0

    s = str(options_str).strip()
    if s.lower() == "nan" or s == "":
        return [], 0

    lines = [ln.strip() for ln in re.split(r"\r?\n", s) if ln.strip()]
    parsed = []
    line_re = re.compile(r"^([A-Z])\s*[\.\):]\s*(.*)$", flags=re.IGNORECASE)
    for ln in lines:
        m = line_re.match(ln)
        if m:
            label = m.group(1).upper()
            text = m.group(2).strip()
            # accept only A-F (or up to max_options)
            if label in ascii_uppercase[:max_options]:
                parsed.append({"label": label, "text": text})
    # Sort by label to ensure correct order
    parsed.sort(key=lambda x: ascii_uppercase.index(x["label"]))
    return parsed, len(parsed)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
    # "You are a bilingual visual reasoning assistant. " +
    # "For every sample: 1) read the question carefully, 2) inspect all provided images, " +
    # "3) reason through the evidence step by step in plain text, and 4) output exactly one final choice. "+
    # "Valid choices are the letters shown in the prompt (A, B, C, …). "+
    # "After your reasoning, write a single sentence: Final: \\boxed{X} where X is the chosen letter. "+
    "Never emit multiple boxed answers or leave the box empty. "
    # "If the question contains Chinese text, you may reason in Chinese before giving the final boxed letter."+
    # "If your confidence is low, make a random guess."
)
import math

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def generate_with_reasoning(question, image, multi_choice_options, category):
    # ------------------------ Turn 1 ------------------------ #
    # resize image
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    resized_image = image.resize((resized_width, resized_height))

    resized_image = image  # No resizing
    
    # prepare prompt
    if category == "Perception/OCR":
        problem = question + "\n"
    else:
        problem = question + " The choices are listed below:\n" + (multi_choice_options if multi_choice_options else "") + "\n"
    
    # check how many options are provided
    parsed_options, num_options = parse_lettered_options(multi_choice_options, max_options=12)
    if num_options == 4:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, or D) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
    elif num_options == 5:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
    elif num_options == 6:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, or F) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, or \\boxed{F}."
    elif num_options == 7:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, F, or G) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, \\boxed{F}, or \\boxed{G}."
    elif num_options == 8:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, F, G, or H) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, \\boxed{F}, \\boxed{G}, or \\boxed{H}."
    elif num_options == 9:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, F, G, H, or I) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, \\boxed{F}, \\boxed{G}, \\boxed{H}, or \\boxed{I}."
    elif num_options == 10:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, F, G, H, I, or J) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, \\boxed{F}, \\boxed{G}, \\boxed{H}, \\boxed{I}, or \\boxed{J}."
    elif num_options == 11:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, F, G, H, I, J, or K) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, \\boxed{F}, \\boxed{G}, \\boxed{H}, \\boxed{I}, \\boxed{J}, or \\boxed{K}."
    elif num_options == 12:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, F, G, H, I, J, K, or L) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, \\boxed{F}, \\boxed{G}, \\boxed{H}, \\boxed{I}, \\boxed{J}, \\boxed{K}, or \\boxed{L}."
    else:
        option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}."
        
    # check how many options are provided
    # parsed_options, num_options = parse_lettered_options(multi_choice_options, max_options=6)
    # if num_options == 4:
    #     option_text = "Provide only the final answer (A, B, C, or D) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}. "
    # elif num_options == 5:
    #     option_text = "Provide only the final answer (A, B, C, D, or E) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}. "
    # else:
    #     option_text = "Provide only the final answer (A, B, C, D, E, or F) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, or \\boxed{F}. "
        

    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": "Question:" + " " + problem + " "
                + option_text
                    # "\n If your confidence is low, make a random guess."
                },
            ],
        },
    ]
    
    # Preparation for inference
    text_1 = processor.apply_chat_template(
        messages_1, tokenize=False, add_generation_prompt=True)
    
    if EVAL_VERBOSE:
        print("---text_1", text_1)
    
    # Vision packing
    image_inputs_1, video_inputs_1 = process_vision_info(messages_1)
    inputs_1 = processor(
        text=[text_1],
        images=image_inputs_1,
        videos=video_inputs_1,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs_1, max_new_tokens=500)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_1.input_ids, generated_ids)
        ]
    
    output_text_1 = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if EVAL_VERBOSE:
        print("===output_text_1", output_text_1)
    
    # ------------------------ Parse output ------------------------ #
    m = None
    m2 = re.search(r"\\boxed{([A-F])}", output_text_1[0], flags=re.IGNORECASE)
    if m2:
        m = m2.group(1).upper()
    
    return m

# --------------- EVALUATION --------------- #


# RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_name = Path(trained_model_id).name or "model"
# bench_name = "treebench"
# run_dir_name = TREEBENCH_RUN_NAME or f"{model_name}-{bench_name}-{timestamp}"
# run_dir = RESULTS_SAVE_DIR / run_dir_name
# run_dir.mkdir(parents=True, exist_ok=True)

# print(f"Results will be saved to: {run_dir}")

stats = defaultdict(lambda: {"count": 0, "correct": 0})

for i in range(len(test_dataset)):
    if EVAL_VERBOSE:
        print(f"\n{'='*50}")
        print(f"Processing {i+1}/{len(test_dataset)}")
        print(f"{'='*50}")

    generated_answer = generate_with_reasoning(
        test_dataset[i]["question"],
        test_dataset[i]["image"],
        test_dataset[i]["multi_choice_options"],
        test_dataset[i]["category"],
    )

    gt = test_dataset[i]["answer"]
    category = test_dataset[i]["category"]
    l2_category = test_dataset[i]["l2_category"]

    key = f"{category}-{l2_category}" if l2_category else category

    if EVAL_VERBOSE:
        print(f"\nGenerated: {generated_answer}, Ground Truth: {gt}")

    stats[key]["count"] += 1
    if generated_answer and generated_answer == gt:
        stats[key]["correct"] += 1
        if EVAL_VERBOSE:
            print("✓ Correct")
    else:
        if EVAL_VERBOSE:
            print("✗ Incorrect")

# --------------- PRINT RESULTS --------------- #
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count if count > 0 else 0
    print(f"{key}: total={count}, correct={correct}, accuracy={acc:.2%}")

# --------------- SAVE RESULTS --------------- #
max_json_path = Path("/home/yangjiacheng/data/jiarui/EVAL/max-tree.json")
with max_json_path.open(encoding="utf-8") as f:
    max_scores = json.load(f)

result = {}
outperform_count = 0
for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count if count > 0 else 0
    baseline_entry = max_scores.get(key, {})
    baseline_acc = baseline_entry.get("accuracy", 0.0)
    outperform = acc >= baseline_acc
    if outperform:
        outperform_count += 1
    result[key] = {"total": count, "correct": correct, "accuracy": f"{acc:.2%}", "outperform_max": outperform}

total_correct = sum(val["correct"] for val in stats.values())
total_count = sum(val["count"] for val in stats.values())
average_accuracy = total_correct / total_count if total_count > 0 else 0

print("=" * 50)
print(f"OVERALL: Outperform count = {outperform_count}, Total={total_count}, Correct={total_correct}, Accuracy={average_accuracy:.2%}")
print("=" * 50)

def get_tree_perception_reasoning(data):
    perception_keys = [
        "Perception/Attributes-Perception/Attributes",
        "Perception/OCR-Perception/OCR",
        "Perception/Physical State-Perception/Physical State",
        "Perception/Object Retrieval-Perception/Object Retrieval",
        "Perception/Material-Perception/Material"
    ]
    reasoning_keys = [
        "Reasoning/Comparison-Reasoning/Comparison",
        "Reasoning/Ordering-Reasoning/Ordering",
        "Reasoning/Contact and Occlusion-Reasoning/Contact and Occlusion",
        "Reasoning/Spatial Containment-Reasoning/Spatial Containment",
        "Reasoning/Perspective Transform-Reasoning/Perspective Transform"
    ]
    def calc(keys):
        total = sum(data[k]["total"] for k in keys if k in data)
        correct = sum(data[k]["correct"] for k in keys if k in data)
        return correct / total * 100 if total > 0 else None
    return calc(perception_keys), calc(reasoning_keys)



result["Overall"] = {
    "outperform_count": outperform_count,
    "total_samples": total_count,
    "total_correct": total_correct,
    "overall_accuracy": f"{average_accuracy:.2%}",
}


perception_acc, reasoning_acc = get_tree_perception_reasoning(result)
result["Overall"]["perception_accuracy"] = perception_acc
result["Overall"]["reasoning_accuracy"] = reasoning_acc

ckpt_name = Path(trained_model_id).name or "model"
step_part = ckpt_name.split("-")[-1] if "checkpoint-" in ckpt_name else "unknown"

os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
summary_path = RESULTS_SAVE_DIR/f"Tree-checkpoint-{step_part}-{average_accuracy:.2%}-{outperform_count}.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"\nResults saved to: {summary_path}")    
