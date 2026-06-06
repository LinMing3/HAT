# --------------- IMPORTS --------------- #
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re
import json
from datetime import datetime
from string import ascii_uppercase
from collections import defaultdict

import datasets
from datasets import Image
import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from dataset_image_process import smart_resize

image_factor = 28
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
max_pixels_sub = 780 * 28 * 28

results_save_dir = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/test"
# visualization_save_dir = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/test"
model_id = "OpenGVLab/InternVL3-8B"
model_path = "/home/dangyunkai/yunkai/VLM/VIG-Group/model/InternVL3-8B"
max_new_tokens = 500

MME_IMAGE_ROOT = os.environ.get(
    "MME_IMAGE_ROOT",
    "/data1/yunkai/VIG_Group/dataset/MME-RealWorld-Lite/imgs",
)
MME_JSON_PATH = os.environ.get(
    "MME_JSON_PATH",
    "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/test/hari-test.json",
)
MME_SKIP_MISSING = os.environ.get("MME_SKIP_MISSING", "0") == "1"
image_root = MME_IMAGE_ROOT

# InternVL specific parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
internvl_input_size = 448
internvl_max_num = 8

def add_full_path(example):
    image_path = os.path.join(image_root, example["Image"])
    example["ImagePath"] = image_path
    example["Image"] = image_path
    return example


def image_exists(example):
    return os.path.exists(example["Image"])


def process_dataset(dataset_split):
    dataset_split = dataset_split.map(add_full_path, num_proc=32)
    if MME_SKIP_MISSING:
        dataset_split = dataset_split.filter(image_exists, num_proc=32)
    dataset_split = dataset_split.cast_column("Image", Image(decode=True))
    return dataset_split


def load_mme_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        if "data" in raw_data and isinstance(raw_data["data"], list):
            return raw_data["data"]
        raise ValueError("JSON root object must contain a 'data' list or be a list of samples.")
    if isinstance(raw_data, list):
        return raw_data
    raise ValueError("Unsupported JSON structure: expected list of samples or {'data': [...]}.")


def load_mme_dataset():
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"MME image_root not found: {image_root}. "
            f"Set env MME_IMAGE_ROOT to the correct directory."
        )

    raw_data = load_mme_json(MME_JSON_PATH)
    dataset = datasets.Dataset.from_list(raw_data)
    dataset = process_dataset(dataset)
    print(dataset)
    print(dataset[0])
    return dataset

def build_transform(input_size):
    """Build the transformation pipeline for InternVL image preprocessing."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from the target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=None, use_thumbnail=False):
    """Preprocess the image into smaller tiles based on aspect ratio for InternVL."""
    if image_size is None:
        image_size = internvl_input_size
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def preprocess_image_for_internvl(image, input_size=None, max_num=12):
    """Preprocess PIL Image for InternVL model."""
    if input_size is None:
        input_size = internvl_input_size
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

def _apply_and_generate_internvl(
    pixel_values,
    question,
    tokenizer,
    model,
    max_new_tokens,
    num_patches_list=None,   # <-- add this
    history=None,            # <-- and this
    return_history=False,
    generation_config=None,
):
    """
    Wrapper for InternVL chat that supports multi-image inputs.

    Args:
        pixel_values: Tensor of shape (sum_tiles, 3, H, W) as in official demo.
        question: str containing exactly N <image> tags for N images.
        num_patches_list: list[int], tiles-per-image counts; len == N.
        history: prior turn history (or None).
        return_history: whether to return history from model.chat.
        generation_config: optional dict overriding defaults.

    Returns:
        response (and history if return_history=True).
    """
    if generation_config is None:
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1)

    # If user didn’t pass counts and this is a single-image tensor, infer it.
    if num_patches_list is None:
        # Assume single image consisting of `pixel_values.size(0)` tiles.
        num_patches_list = [pixel_values.size(0)]

    out = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=history,
        return_history=return_history
    )
    return out

def generate_with_reasoning(question, image, multi_choice_options, category, tokenizer, model):
    """Run single-turn reasoning directly on the original image."""
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    resized_image = image.resize((resized_width, resized_height))

    pixel_values = preprocess_image_for_internvl(
        resized_image,
        input_size=internvl_input_size,
        max_num=internvl_max_num
    ).to(torch.bfloat16).to(model.device)

    SYSTEM_PROMPT = (
        # "You are a careful visual reasoning assistant. Study the entire image and question, "
        # "explain your reasoning, and then give the final answer enclosed within \\boxed{}. "
        # "Work directly on the provided image without extracting intermediate coordinates."
        "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
    )

    if category == "Perception/OCR":
        qs = question + "\n"
    else:
        qs = question + " The choices are listed below:\n" + (multi_choice_options if multi_choice_options else "") + "\n"

    parsed_options, num_options = parse_lettered_options(multi_choice_options, max_options=6)
    option_text = " Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}. "
    # if num_options == 4:
    #     option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, or D) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}. "
    # elif num_options == 5:
    #     option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}. "
    # else:
    #     option_text = "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, E, or F) enclosed within \\boxed{}. For example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, \\boxed{E}, or \\boxed{F}. "
    # if category == "Perception/OCR":
    #     task_instruction = "Recognize the question and options directly from the image and answer it. "
    # else:
    #     task_instruction = "Use the visual information in the image to answer the question. "

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"<image>\n{qs}\n\n"
        f"{option_text}"
        # f"If you can not arrive at a conclusion, make a reasoned guess and provide the answer in the same format."
    )

    print("---prompt", prompt)

    output_text = _apply_and_generate_internvl(
        pixel_values,
        prompt,
        tokenizer,
        model,
        max_new_tokens
    )

    print("===output_text", output_text)

    predicted_answer = None
    match = re.search(r"\\boxed{([A-F])}", output_text, flags=re.IGNORECASE)
    if match:
        predicted_answer = match.group(1).upper()

    return predicted_answer, output_text


def save_evaluation_results(stats, detailed_results, test_dataset, overall_acc, save_dir):
    """Save evaluation results to JSON files"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Prepare summary statistics for saving
    summary_stats = {}
    for key, val in stats.items():
        count = val["count"]
        correct = val["correct"]
        acc = correct / count if count > 0 else 0
        summary_stats[key] = {
            "total": count,
            "correct": correct,
            "accuracy": acc
        }

    # Save detailed results
    results_filename = os.path.join(save_dir, f"detailed_results.json")
    # transform from array to JSON with index as key
    detailed_results_dict = {str(i): result for i, result in enumerate(detailed_results)}

    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(detailed_results_dict, f, indent=2, ensure_ascii=False)

    # Save summary report
    summary_filename = os.path.join(save_dir, "summary_report.json")
    summary_report = {
        "overall_accuracy": overall_acc,
        "total_evaluated": len(detailed_results),
        "total_correct": sum(1 for r in detailed_results if r["is_correct"]),
        "category_breakdown": summary_stats
    }

    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_dir}")
    print(f"  - Detailed results: {os.path.basename(results_filename)}")
    print(f"  - Summary report: {os.path.basename(summary_filename)}")
    print(f"  - Total problems evaluated: {len(detailed_results)}")

    return results_filename, summary_filename

# --------------- MAIN EXECUTION --------------- #


def main():
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name from model_id for directory naming
    model_name = model_id.split("/")[-1]  # Gets "Qwen2.5-VL-7B-Instruct" from "Qwen/Qwen2.5-VL-7B-Instruct"
    bench_name = "treebench"
    
    # Create timestamped run directory
    run_dir_name = f"{model_name}-{bench_name}-{timestamp}"
    run_dir = os.path.join(results_save_dir, run_dir_name)
    vis_run_dir = os.path.join(run_dir, "visualizations")
    cropped_run_dir = os.path.join(run_dir, "cropped_images")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(vis_run_dir, exist_ok=True)
    os.makedirs(cropped_run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    print(f"Visualizations will be saved to: {vis_run_dir}")
    print(f"Cropped images will be saved to: {cropped_run_dir}")
    
    # Load and process MME dataset
    print("Loading MME dataset...")
    test_dataset = load_mme_dataset()
    print(f"Loaded {len(test_dataset)} samples for testing")

    if len(test_dataset) > 0:
        sample_preview = test_dataset[0]
        print("Sample data:")
        print(f"Question ID: {sample_preview.get('Question_id')}")
        print(f"Question: {sample_preview.get('Text')}")
        print(f"Answer: {sample_preview.get('Ground truth')}")
        print(f"Answer choices: {sample_preview.get('Answer choices')}")
        print(f"Task/Subtask: {sample_preview.get('Task')} / {sample_preview.get('Subtask')}")

    # Load tokenizer and model for InternVL
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    # Run evaluation
    print("Starting evaluation...")
    stats = defaultdict(lambda: {"count": 0, "correct": 0})
    detailed_results = [] 

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        print(f"Processing {i+1}/{len(test_dataset)}")

        question_text = sample["Text"]
        answer_choices = sample.get("Answer choices") or []
        if isinstance(answer_choices, list):
            answer_choices_str = "\n".join(answer_choices)
        else:
            answer_choices_str = str(answer_choices)

        generated_answer, model_output = generate_with_reasoning(
            question_text,
            sample["Image"],
            answer_choices_str,
            sample.get("Task", "MME"),
            tokenizer,
            model
        )

        gt_answer = sample["Ground truth"]
        task = sample.get("Task", "MME")
        subtask = sample.get("Subtask")

        key = f"{task}-{subtask}" if subtask else task
        print(f"Generated: {generated_answer}, Ground Truth: {gt_answer}")

        # Record detailed result for this problem
        is_correct = generated_answer and generated_answer == gt_answer
        detailed_result = {
            "index": i,
            "question_id": sample.get("Question_id"),
            "question": question_text,
            "multi_choice_options": answer_choices,
            "ground_truth": gt_answer,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "task": task,
            "subtask": subtask,
            "image_path": sample.get("ImagePath"),
            "model_output": model_output
        }
        detailed_results.append(detailed_result)

        stats[key]["count"] += 1
        if is_correct:
            stats[key]["correct"] += 1
            print("✓ Correct")
        else:
            print("✗ Incorrect")

        print("="*50)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    total_count = 0
    total_correct = 0

    for key, val in stats.items():
        count = val["count"]
        correct = val["correct"]
        acc = correct / count if count > 0 else 0
        print(f"{key}: Total={count}, Correct={correct}, Accuracy={acc:.2%}")
        total_count += count
        total_correct += correct

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print("="*50)
    print(
        f"OVERALL: Total={total_count}, Correct={total_correct}, Accuracy={overall_acc:.2%}")
    print("="*50)

    # Save results to files
    save_evaluation_results(stats, detailed_results,
                            test_dataset, overall_acc, run_dir)
    

if __name__ == "__main__":
    main()
