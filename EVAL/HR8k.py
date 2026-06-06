import os
import sys
from pathlib import Path
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --------------- LOAD DATA --------------- #
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import re
from collections import defaultdict

RESULTS_SAVE_DIR = Path(
    os.environ.get(
        "RESULT_SAVE_PATH",
        "/home/yangjiacheng/data/jiarui/GRPO/EVAL-all/HR8k",
    )
)

print("Loading HR-Bench 8K dataset...")
dataset = load_dataset("/data1/yangjiacheng/HR-Bench")
test_dataset = dataset["hrbench_8k"]
print(f"Loaded {len(test_dataset)} samples")

# --------------- LOAD processor --------------- #
# trained_model_id = "/home/dangyunkai/yunkai/VLM/VIG-Group/model/Qwen2.5-VL-7B-Instruct"
# trained_model_id = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiacheng/251014-HURT3/MGPO/MGPO"
trained_model_id = os.environ.get(
    "TRAINED_MODEL_ID",
    "/home/yangjiacheng/data/jiarui/HARI-75retry/HARI/checkpoint-30000",
)


from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=trained_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     trained_model_id,
#     torch_dtype="auto",
#     device_map="auto"
# )
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

# processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

# --------------- conversation --------------- #
SYSTEM_PROMPT = (
    # "You are a helpful assistant. "
    # "You will be given an image and a question. "
    # "First, identify the key image area relevant to solving the problem. "
    # "Then, provide the final answer enclosed within \\boxed{}."
    "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
)


def build_prompt(question_text, choices):
    """
    Build a formal, publication-grade English evaluation prompt for multiple-choice VQA.
    """

    opt_map = dict(re.findall(r'([ABCD]):\s*([^,]+)(?=,|$)', choices))
    A = opt_map.get('A', '').strip()
    B = opt_map.get('B', '').strip()
    C = opt_map.get('C', '').strip()
    D = opt_map.get('D', '').strip()

    prompt = (
        f"Question:\n{question_text}\n"
        "The choices are listed below:\n"
        f"A. {A}\n"
        f"B. {B}\n"
        f"C. {C}\n"
        f"D. {D}\n"
        # "First, identify and output the coordinates of the key image area relevant to solving the problem. "
        # "Carefully analyze both the original image and the key image area to solve the question step by step. "
        # "Present your reasoning clearly, and provide the final answer (A, B, C, or D) enclosed within \\boxed{}, "
        # "for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        # "Provide only the final answer (A, B, C, or D) enclosed within \\boxed{}, "
        # "for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
    )
    return prompt


def extract_answer_from_response(response_text):
    """
    Extract the final single-letter answer (A/B/C/D) from model output.
    """
    if not response_text:
        return None
    # Look for boxed answer
    matches = re.findall(r'\\boxed\{([A-D])\}', response_text.strip(), flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


def generate_with_reasoning(prompt, image_path):
    """
    Generate answer using the model. Returns model output text.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }
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
    
    return output_text[0] if output_text else ""


# --------------- EVALUATION --------------- #
# RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_name = Path(trained_model_id).name or "model"
# bench_name = "hrbench_8k"
# run_dir_name = f"{model_name}-{bench_name}-{timestamp}"
# run_dir = RESULTS_SAVE_DIR / run_dir_name
# run_dir.mkdir(parents=True, exist_ok=True)

# print(f"Results will be saved to: {run_dir}")

stats = defaultdict(lambda: {"count": 0, "correct": 0})
all_responses = []

local_image_path = '/data1/yangjiacheng/HR-Bench/8k-images'


for i, item in enumerate(tqdm(test_dataset, desc="Processing")):
    print(f"\n{'='*50}")
    print(f"Processing {i+1}/{len(test_dataset)}")
    print(f"{'='*50}")
    
    # Extract fields
    question_id = item.get("index", "")
    question_text = item.get("question", "")
    true_answer = item.get("answer", "")
    category = item.get("category", "")
    
    # Build image path
    image_path = os.path.join(local_image_path, f"{question_id}.png")

    # Build choices
    choices = f"A: {item.get('A', '')}, B: {item.get('B', '')}, C: {item.get('C', '')}, D: {item.get('D', '')}"

    # Build prompt
    prompt = build_prompt(question_text, choices)

    # Model response
    response_text = generate_with_reasoning(prompt, image_path)
    response_answer = (response_text or "").strip()

    # Extract letter
    extracted_answer = extract_answer_from_response(response_answer)
    
    key = category
    
    print(f"\nGenerated: {extracted_answer}, Ground Truth: {true_answer}")
    
    stats[key]["count"] += 1
    if extracted_answer and true_answer and extracted_answer.upper() == true_answer.upper():
        stats[key]["correct"] += 1
        print("✓ Correct")
    else:
        print("✗ Incorrect")

    result = {
        "Question ID": question_id,
        "Category": category,
        "Question Text": question_text,
        "Image Path": image_path,
        "Choices": choices,
        "True Answer": true_answer,
        "Model Response": response_answer,
        "Extracted Answer": extracted_answer
    }
    all_responses.append(result)

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
result_summary = {}
for key, val in stats.items():
    count = val["count"]
    correct = val["correct"]
    acc = correct / count if count > 0 else 0
    result_summary[key] = {
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

result_summary["Overall"] = {
    "total_samples": total_count,
    "total_correct": total_correct,
    "overall_accuracy": f"{average_accuracy:.2%}"
}

ckpt_name = Path(trained_model_id).name or "model"
step_part = ckpt_name.split("-")[-1] if "checkpoint-" in ckpt_name else "unknown"

os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

summary_path = RESULTS_SAVE_DIR/f"HR8k-checkpoint-{step_part}-{average_accuracy:.2%}.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=4)

# Save detailed responses
# detailed_path = RESULTS_SAVE_DIR / f"HR8k-checkpoint-{step_part}-{average_accuracy:.2%}-detailed.json"
# with detailed_path.open("w", encoding="utf-8") as f:
#     json.dump(
#         {
#             "accuracy": average_accuracy,
#             "correct": total_correct,
#             "total": total_count,
#             "results": all_responses
#         },
#         f,
#         indent=4,
#         ensure_ascii=False
#     )

print(f"\nResults saved to: {summary_path}")
# print(f"Detailed results saved to: {detailed_path}")
