# --------------- LOAD DATA --------------- #
from datasets import load_dataset, Image
import os
from test.dataset_image_process import smart_resize
from collections import defaultdict

def add_full_path(example):
    example["Image"] = os.path.join(image_root, example["Image"])
    return example

def resize_dataset(example):
    image = example["Image"]
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor = image_factor,
        min_pixels = min_pixels,
        max_pixels = max_pixels
    )
    image = image.resize((resized_width, resized_height))
    example["Image"] = image
    return example

def process_dataset(dataset_split):
    dataset_split = dataset_split.map(add_full_path, writer_batch_size = 100, batch_size =100)
    dataset_split = dataset_split.cast_column("Image", Image(decode=True))
    dataset_split = dataset_split.map(resize_dataset, writer_batch_size = 100, batch_size =100)
    return dataset_split

image_factor = 28
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

image_root = "/home/dangyunkai/yunkai/VLM/VIG-Group/dataset/MME-RealWorld"
dataset = load_dataset(
    "json", 
    data_files="/home/dangyunkai/yunkai/VLM/VIG-Group/dataset/MME-RealWorld/MME_RealWorld.json",
    split="train[:15%]",
    )
split_dataset = dataset.train_test_split(test_size=0.2, seed=342)

# train_dataset = process_dataset(split_dataset["train"])
# print(train_dataset)
# print(train_dataset[0])

test_dataset = process_dataset(split_dataset["test"])
print(test_dataset)
print(test_dataset[0])

# --------------- LOAD processor --------------- #
from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

# ---------------  model --------------- #
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# --------------- conversation --------------- #
import re
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Given an image and one question. "
    "provide the answer (A, B, C, D, or E) enclosed within \\boxed{}, "
    "i.e., \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
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
                "Present your reasoning clearly, and provide the final answer (A, B, C, D, or E) " +\
                "enclosed within \\boxed{}, "
                "i.e., \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
                },
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
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
    print("output_text", output_text)
    
    new_ansewr = re.search(r'\\boxed\{([A-E])\}', output_text[0])
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
    print(f"{key}: 总数={count}, 正确={correct}, 准确率={acc:.2%}")
