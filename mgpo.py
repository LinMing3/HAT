import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"         
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --------------- LOAD DATA --------------- #
import datasets
from collections import defaultdict

dataset_dir = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"
train_dataset = datasets.load_from_disk(dataset_dir)
print(train_dataset)
print(train_dataset[0])

# --------------- LOAD processor --------------- #
from transformers import AutoProcessor

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

# --------------- First conversation --------------- #
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You will be given an image and a question. "
    "First, identify the key image area relevant to solving the problem. "
    "Then, provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, "
    "for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
)

def make_conversation(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Question:" + " " + example["Text"] + " "  +\
                "The choices are listed below:" + " " + ' '.join(example["Answer choices"]) + " " +\
                "First, identify and output the coordinates of the key image area relevant to solving the problem. " +\
                "Carefully analyze both the original image and the key image area to solve the question step by step. "
                "Present your reasoning clearly, and provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, "
                "for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
                },
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["Image"],
    }
train_dataset = train_dataset.select(range(50))
train_dataset = train_dataset.map(make_conversation, num_proc=16)
print(train_dataset[0]["prompt"])
print(train_dataset[0]["image"])

# --------------- Load model and lora --------------- #
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------- Reward function --------------- #
import re

def accuracy_reward(completions, **kwargs):
    rewards = []
    
    for completion, org_ground_truth in zip(completions, kwargs["Ground truth"]):
        print(completion)
        new_ansewr = re.search(r'\\boxed\{([A-E])\}', completion)
        if new_ansewr:
            print("new_ansewr: ", new_ansewr.group(1))
            print("ground_truth: ", org_ground_truth)
        if new_ansewr and new_ansewr.group(1) == org_ground_truth:
            rewards.append(1)
        else:
            rewards.append(0)
        print("reward: ", rewards[-1])

    return rewards

# --------------- Configure training arguments --------------- #
from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="MGPO",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    num_train_epochs=2,
    bf16=True,
    # Parameters that control the data preprocessing
    per_device_train_batch_size=4,
    max_completion_length=512,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=2048,
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=500,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=500,
)

# ---------------Training --------------- #
from trl import GRPOTrainer
import time

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

print("start training")
start_time = time.time()

trainer.train()

end_time = time.time()
print("end training")
print(f"Training time: {(end_time - start_time)/60:.2f} minutes")

trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")