import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"         
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer,AutoModel
path = "/home/yangjiacheng/data/model/InternVL3_5-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage = True,
    use_flash_attn=True,
    trust_remote_code = True
).cuda()

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(path)


from datasets import load_dataset,load_from_disk

DATASET_ROOT = "/home/yangjiacheng/data/jiarui/dataset"  
train_dataset = load_from_disk(DATASET_ROOT)
print(train_dataset[0])

SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
)


def make_conversation(example):
    conversation = [
        {
            "role": "system", 
            "content":[
                {"type": "text", "text": SYSTEM_PROMPT}]},
        # {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Question:" + " " + example["Text"] + " "  +\
                "The choices are listed below:" +  " " .join(example["Answer choices"])+" "+\
                "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}." 
                },
            ],
        },
    ]
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": conversation,
        # "prompt": prompt,
        # "image": example["Image"],
    }

# train_dataset = process_dataset(train_dataset)
train_dataset = train_dataset.rename_column("Image", "image")
train_dataset = train_dataset.map(make_conversation, num_proc=32)
print(train_dataset[0]["prompt"])
print(train_dataset[0]["image"])

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

def _completion_to_text(completion):
    """Join all text blocks from a conversation-format completion."""
    parts = []
    for message in completion:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(part for part in parts if part)

BOXED_RE = re.compile(r"\\boxed\{([A-E])\}", re.IGNORECASE)

MAX_ANSWER_CHARS = 40  

def answer_reward(completions, **kwargs):
    ground_truth = kwargs.get("Ground truth")
    texts = [_completion_to_text(c) for c in completions]
    rewards = []
    for txt, gt in zip(texts, ground_truth):
        cleaned = "".join(txt.split())
        if len(cleaned) > MAX_ANSWER_CHARS:
            rewards.append(0.0)
            continue

        matches = BOXED_RE.findall(txt)
        if len(matches) != 1:          # 没有或出现多个 \boxed{}
            rewards.append(0.0)
            continue

        ans = matches[0].upper()
        rewards.append(1.0 if ans == gt.upper() else 0.0)
    print("Rewards:", rewards)
    return rewards

from trl_main import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    output_dir="HARI",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    num_train_epochs=3,
    bf16=True,

    # Parameters that control the data preprocessing
    per_device_train_batch_size=4,
    max_completion_length=512, # default: 256
    num_generations=4, # default: 8
    # generation_batch_size=12, # default: 8

    # dataloader_num_workers=0,
    # need to think twice
    max_prompt_length=4096, # default: 2048

    # dataloader_drop_last=True,

    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=500,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=1500,
)

trainer = GRPOTrainer(
    model=model,
    # processing_class=processor,
    reward_funcs=[answer_reward],
    # reward_funcs=[accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training...")
start_time = time.time()

trainer.train()

end_time = time.time()
print("end training")
print(f"Training time: {(end_time - start_time)/60:.2f} minutes")

trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")
