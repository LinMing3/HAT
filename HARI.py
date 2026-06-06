import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"         
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import random
import time
from collections import defaultdict

import datasets
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

import copy
import inspect
import os
import re
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
    prepare_multimodal_messages,
    # prepare_multimodal_messages_vllm,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    RepeatSampler,
    # CurriculumSampler,
    # WeightedRepeatSampler,
    disable_dropout_in_model,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    truncate_with_protected_tokens,
    unsplit_pixel_values_by_grid,
)


# -------------------- load dataset -------------------- #
dataset_dir = "/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/dataset"
train_dataset = datasets.load_from_disk(dataset_dir)
print(train_dataset)
print(train_dataset[0])

# -------------------- Processor -------------------- #
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an image and one question. Output the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
)

def make_conversation(example):
    # 读取字段：Text, Answer choices, Ground truth, Image
    question = example["Text"]
    choices = example["Answer choices"]  # list of strings (A)-(E)
    gt = example["Ground truth"]         # single letter，如 "D"
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": 
                        "Question: " + question + " " +\
                        "The choices are listed below: " + " ".join(choices) + " "+\
                        "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
                    ,
                },
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt, 
        # "image": example["Image"], 
        "Ground truth": gt,
        # "conversation": conversation
    }

# train_dataset = train_dataset.select(range(50))
train_dataset = train_dataset.map(make_conversation, num_proc=16, writer_batch_size=64)
train_dataset = train_dataset.rename_column("Image", "image")
print(train_dataset[0]["prompt"])

print(train_dataset[0]["image"])

# -------------------- 模型 + LoRA -------------------- #
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -------------------- 自定义 Curriculum Sampler -------------------- #
class CurriculumSampler(Sampler[int]):
    def __init__(
        self,
        data_source,
        difficulties: torch.Tensor,
        buckets: torch.Tensor,
        batch_size: int,
        total_steps: int,
        sigma: float = 0.25,
        lambda_: float = 0.1,
        num_buckets: int = 5,
        mini_repeat_count: int = 1,
        repeat_count: int = 1,
        replacement: bool = False,
        generator: torch.Generator | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.difficulties = difficulties.double()
        self.buckets = buckets.long()
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.sigma = sigma
        self.lambda_ = lambda_
        self.num_buckets = num_buckets
        self.mini_repeat_count = mini_repeat_count
        self.repeat_count = repeat_count
        self.replacement = replacement
        self.generator = generator
        # self.rank = rank
        # self.world_size = world_size

        self.bucket_prev = torch.zeros(num_buckets, dtype=torch.double)
        self.bucket_now = torch.zeros(num_buckets, dtype=torch.double)
        self.bucket_correct_cum = torch.zeros(num_buckets, dtype=torch.double)
        self.bucket_total_cum = torch.zeros(num_buckets, dtype=torch.double)
        self.step = 0
        self.weights = torch.ones(len(data_source), dtype=torch.double)
        self.used_mask = torch.zeros(len(data_source), dtype=torch.bool)

    def set_step(self, step: int):
        self.step = step
        t = min(max(step / max(1, self.total_steps), 0.0), 1.0)
        p_td = torch.exp(- (self.difficulties - t) ** 2 / (2 * self.sigma ** 2))
        delta_b = torch.abs(self.bucket_now - self.bucket_prev)
        bucket_boost = delta_b[self.buckets]
        self.weights = torch.clamp(p_td + self.lambda_ * bucket_boost, min=1e-8)

    def update_bucket_stats(self, bucket_correct: torch.Tensor, bucket_total: torch.Tensor):
        self.bucket_prev = self.bucket_now.clone()
        self.bucket_correct_cum += bucket_correct.double()
        self.bucket_total_cum += bucket_total.double()
        mask = self.bucket_total_cum > 0
        new_rates = torch.zeros_like(self.bucket_now)
        new_rates[mask] = self.bucket_correct_cum[mask] / self.bucket_total_cum[mask]
        self.bucket_now = torch.clamp(new_rates, 0.0, 1.0)

    # def __iter__(self):
    #     total = len(self.data_source)
    #     if total == 0:
    #         return iter([])

    #     # rank 切分索引
    #     all_idx = torch.arange(total)
    #     shard_idx = all_idx[self.rank::self.world_size]
    #     shard_weights = self.weights[shard_idx]

    #     # 按向上取整的 chunk 数，允许最后一批不足
    #     num_chunks = math.ceil(len(shard_idx) / self.batch_size)
    #     chunks: list[list[int]] = []

    #     # 本地掩码
    #     local_mask = torch.zeros_like(shard_weights, dtype=torch.bool)
    #     for _ in range(num_chunks):
    #         avail = (shard_weights[~local_mask] > 0).sum().item()
    #         if avail == 0:
    #             break
    #         take = min(self.batch_size, avail)
    #         local_weights = shard_weights.clone()
    #         local_weights[local_mask] = 0.0
    #         batch_local = torch.multinomial(
    #             local_weights, num_samples=take, replacement=self.replacement, generator=self.generator
    #         )
    #         batch_global = shard_idx[batch_local].tolist()
    #         chunks.append(batch_global)
    #         local_mask[batch_local] = True

    #     for batch_id, chunk in enumerate(chunks):
    #         print(f"[sampler] step={self.step}, batch={batch_id}, idx={chunk},repeat_count={self.repeat_count}, mini_repeat_count={self.mini_repeat_count},batch_size={self.batch_size}")
    #         for _ in range(self.repeat_count):
    #             for idx in chunk:
    #                 for _ in range(self.mini_repeat_count):
    #                     yield idx

    def __iter__(self):
        total = len(self.data_source)
        if total == 0:
            return iter([])

        num_chunks = (total + self.batch_size - 1) // self.batch_size  # 允许尾批
        if self.used_mask is None or self.used_mask.all():
            self.used_mask = torch.zeros(total, dtype=torch.bool)

        local_weights = self.weights.clone()
        local_weights[self.used_mask] = 0.0

        chunks = []
        for _ in range(num_chunks):
            avail = (local_weights > 0).sum().item()
            if avail == 0:
                break
            take = min(self.batch_size, avail)
            batch_idx = torch.multinomial(
                local_weights, num_samples=take, replacement=self.replacement, generator=self.generator
            )
            chunks.append(batch_idx.tolist())
            self.used_mask[batch_idx] = True
            local_weights[batch_idx] = 0.0

        for batch_id, chunk in enumerate(chunks):
            for _ in range(self.repeat_count):
                for idx in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield idx

    def __len__(self):
        total = len(self.data_source)
        num_chunks = (total + self.batch_size - 1) // self.batch_size
        return num_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count

    # def __len__(self):
    #     total = len(self.data_source)
    #     # 这个 rank 拿到的样本数（向上取整分片）
    #     per_rank = math.ceil((total - self.rank) / self.world_size)
    #     num_chunks = math.ceil(per_rank / self.batch_size)
    #     return num_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count


# -------------------- 自定义 GRPO Trainer -------------------- #
import copy, re, random
from transformers import Trainer as HFTrainer

class GRPORetryTrainer(GRPOTrainer):
    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        print("Initializing CurriculumSampler...")
        difficulties = torch.tensor(dataset["difficulty"], dtype=torch.float)
        buckets = torch.tensor(dataset["bucket"], dtype=torch.long)

        total_steps = self.args.max_steps  # 或 num_train_epochs * steps_per_epoch
        sampler = CurriculumSampler(
            data_source=dataset,
            difficulties=difficulties,
            buckets=buckets,
            batch_size=self.args.generation_batch_size // self.num_generations,
            total_steps=total_steps,
            rank=self.accelerator.process_index,
            world_size=self.accelerator.num_processes,
            sigma=0.15,
            lambda_=0.1,
            num_buckets=int(buckets.max().item() + 1),
            mini_repeat_count=self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            replacement=False,
            # shuffle=self.shuffle_dataset if hasattr(self, "shuffle_dataset") else True,
        )

        # 初始化权重
        sampler.set_step(self.state.global_step)
        self.curriculum_sampler = sampler
        print("CurriculumSampler initialized.")

        return sampler
        # return RepeatSampler(
        #     data_source=dataset,
        #     mini_repeat_count=self.num_generations,
        #     batch_size=self.args.generation_batch_size // self.num_generations,
        #     repeat_count=self.num_iterations * self.args.steps_per_generation,
        #     shuffle=self.shuffle_dataset,
        #     seed=self.args.seed,
        # )

    # def _generate_and_score_completions(
    #     self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    # ) -> dict[str, Union[torch.Tensor, Any]]:
    #     device = self.accelerator.device
    #     mode = "train" if self.model.training else "eval"

    #     prompts = [x["prompt"] for x in inputs]
    #     # print(f"[DEBUG] prompt: {prompts[0]}")
    #     original_prompts = copy.deepcopy(prompts)
    #     # print(f"[DEBUG] first_prompt[0]: {original_prompts[0]}")

    #     # 多模态处理
    #     kwargs = {}
    #     has_images = "image" in inputs[0]
    #     image_split_sizes = None
    #     if has_images:
    #         images = [example.get("image") for example in inputs]
    #         kwargs = {"images": [[img] for img in images]}
    #         for prompt in prompts:
    #             if isinstance(prompt, list):
    #                 prepare_multimodal_messages(prompt, num_images=1)
    #                 # print(f"[DEBUG] prepared prompt[0]: {prompt}")
    #         if hasattr(self.processing_class, "_get_num_multimodal_tokens"):
    #             image_sizes = [(image.height, image.width) for image in images]
    #             multimodal_extra_data = self.processing_class._get_num_multimodal_tokens(image_sizes)
    #             image_split_sizes = multimodal_extra_data.num_image_patches

    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

    #     prompt_inputs = self.processing_class(
    #         text=prompts_text,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left",
    #         add_special_tokens=False,
    #         **kwargs,
    #     )
    #     # 重要：用 HFTrainer._prepare_inputs，避免递归
    #     prompt_inputs = HFTrainer._prepare_inputs(self, prompt_inputs)
    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    #     if "image_grid_thw" in prompt_inputs and image_split_sizes is None:
    #         image_split_sizes = prompt_inputs["image_grid_thw"].prod(dim=1).tolist()

    #     # ===== 生成分支（保持原版 vLLM/paged/regular） =====
    #     # 这里只给 regular path，若你启用 vLLM/paged，请从原文件复制对应分支
    #     with (
    #         profiling_context(self, "transformers.generate"),
    #         unwrap_model_for_generation(
    #             self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
    #         ) as unwrapped_model,
    #         torch.no_grad(),
    #         FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
    #     ):
    #         prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
    #         prompt_completion_ids = unwrapped_model.generate(
    #             **prompt_inputs, generation_config=self.generation_config, disable_compile=True
    #         )
    #     prompt_length = prompt_ids.size(1)
    #     prompt_ids = prompt_completion_ids[:, :prompt_length]
    #     completion_ids = prompt_completion_ids[:, prompt_length:]

    #     # ===== 解码 & 20% 重试 =====
    #     completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     if is_conversational(inputs[0]):
    #         completions = []
    #         for prompt, completion in zip(prompts, completions_text):
    #             bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
    #             completions.append([{"role": "assistant", "content": bootstrap + completion}])
    #     else:
    #         completions = completions_text
        
        
    #     # print(f"[DEBUG] first_prompt[0]: {prompts_text}")
    #     print(f"[DEBUG] first_text[0]: {completions_text}")

    #     boxed = re.compile(r"\\boxed\{([A-E])\}", re.IGNORECASE)
    #     gts = [x["Ground truth"] for x in inputs]

    #     def judge(txt, gt):
    #         m = boxed.search(txt or "")
    #         return bool(m and m.group(1).upper() == gt.upper())

    #     retry_idx = [i for i, (c, gt) in enumerate(zip(completions_text, gts))
    #                  if not judge(c, gt) and random.random() < 0.2]

    #     print(f"[DEBUG] retry_idx: {retry_idx}")


    #     if retry_idx:
    #         refine = ("Your answer was incorrect. Try again. Provide the final answer (A, B, C, D, or E) enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}.")
    #         # prompts_text 是前面 apply_chat_template 后的字符串，这里仅拼接 refine，切勿再套模板
    #         # retry_text = [prompts_text[i] + "\n" + refine for i in retry_idx]
    #         retry_text = []
    #         retry_imgs = []
    #         for i in retry_idx:
    #             m = boxed.search(completions_text[i] or "")
    #             first_letter = m.group(1).upper() if m else completions_text[i].strip()

    #             msgs = [
    #                 {"role": "system", "content": SYSTEM_PROMPT},
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {"type": "image"},
    #                         {
    #                             "type": "text",
    #                             "text": (
    #                                 "Question: " + inputs[i]["Text"] + " "
    #                                 "The choices are listed below: " + " ".join(inputs[i]["Answer choices"]) + " "
    #                                 "Perform reasoning on the problem, and only provide the final answer (A, B, C, D, or E) "
    #                                 "enclosed within \\boxed{}, for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
    #                             ),
    #                         },
    #                     ],
    #                 },
    #                 {"role": "assistant", "content": first_letter},     # 用字符串即可
    #                 {"role": "user", "content": {"type": "text", "text": refine}},
    #             ]
    #             prepare_multimodal_messages(msgs, num_images=1)         # 需要插入图像占位符
    #             retry_text.append(
    #                 self.processing_class.apply_chat_template(
    #                     msgs, tokenize=False, add_generation_prompt=True
    #                 )
    #             )

    #         # print(f"[DEBUG] second_prompt[0]: {retry_text[0]}")

    #         retry_inputs = self.processing_class(
    #             text=retry_text,
    #             images=[[images[i]] for i in retry_idx] if has_images else None,
    #             return_tensors="pt",
    #             padding=True,
    #             padding_side="left",
    #             add_special_tokens=False,
    #         )

    #         retry_inputs = HFTrainer._prepare_inputs(self, retry_inputs)

    #         with torch.no_grad():
    #             retry_out = self.model.generate(
    #                 **retry_inputs, generation_config=self.generation_config, disable_compile=True
    #             )
    #         pl = retry_inputs["input_ids"].size(1)
    #         retry_comp_ids = retry_out[:, pl:]
    #         retry_comp_text = self.processing_class.batch_decode(retry_comp_ids, skip_special_tokens=True)

    #         print(f"[DEBUG] second_text[0]: {retry_comp_text}")
    #         for local_i, orig_i in enumerate(retry_idx):

    #             first_ok = judge(completions_text[orig_i], gts[orig_i])
    #             second_ok = judge(retry_comp_text[local_i], gts[orig_i])
    #             reward_dbg = 1.0 if first_ok else (0.5 if second_ok else 0.0)
    #             print(f"[DEBUG] idx={orig_i} gt={gts[orig_i]} | first_ok={first_ok} "
    #                   f"text1='{completions_text[orig_i]}' | second_ok={second_ok} "
    #                   f"text2='{retry_comp_text[local_i]}' | reward={reward_dbg}")
                
    #             new_ids = retry_comp_ids[local_i]
    #             # 对齐长度：不足则 pad，过长则截断
    #             if new_ids.size(0) < completion_ids.size(1):
    #                 pad_len = completion_ids.size(1) - new_ids.size(0)
    #                 new_ids = torch.cat([new_ids, new_ids.new_full((pad_len,), self.pad_token_id)])
    #             else:
    #                 new_ids = new_ids[: completion_ids.size(1)]
    #             completion_ids[orig_i] = new_ids
    #             completions_text[orig_i] = retry_comp_text[local_i]
    #             if is_conversational(inputs[0]):
    #                 completions[orig_i] = [{"role": "assistant", "content": retry_comp_text[local_i]}]


    #     # ===== 重新计算 mask/长度（原逻辑） =====
    #     is_eos = completion_ids == self.eos_token_id
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    #     completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]
    #     completion_lengths = completion_mask.sum(1)
    #     agg_completion_lengths = self.accelerator.gather(completion_lengths)
    #     num_items_in_batch = agg_completion_lengths.sum()

    #     if self.mask_truncated_completions:
    #         truncated_completions = ~is_eos.any(dim=1)
    #         completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    #     logits_to_keep = completion_ids.size(1)
    #     batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

    #     # ===== 参考/旧 logprob 计算（原版） =====
    #     old_per_token_logps, sampling_per_token_logps = None, None
    #     ref_per_token_logps = None
    #     # 如需 vLLM 或 importance sampling，请复制原文件对应分支，这里省略

    #     # ===== 计算奖励、优势、日志，保持原版 =====
    #     rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)
    #     rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = rewards - mean_grouped_rewards

    #     if self.scale_rewards in ["group", "none"]:
    #         std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
    #         std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
    #     else:
    #         std_rewards = rewards.std().expand_as(rewards)

    #     is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
    #     if self.scale_rewards != "none":
    #         advantages = advantages / (std_rewards + 1e-4)

    #     process_slice = slice(
    #         self.accelerator.process_index * len(prompts),
    #         (self.accelerator.process_index + 1) * len(prompts),
    #     )
    #     all_process_advantages = advantages.clone()
    #     advantages = advantages[process_slice]

    #     # 日志/metrics（保留原版）
    #     if mode == "train":
    #         self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
    #     self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
    #     self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
    #     self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
    #     self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
    #     agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
    #     term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
    #     clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
    #     self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
    #     if len(term_completion_lengths) == 0:
    #         term_completion_lengths = torch.zeros(1, device=device)
    #     self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
    #     self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
    #     self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())
    #     for i, reward_func_name in enumerate(self.reward_func_names):
    #         mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
    #         self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
    #         std_func_rewards = nanstd(rewards_per_func[:, i]).item()
    #         self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
    #     self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
    #     self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
    #     self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

    #     self._logs["prompt"].extend(gather_object(prompts_text))
    #     self._logs["completion"].extend(gather_object(completions_text))
    #     for i, name in enumerate(self.reward_func_names):
    #         self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
    #     self._logs["advantages"].extend(all_process_advantages.tolist())
    #     if has_images:
    #         self._logs["image"].extend(gather_object(images))

    #     output = {
    #         "prompt_ids": prompt_ids,
    #         "prompt_mask": prompt_mask,
    #         "completion_ids": completion_ids,
    #         "completion_mask": completion_mask,
    #         "advantages": advantages,
    #         "num_items_in_batch": num_items_in_batch,
    #     }
    #     if old_per_token_logps is not None:
    #         output["old_per_token_logps"] = old_per_token_logps
    #     if ref_per_token_logps is not None:
    #         output["ref_per_token_logps"] = ref_per_token_logps
    #     if self.use_vllm and sampling_per_token_logps is not None:
    #         output["importance_sampling_ratio"] = torch.exp(old_per_token_logps - sampling_per_token_logps)
    #     if "pixel_values" in prompt_inputs:
    #         output["pixel_values"] = prompt_inputs["pixel_values"]
    #     if "image_grid_thw" in prompt_inputs:
    #         output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
    #     if "pixel_attention_mask" in prompt_inputs:
    #         output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
    #     if "image_sizes" in prompt_inputs:
    #         output["image_sizes"] = prompt_inputs["image_sizes"]
    #     if image_split_sizes is not None:
    #         output["image_split_sizes"] = image_split_sizes
    #     return output
    
    def _generate_with_retry(self, prompts, inputs):
        device = self.accelerator.device

        # 第一次生成推理+答案 
        first_pids, first_cids, first_logps, extra = self._generate_single_turn(prompts)

        # 判定答案是否正确
        texts = self.processing_class.batch_decode(
            [torch.tensor(ids, device=device) for ids in first_cids],
            skip_special_tokens=True,
        )
        print(f'<_generate_with_retry> First texts 0: {texts}')
        # print(f'<_generate_with_retry> First texts 1: {texts[1]}')
        import re
        boxed = re.compile(r"\\boxed\{([A-E])\}", re.IGNORECASE)
        gts = [x["Ground truth"] for x in inputs]
        rewards = []
        answers = []
        for t, gt in zip(texts, gts):
            m = boxed.search(t)
            ans = m.group(1).upper() if m else None
            answers.append(ans)
            rewards.append(1.0 if ans == gt.upper() else 0.0)

        retry_idx = [i for i, r in enumerate(rewards) if r == 0.0]
        # retry_idx_all = [i for i, r in enumerate(rewards) if r == 0.0]
        # # 50% 概率重试
        # retry_idx = [i for i in retry_idx_all if random.random() <= 0.5]
        print(f'<_generate_with_retry> Rewards: {rewards}, retry indices: {retry_idx}')
        if not retry_idx:
            total_tokens = torch.tensor([len(c) for c in first_cids], device=device)
            total_tokens = self.accelerator.gather(total_tokens).sum()
            return first_pids, first_cids, total_tokens, first_logps, extra

        # 对错误的样本再生成一次
        retry_prompts = []
        for i in retry_idx:
            base = prompts[i]
            # refine = "Your answer was incorrect. Try again.  Give a reasoning trace and final answer in \\boxed{} (A-E only)."
            refine = "Your answer was incorrect. Try again. Provide the final answer (A, B, C, D, or E) enclosed within \\boxed{},for example, \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, or \\boxed{E}."
            if isinstance(base, list):
                retry_prompts.append(base 
                                    # + [{"role": "assistant", "content": [{"type": "text", "text": texts[i]}]}]
                                    + [{"role": "assistant", "content": [{"type": "text", "text": answers[i]}]}]
                                    + [{"role": "user", "content": [{"type": "text", "text": refine}]}]
                                    )
            else:
                retry_prompts.append(f"{base}\n{refine}")

        # print(f'<_generate_with_retry> Retry prompts 0: {retry_prompts[0]}')

        second_pids, second_cids, second_logps, _ = self._generate_single_turn(retry_prompts)

        second_texts = self.processing_class.batch_decode(
            [torch.tensor(ids, device=device) for ids in second_cids],
            skip_special_tokens=True,
        )
        print(f'<_generate_with_retry> Second texts : {second_texts}')

        second_rewards = []
        second_round_success = [False] * len(prompts)
        for local_i, (txt, gt) in enumerate(zip(second_texts, [gts[i] for i in retry_idx])):
            m = boxed.search(txt)
            ans = m.group(1).upper() if m else None
            second_rewards.append(1.0 if ans == gt.upper() else 0.0)
            if second_rewards[-1] == 1.0:
                orig_i = retry_idx[local_i]
                second_round_success[orig_i] = True
        final_correct = rewards[:]
        for local_i, orig_i in enumerate(retry_idx):
            final_correct[orig_i] = second_rewards[local_i]
        print(f'<_generate_with_retry> Second rewards: {final_correct}')

        extra["second_round_success"] = second_round_success
        extra["final_correct"] = final_correct
        # 替换对应位置
        for local_i, orig_i in enumerate(retry_idx):
            first_cids[orig_i] = second_cids[local_i]
            if first_logps is not None and second_logps is not None:
                first_logps[orig_i] = second_logps[local_i]

        total_tokens = torch.tensor([len(c) for c in first_cids], device=device)
        total_tokens = self.accelerator.gather(total_tokens).sum()
        return first_pids, first_cids, total_tokens, first_logps, extra

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        # kwargs = {}
        # has_images = "image" in inputs[0]
        # image_split_sizes = None
        # if has_images:
        #     images = [example.get("image") for example in inputs]
        #     kwargs = {"images": [[img] for img in images]}
        #     for prompt in prompts:
        #         if isinstance(prompt, list):  # i.e., when using conversational data
        #             prepare_multimodal_messages(prompt, num_images=1)

        #     if hasattr(self.processing_class, "_get_num_multimodal_tokens"):
        #         image_sizes = [(image.height, image.width) for image in images]
        #         multimodal_extra_data = self.processing_class._get_num_multimodal_tokens(image_sizes)
        #         image_split_sizes = multimodal_extra_data.num_image_patches
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # # [{"role": "user", "content": "What color is the sky?"}] to
        # # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        
        if images is not None:
            print('images detected in conversational prompt, preparing multimodal messages')
            prompts = [
                prepare_multimodal_messages(prompt, image_list) for prompt, image_list in zip(prompts, images, strict=True)
            ]

        # print("Generating completions for batch of size", len(prompts))

        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
             self._generate_with_retry(prompts, inputs)
        )

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                if isinstance(bootstrap, list):  # for VLM, the format might be [{"type": "text", "text": "..."}]
                    assert len(bootstrap) == 1 and bootstrap[0]["type"] == "text"
                    bootstrap = bootstrap[0]["text"]
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # # 合并 extra_fields 后：
        # if "final_correct" in inputs[0]:
        #     final_correct = torch.tensor([bool(inp.get("final_correct", False)) for inp in inputs], device=device)
        #     second_mask = torch.tensor([bool(inp.get("second_round_success", False)) for inp in inputs], device=device)
        #     factor = torch.where(second_mask, torch.tensor(0.5, device=device), torch.tensor(1.0, device=device))
        #     final_reward = final_correct.float() * factor
        #     rewards_per_func = final_reward.unsqueeze(1)  # shape (B,1)
        # else:
        #     rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        mask = torch.tensor([inp.get("second_round_success", False) for inp in inputs], device=device)
        factor = torch.tensor(
            [bool(inp.get("second_round_success", False)) for inp in inputs],
            device=device,
        )
        factor = self.accelerator.gather(factor)          # 长度对齐
        factor_bool = factor.bool()                       # 或 factor > 0
        scale = torch.where(factor_bool,
                            torch.tensor(0.5, device=device),
                            torch.tensor(1.0, device=device))
        rewards_per_func = rewards_per_func * scale.unsqueeze(1)

        
        print(f"<_generate_and_score_completions> Applied second round success mask, rewards_per_func: {rewards_per_func.tolist()}")

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        # print(f"<_generate_and_score_completions> Computed rewards: {rewards}")
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # if images is not None:
        #     self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        sampler = self.curriculum_sampler if hasattr(self, "curriculum_sampler") else None
        if self.use_curriculum_sampler and sampler is not None:
            # print("[sampler update] updating curriculum sampler statistics...")
            num_buckets = sampler.num_buckets
            bucket_correct = torch.zeros(num_buckets, device=self.accelerator.device)
            bucket_total = torch.zeros(num_buckets, device=self.accelerator.device)

            group_size = self.num_generations
            assert len(inputs) % group_size == 0, "expect prompts repeated num_generations times"
            success_threshold = getattr(self, "curriculum_success_threshold", 0.999)

            for start in range(0, len(inputs), group_size):
                bucket_id = inputs[start].get("bucket")
                if bucket_id is None:
                    continue
                group_rewards = rewards[start : start + group_size]
                correct_cnt = (group_rewards >= success_threshold).sum()
                ratio = correct_cnt / group_size
                bucket_correct[bucket_id] += ratio
                bucket_total[bucket_id] += 1

            self.accelerator.reduce(bucket_correct, reduction="sum")
            self.accelerator.reduce(bucket_total, reduction="sum")
            sampler.update_bucket_stats(bucket_correct.cpu(), bucket_total.cpu())
            sampler.set_step(self.state.global_step)
            if self.accelerator.is_main_process and self.state.global_step % 2 == 0:
                w = sampler.weights
                delta_b = (sampler.bucket_now - sampler.bucket_prev).abs()
                print(f"[sampler update] step={self.state.global_step}, "
                    f"w[min={w.min():.3g}, max={w.max():.3g}, mean={w.mean():.3g}], "
                    f"delta_b={delta_b.tolist()}")

        # print(f"<_generate_and_score_completions> Done generating and scoring completions.{inputs}")

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output


# -------------------- 奖励函数 -------------------- #
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

# -------------------- 训练参数 -------------------- #
training_args = GRPOConfig(
    output_dir="HARI",
    learning_rate=1e-5,
    remove_unused_columns=False,
    num_train_epochs=2,
    bf16=True,
    per_device_train_batch_size=4,
    max_completion_length=512,
    num_generations=4,
    max_prompt_length=4096,
    report_to=["tensorboard"],
    logging_steps=500,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=1000,
)

# -------------------- 训练 -------------------- #
trainer = GRPORetryTrainer(
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
