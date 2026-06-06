import json

# 读取 conversation_vars.jsonl 文件
conversations = []
with open("/home/yangjiacheng/data/jiarui/prompt/conversation_vars.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            conversations.append(json.loads(line))

# # 拼接对话为 ChatML 格式
# def to_chatml(prompts, texts, retry_prompts=None, second_texts=None, retry_idx=None):
#     chat = []
#     # prompts: List[List[dict]] or List[str]
#     # texts: List[str]
#     for i, prompt in enumerate(prompts):
#         # 如果 prompt 是 list（如多轮消息），拼接每一轮
#         if isinstance(prompt, list):
#             for turn in prompt:
#                 role = turn.get("role", "user")
#                 # content 可能是 list（多模态），只拼接 type=text 的内容
#                 content = turn.get("content", "")
#                 if isinstance(content, list):
#                     text = "".join([c.get("text", "") for c in content if c.get("type") == "text" and c.get("text")])
#                 else:
#                     text = str(content)
#                 chat.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
#         else:
#             chat.append(f"<|im_start|>user\n<|vision start|><|image pad|><|vision end|>{prompt}<|im_end|>\n")
#         # assistant 回复
#         chat.append(f"<|im_start|>assistant\n{texts[i]}<|im_end|>\n")
#     # 如果有重试，插入重试内容
#     if retry_prompts and second_texts and retry_idx:
#         # retry_prompts: List[List[dict]]
#         for offset, (idx, retry_prompt) in enumerate(zip(retry_idx, retry_prompts)):
#             insert_pos = 2 * idx + 2 + offset * 2
#             # 拼接 retry_prompt
#             retry_chat = []
#             if isinstance(retry_prompt, list):
#                 for turn in retry_prompt:
#                     role = turn.get("role", "user")
#                     content = turn.get("content", "")
#                     if isinstance(content, list):
#                         text = "".join([c.get("text", "") for c in content if c.get("type") == "text" and c.get("text")])
#                     else:
#                         text = str(content)
#                     retry_chat.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
#             else:
#                 retry_chat.append(f"<|im_start|>user\n<|vision start|><|image pad|><|vision end|>{retry_prompt}<|im_end|>\n")
#             # assistant 回复
#             retry_chat.append(f"<|im_start|>assistant\n{second_texts[retry_idx.index(idx)]}<|im_end|>\n")
#             # 插入到主 chat
#             chat[insert_pos:insert_pos] = retry_chat
#     return "".join(chat)

# 每个 json 条目按顺序展开，每个 prompt/text 对应一个 chatml
chatml_list = []
for conv in conversations:
    if len(chatml_list) >= 1500:
        break
    prompts = conv.get("prompts", [])
    texts = conv.get("texts", [])
    retry_prompts = conv.get("retry_prompts", [])
    second_texts = conv.get("second_texts", [])
    retry_idx = conv.get("retry_idx", [])

    # 先处理主轮
    for i in range(len(prompts)):
        chat = []
        # 展开主 prompt
        prompt = prompts[i]
        if isinstance(prompt, list):
            for turn in prompt:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if isinstance(content, list):
                    text = "".join([c.get("text", "") for c in content if c.get("type") == "text" and c.get("text")])
                else:
                    text = str(content)
                chat.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
        else:
            chat.append(f"<|im_start|>user\n<|vision start|><|image pad|><|vision end|>{prompt}<|im_end|>\n")
        # assistant 回复
        chat.append(f"<|im_start|>assistant\n{texts[i]}<|im_end|>\n")
        if i in retry_idx:
            chat.append(f"<|im_start|>user\nYour answer was incorrect. Try again. Provide reasoning and the final answer (A, B, C, D, or E) enclosed within \\boxed{{}},for example, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, or \\boxed{{E}}.<|im_end|>\n")
            second_text = second_texts[retry_idx.index(i)]
            chat.append(f"<|im_start|>assistant\n{second_text}<|im_end|>\n")
            chatml_list.append({"chatml": "".join(chat)})


# 保存为 json 文件
with open("conversation_vars_chatml-1500.json", "w", encoding="utf-8") as f:
    json.dump(chatml_list, f, ensure_ascii=False, indent=2)

print("已保存为 conversation_vars_chatml-1500.json")
if chatml_list:
    print(chatml_list[0]["chatml"])  
    print("-----")
    print(chatml_list[1]["chatml"])  
    print("-----")
    print(chatml_list[2]["chatml"])
    print("-----")
    print(chatml_list[3]["chatml"])