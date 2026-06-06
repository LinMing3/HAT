import json
from pathlib import Path
import re

input_dir = Path("/home/yangjiacheng/data/jiarui/EVAL/GRPO-Vstar")
output_path = Path("/home/yangjiacheng/data/jiarui/curves/GRPO-Vstar.json")

result = []
for file in sorted(input_dir.glob("*.json")):
    # 从文件名提取step
    m = re.search(r"checkpoint-(\d+)", file.name)
    if not m:
        m = re.search(r"_(\d+)\.json", file.name)
    if not m:
        continue
    step = int(m.group(1))
    with file.open(encoding="utf-8") as f:
        data = json.load(f)
    acc_str = data.get("Overall", {}).get("overall_accuracy", None)
    if acc_str and isinstance(acc_str, str) and "%" in acc_str:
        acc = float(acc_str.replace("%", "")) / 100
        result.append({"step": int(step/3), "acc": acc})

# 按step排序
result.sort(key=lambda x: x["step"])

with output_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)