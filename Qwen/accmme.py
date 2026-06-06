import json

def main():
    with open("/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/HARI-sampling/EVAL-all/MME-logs/MME-57.32-checkpoint-7045.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total_correct = 0
    total_total = 0

    perception_correct = 0
    perception_total = 0

    reasoning_correct = 0
    reasoning_total = 0

    for name, info in data.items():
        # ① 显式排除最后一个
        if name == "Average Reward":
            continue

        # ② 再防一手：只要没有 total/correct 也跳过
        if "total" not in info or "correct" not in info:
            continue

        total = info["total"]
        correct = info["correct"]

        # 总体
        total_total += total
        total_correct += correct

        # 按大类
        if name.startswith("Perception"):
            perception_total += total
            perception_correct += correct
        elif name.startswith("Reasoning"):
            reasoning_total += total
            reasoning_correct += correct

    # 计算
    total_acc = total_correct / total_total * 100
    perception_acc = perception_correct / perception_total * 100
    reasoning_acc = reasoning_correct / reasoning_total * 100

    print(f"总准确率: {total_acc:.1f}% (correct={total_correct}, total={total_total})")
    print(f"Perception 准确率: {perception_acc:.1f}% (correct={perception_correct}, total={perception_total})")
    print(f"Reasoning 准确率: {reasoning_acc:.1f}% (correct={reasoning_correct}, total={reasoning_total})")

if __name__ == "__main__":
    main()
