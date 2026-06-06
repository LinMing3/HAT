import json

def main():
    with open("/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/HARI_random/EVAL/checkpoint-7045-treebench-20251230_162042/summary_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    overall_total = 0
    overall_correct = 0

    per_total = 0
    per_correct = 0

    rea_total = 0
    rea_correct = 0

    for name, info in data.items():
        # 跳过最后这个总汇
        if name == "Overall":
            continue

        total = info["total"]
        correct = info["correct"]

        # overall
        overall_total += total
        overall_correct += correct

        # 分类
        if name.startswith("Perception/"):
            per_total += total
            per_correct += correct
        elif name.startswith("Reasoning/"):
            rea_total += total
            rea_correct += correct

    overall_acc = overall_correct / overall_total * 100
    per_acc = per_correct / per_total * 100
    rea_acc = rea_correct / rea_total * 100

    print(f"Overall accuracy: {overall_acc:.1f}% (correct={overall_correct}, total={overall_total})")
    print(f"Perception accuracy: {per_acc:.1f}% (correct={per_correct}, total={per_total})")
    print(f"Reasoning accuracy: {rea_acc:.1f}% (correct={rea_correct}, total={rea_total})")

if __name__ == "__main__":
    main()
