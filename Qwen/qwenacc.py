import json

def recover_int_correct(total: int, shown_acc: float) -> int:
    """
    给定：
      - total: 这个任务的样本数
      - shown_acc: 表格里这个任务的准确率（已经四舍五入到 1 位小数，比如 43.4）
    反推出：整数 correct，使 round(correct / total * 100, 1) == shown_acc
    """
    candidates = []
    for k in range(total + 1):
        pct = k / total * 100
        if round(pct, 1) == shown_acc:
            candidates.append(k)

    if not candidates:
        # 理论上不会进来，大多数时候是浮点误差的锅
        return round(total * shown_acc / 100.0)

    if len(candidates) == 1:
        return candidates[0]

    # 万一有多个，就选真正百分比最接近的那个
    return min(candidates, key=lambda k: abs(k / total * 100 - shown_acc))


def main():
    # ① 读你的 result.json
    with open("/home/dangyunkai/yunkai/VLM/VIG-Group/jiarui/qwen.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # ② 这里填“表格里的这一行方法”的准确率（1 位小数、百分数）
    #    ——下面这个是示例，请按你真实表格改
    method_acc = {
        "Perception-Remote Sensing": 32.7,
        "Perception-Monitoring": 27.3,
        "Perception-Autonomous_Driving": 30.0,
        "Perception-Diagram and Table": 83.0,
        "Perception-OCR with Complex Context": 87.6,
        "Reasoning-Monitoring": 28.7,
        "Reasoning-Autonomous_Driving": 23.0,
        "Reasoning-Diagram and Table": 62.0,
        "Reasoning-OCR with Complex Context": 72.0,
    }
    # ↑↑↑ 这里的 key 名必须跟 json 里的完全一致，否则要么报错，要么就漏掉

    overall_correct = 0
    overall_total = 0

    per_correct = 0
    per_total = 0

    rea_correct = 0
    rea_total = 0

    for name, info in data.items():
        if name == "Overall":
            continue

        total = info["total"]

        if name not in method_acc:
            raise ValueError(f"表格中没有这个任务的准确率: {name}")

        shown_acc = method_acc[name]

        # 关键一步：反推整数 correct
        correct = recover_int_correct(total, shown_acc)

        # overall
        overall_total += total
        overall_correct += correct

        # 分类
        if name.startswith("Perception-"):
            per_total += total
            per_correct += correct
        elif name.startswith("Reasoning-"):
            rea_total += total
            rea_correct += correct

    # ③ 统一再算一次加权准确率
    overall_acc = overall_correct / overall_total * 100
    per_acc = per_correct / per_total * 100
    rea_acc = rea_correct / rea_total * 100

    print(f"Overall accuracy: {overall_acc:.1f}% (correct={overall_correct}, total={overall_total})")
    print(f"Perception accuracy: {per_acc:.1f}% (correct={per_correct}, total={per_total})")
    print(f"Reasoning accuracy: {rea_acc:.1f}% (correct={rea_correct}, total={rea_total})")

if __name__ == "__main__":
    main()
