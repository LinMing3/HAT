import json

def get_mme_perception_reasoning(data):
    perception_keys = [
        "Perception-Remote Sensing",
        "Perception-Monitoring",
        "Perception-Autonomous_Driving",
        "Perception-OCR with Complex Context",
        "Perception-Diagram and Table"
    ]
    reasoning_keys = [
        "Reasoning-Monitoring",
        "Reasoning-Autonomous_Driving",
        "Reasoning-OCR with Complex Context",
        "Reasoning-Diagram and Table"
    ]
    def calc(keys):
        total = sum(data[k]["total"] for k in keys if k in data)
        correct = sum(data[k]["correct"] for k in keys if k in data)
        return correct / total * 100 if total > 0 else None
    return calc(perception_keys), calc(reasoning_keys)

def get_tree_perception_reasoning(data):
    perception_keys = [
        "Perception/Attributes-Perception/Attributes",
        "Perception/OCR-Perception/OCR",
        "Perception/Physical State-Perception/Physical State",
        "Perception/Object Retrieval-Perception/Object Retrieval",
        "Perception/Material-Perception/Material"
    ]
    reasoning_keys = [
        "Reasoning/Comparison-Reasoning/Comparison",
        "Reasoning/Ordering-Reasoning/Ordering",
        "Reasoning/Contact and Occlusion-Reasoning/Contact and Occlusion",
        "Reasoning/Spatial Containment-Reasoning/Spatial Containment",
        "Reasoning/Perspective Transform-Reasoning/Perspective Transform"
    ]
    def calc(keys):
        total = sum(data[k]["total"] for k in keys if k in data)
        correct = sum(data[k]["correct"] for k in keys if k in data)
        return correct / total * 100 if total > 0 else None
    return calc(perception_keys), calc(reasoning_keys)

def print_result(name, perception, reasoning):
    print(f"{name}:")
    if perception is not None:
        print(f"  Perception加权准确率: {perception:.2f}%")
    else:
        print("  Perception无数据")
    if reasoning is not None:
        print(f"  Reasoning加权准确率: {reasoning:.2f}%")
    else:
        print("  Reasoning无数据")
    print()

# 路径
mme_path = "/home/yangjiacheng/data/jiarui/EVAL/hardtoeasy-MME/mme_checkpoint-60000.json"
tree_path = "/home/yangjiacheng/data/jiarui/EVAL/hardtoeasy-TreeBench/Tree-checkpoint-60000-39.26%-2.json"

# 读取并处理
with open(mme_path, "r", encoding="utf-8") as f:
    mme_data = json.load(f)
with open(tree_path, "r", encoding="utf-8") as f:
    tree_data = json.load(f)

mme_per, mme_rea = get_mme_perception_reasoning(mme_data)
tree_per, tree_rea = get_tree_perception_reasoning(tree_data)

print_result("MME", mme_per, mme_rea)
print_result("TreeBench", tree_per, tree_rea)