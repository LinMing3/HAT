import re
BOXED_ONLY_RE = re.compile(r"^\s*\\boxed\{([A-E])\}\s*$", re.IGNORECASE)

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

def accuracy_reward(completions, **kwargs):
    rewards = []
    ground_truth = kwargs.get("Ground truth", [])
    texts = [
        _completion_to_text(c) if not isinstance(c, str) else c
        for c in completions
    ]
    for text, org_ground_truth in zip(texts, ground_truth):
        match = BOXED_ONLY_RE.match(text)
        rewards.append(int(bool(match and match.group(1).upper()
                                 == org_ground_truth.upper())))
    print("Rewards:", rewards)
    return rewards

def demo_accuracy_reward():
    completions = [
        r"\boxed{A}",
        r"\boxed{D},\boxed{B}",
        r"Final answer: \boxed{C}",
        r"\boxed{D}\nExtra",
        r"\boxed{E}",
    ]
    ground_truth = ["A", "D", "C", "d", "X"]
    rewards = accuracy_reward(completions, **{"Ground truth": ground_truth})
    print("Expected:", [1, 1, 0, 0, 0])
    print("Actual:  ", rewards)

demo_accuracy_reward()