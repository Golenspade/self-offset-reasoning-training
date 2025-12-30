"""
generate_simple_dataset.py

生成一份極簡、乾淨的命題邏輯數據集，用於訓練 / 檢驗
「由噪聲析取式推回逆否命題」的基本能力。

特點：
- 僅覆蓋最核心的 4 種模式（與 logic_rules.validate_rule_logic 中一致）
- 每條樣本都滿足：original_prop ≡ noisy_prop，且 target_contrapositive
  是 original_prop 的逆否命題（在這 4 種模板下由設計保證）
- 輸出格式與 data/train.json、data/val.json 一致

用法：
    python scripts/generate_simple_dataset.py
會在 data/ 目錄下生成：
    data/train_simple.json
    data/val_simple.json
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# 確保可以從 scripts/ 目錄正確導入項目包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logic_transformer.logic_rules import (
    corrected_disjunction_to_contrapositive,
)


# 使用的命題變量集合（可按需擴展）
VARIABLES: List[str] = ["p", "q", "r", "s", "t"]


# 與 logic_rules.validate_rule_logic 中的 6 個測試樣例一致的 4 類模式
# 只允許變量或單一否定，避免產生雙重否定或複雜嵌套
TEMPLATES: List[Dict[str, str]] = [
    {
        "name": "p_implies_q",
        "original": "{a} -> {b}",
        "noisy": "(~{a} | {b})",
        "contrapositive": "~{b} -> ~{a}",
    },
    {
        "name": "not_p_implies_not_q",
        "original": "~{a} -> ~{b}",
        "noisy": "({a} | ~{b})",
        "contrapositive": "{b} -> {a}",
    },
    {
        "name": "p_implies_not_q",
        "original": "{a} -> ~{b}",
        "noisy": "(~{a} | ~{b})",
        "contrapositive": "{b} -> ~{a}",
    },
    {
        "name": "not_p_implies_q",
        "original": "~{a} -> {b}",
        "noisy": "({a} | {b})",
        "contrapositive": "~{b} -> {a}",
    },
]


def sample_variables() -> Dict[str, str]:
    """隨機選擇兩個不同的命題變量。"""
    a, b = random.sample(VARIABLES, 2)
    return {"a": a, "b": b}


def generate_simple_sample(complexity: str = "simple_clean") -> Dict[str, str]:
    """生成一條極簡、保證正確的樣本。"""
    template = random.choice(TEMPLATES)
    vars_map = sample_variables()

    original = template["original"].format(**vars_map)
    noisy = template["noisy"].format(**vars_map)
    contrapositive = template["contrapositive"].format(**vars_map)

    return {
        "original_prop": original,
        "noisy_prop": noisy,
        "target_contrapositive": contrapositive,
        "complexity": complexity,
    }


def generate_dataset(
    n_samples: int, complexity: str = "simple_clean"
) -> List[Dict[str, str]]:
    """批量生成數據集。"""
    return [generate_simple_sample(complexity) for _ in range(n_samples)]


def save_dataset(samples: List[Dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in samples:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def self_check(num_checks: int = 32) -> bool:
    """用現有的 symbolic 規則檢查模板是否與 rule-based 邏輯一致。

    對若干條隨機樣本，驗證：
        corrected_disjunction_to_contrapositive(noisy_prop)
        == target_contrapositive
    若有不一致，返回 False 並打印示例。
    """

    for _ in range(num_checks):
        sample = generate_simple_sample()
        predicted = corrected_disjunction_to_contrapositive(sample["noisy_prop"])
        if predicted != sample["target_contrapositive"]:
            print("[WARNING] rule-based prediction mismatch:")
            print("  noisy_prop:", sample["noisy_prop"])
            print("  rule_based:", predicted)
            print("  target   :", sample["target_contrapositive"])
            return False
    return True


def main() -> None:
    # 可以按需調整樣本數量
    train_size = 2000
    val_size = 400

    if not self_check():
        print("模板與 rule-based 規則不一致，請先檢查 generate_simple_dataset.py。")
        return

    train_samples = generate_dataset(train_size)
    val_samples = generate_dataset(val_size)

    train_path = os.path.join("data", "train_simple.json")
    val_path = os.path.join("data", "val_simple.json")

    save_dataset(train_samples, train_path)
    save_dataset(val_samples, val_path)

    print(f"已生成 train_simple.json, 條數: {len(train_samples)}")
    print(f"已生成 val_simple.json, 條數: {len(val_samples)}")
    print("示例樣本:")
    for example in train_samples[:3]:
        print("  - original_prop       =", example["original_prop"])
        print("    noisy_prop          =", example["noisy_prop"])
        print("    target_contrapositive =", example["target_contrapositive"])


if __name__ == "__main__":
    main()
