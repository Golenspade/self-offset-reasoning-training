"""
文件名: generate_robust_dataset.py
生成"无法作弊"的鲁棒数据集
增加数据的熵和多样性，堵死所有作弊捷径
"""

import json
import random
import os
import re
import sys
from pathlib import Path

# 确保可以从 scripts/ 目录导入项目根模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from logic_utils import (
    generate_simple_proposition,
    generate_recursive_implication,
    to_contrapositive,
    verify_equivalence,
)


def add_robust_noise(prop_str: str, num_applications: int = 3) -> str:
    """添加鲁棒的、多样化的噪声，确保没有简单的字符串对应关系"""
    result = prop_str

    # 定义所有可用的噪声变换
    noise_functions = [
        add_noise_type1_robust,
        add_noise_type2_robust,
        add_noise_type3_robust,
        add_noise_type4_robust,
        add_noise_type5_new,
        add_noise_type6_new,
    ]

    # 随机应用多次噪声
    for _ in range(num_applications):
        noise_func = random.choice(noise_functions)
        try:
            result = noise_func(result)
        except Exception:
            continue  # 如果某个噪声函数失败，继续尝试其他的

    return result


def add_noise_type1_robust(prop_str: str) -> str:
    """鲁棒版噪声类型1：蕴含转析取，增加随机性"""
    # 随机决定是否应用
    if random.random() < 0.7:
        # 找到主蕴含符并转换
        if " -> " in prop_str:
            parts = prop_str.split(" -> ", 1)
            if len(parts) == 2:
                antecedent = parts[0].strip()
                consequent = parts[1].strip()

                # 随机选择否定方式
                if random.random() < 0.5:
                    neg_antecedent = (
                        f"~({antecedent})" if " " in antecedent else f"~{antecedent}"
                    )
                else:
                    neg_antecedent = f"~{antecedent}"

                return f"({neg_antecedent} | {consequent})"

    return prop_str


def add_noise_type2_robust(prop_str: str) -> str:
    """鲁棒版噪声类型2：双重否定，随机应用到不同位置"""
    variables = re.findall(r"\b[pqrst]\b", prop_str)
    if variables and random.random() < 0.6:
        # 随机选择1-2个变量
        num_vars = random.randint(1, min(2, len(variables)))
        selected_vars = random.sample(variables, num_vars)

        for var in selected_vars:
            # 随机选择双重否定的形式
            if random.random() < 0.5:
                replacement = f"~~{var}"
            else:
                replacement = f"~(~{var})"

            prop_str = re.sub(rf"\b{var}\b", replacement, prop_str, count=1)

    return prop_str


def add_noise_type3_robust(prop_str: str) -> str:
    """鲁棒版噪声类型3：冗余括号，随机应用"""
    variables = re.findall(r"\b[pqrst]\b", prop_str)
    if variables and random.random() < 0.5:
        var = random.choice(variables)
        # 随机选择括号的形式
        if random.random() < 0.5:
            replacement = f"({var})"
        else:
            replacement = f"(({var}))"

        prop_str = re.sub(rf"\b{var}\b", replacement, prop_str, count=1)

    return prop_str


def add_noise_type4_robust(prop_str: str) -> str:
    """鲁棒版噪声类型4：交换律，随机应用"""
    # 查找可交换的表达式
    patterns = [
        (r"\(([^()]+)\s*&\s*([^()]+)\)", r"(\2 & \1)"),
        (r"\(([^()]+)\s*\|\s*([^()]+)\)", r"(\2 | \1)"),
    ]

    if random.random() < 0.4:
        for pattern, replacement in patterns:
            if re.search(pattern, prop_str):
                prop_str = re.sub(pattern, replacement, prop_str, count=1)
                break

    return prop_str


def add_noise_type5_new(prop_str: str) -> str:
    """新噪声类型5：添加恒真/恒假表达式"""
    if random.random() < 0.3:
        variables = re.findall(r"\b[pqrst]\b", prop_str)
        if variables:
            var = random.choice(variables)
            # 添加恒真或恒假表达式
            if random.random() < 0.5:
                tautology = f"({var} | ~{var})"  # 恒真
                prop_str = f"({prop_str} & {tautology})"
            else:
                contradiction = f"({var} & ~{var})"  # 恒假
                prop_str = f"({prop_str} | {contradiction})"

    return prop_str


def add_noise_type6_new(prop_str: str) -> str:
    """新噪声类型6：德摩根定律变换"""
    if random.random() < 0.3:
        # 查找可以应用德摩根定律的模式
        patterns = [
            (r"~\(([^()]+)\s*&\s*([^()]+)\)", r"(~\1 | ~\2)"),
            (r"~\(([^()]+)\s*\|\s*([^()]+)\)", r"(~\1 & ~\2)"),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, prop_str):
                prop_str = re.sub(pattern, replacement, prop_str, count=1)
                break

    return prop_str


def generate_diverse_proposition(complexity_level: str = "medium") -> str:
    """生成多样化的命题，确保结构多样性"""
    if complexity_level == "simple":
        return generate_simple_proposition()
    elif complexity_level == "medium":
        # 50%概率生成简单命题，50%生成中等复杂度
        if random.random() < 0.5:
            return generate_simple_proposition()
        else:
            return generate_recursive_implication(max_depth=2)
    else:  # complex
        # 30%简单，40%中等，30%复杂
        rand = random.random()
        if rand < 0.3:
            return generate_simple_proposition()
        elif rand < 0.7:
            return generate_recursive_implication(max_depth=2)
        else:
            return generate_recursive_implication(max_depth=3)


def generate_robust_sample(complexity_level: str = "medium") -> dict:
    """生成一个鲁棒的训练样本"""
    # 生成原始命题
    original_prop = generate_diverse_proposition(complexity_level)

    # 生成逆否命题
    target_contrapositive = to_contrapositive(original_prop)

    # 应用鲁棒噪声
    noise_applications = random.randint(2, 4)  # 随机2-4次噪声
    noisy_prop = add_robust_noise(original_prop, noise_applications)

    return {
        "original_prop": original_prop,
        "noisy_prop": noisy_prop,
        "target_contrapositive": target_contrapositive,
        "complexity_level": complexity_level,
        "noise_applications": noise_applications,
    }


def generate_robust_dataset(num_samples: int, complexity_level: str = "medium") -> list:
    """生成鲁棒数据集"""
    dataset = []
    successful_samples = 0
    attempts = 0
    max_attempts = num_samples * 5

    print(f"生成鲁棒数据集: {complexity_level} 级别, {num_samples} 样本")

    while successful_samples < num_samples and attempts < max_attempts:
        attempts += 1

        try:
            sample = generate_robust_sample(complexity_level)

            # 验证样本质量
            if (
                sample["noisy_prop"].strip()
                and sample["target_contrapositive"].strip()
                and sample["noisy_prop"] != sample["target_contrapositive"]
                and len(sample["noisy_prop"]) > 5
            ):

                dataset.append(sample)
                successful_samples += 1

                if successful_samples % 500 == 0:
                    print(f"  已生成 {successful_samples}/{num_samples} 个样本...")

        except Exception:
            continue

    print(f"  ✅ 成功生成 {successful_samples} 个样本")
    return dataset


def save_robust_dataset(dataset: list, filename: str):
    """保存鲁棒数据集"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  ✅ 数据集已保存到 {filename}")


def analyze_robust_dataset(dataset: list, name: str):
    """分析鲁棒数据集的质量"""
    if not dataset:
        return

    print(f"\n📊 {name} 质量分析:")

    # 长度分析
    input_lengths = [len(sample["noisy_prop"]) for sample in dataset]
    target_lengths = [len(sample["target_contrapositive"]) for sample in dataset]

    print(f"  样本数量: {len(dataset)}")
    print(f"  平均输入长度: {sum(input_lengths)/len(input_lengths):.1f}")
    print(f"  平均目标长度: {sum(target_lengths)/len(target_lengths):.1f}")
    print(
        f"  长度范围: 输入({min(input_lengths)}-{max(input_lengths)}), "
        f"目标({min(target_lengths)}-{max(target_lengths)})"
    )

    # 噪声应用分析
    noise_counts = {}
    for sample in dataset:
        count = sample.get("noise_applications", 1)
        noise_counts[count] = noise_counts.get(count, 0) + 1

    print(f"  噪声应用分布: {noise_counts}")

    # 显示样本示例
    print("  样本示例:")
    for i, sample in enumerate(dataset[:3]):
        print(f"    {i+1}. 原始: {sample['original_prop']}")
        print(f"       噪声: {sample['noisy_prop']}")
        print(f"       目标: {sample['target_contrapositive']}")


def main():
    """主函数：生成鲁棒数据集"""
    print("🛡️ 生成鲁棒的、无法作弊的数据集")
    print("=" * 60)

    random.seed(42)

    # 生成三个级别的鲁棒数据集
    datasets_config = [
        {
            "name": "Level 1 鲁棒版",
            "complexity": "simple",
            "train_size": 3000,
            "val_size": 500,
        },
        {
            "name": "Level 2 鲁棒版",
            "complexity": "medium",
            "train_size": 2500,
            "val_size": 400,
        },
        {
            "name": "Level 3 鲁棒版",
            "complexity": "complex",
            "train_size": 2000,
            "val_size": 300,
        },
    ]

    for config in datasets_config:
        print(f"\n🔧 生成 {config['name']}")
        print("-" * 40)

        # 生成训练集
        train_dataset = generate_robust_dataset(
            config["train_size"], config["complexity"]
        )
        train_filename = f"data/train_{config['name'].replace(' ', '_').lower()}.json"
        save_robust_dataset(train_dataset, train_filename)
        analyze_robust_dataset(train_dataset, f"{config['name']} 训练集")

        # 生成验证集
        val_dataset = generate_robust_dataset(config["val_size"], config["complexity"])
        val_filename = f"data/val_{config['name'].replace(' ', '_').lower()}.json"
        save_robust_dataset(val_dataset, val_filename)
        analyze_robust_dataset(val_dataset, f"{config['name']} 验证集")

    print(f"\n🎉 所有鲁棒数据集生成完成！")
    print(f"\n📋 生成的文件:")
    for config in datasets_config:
        train_file = f"data/train_{config['name'].replace(' ', '_').lower()}.json"
        val_file = f"data/val_{config['name'].replace(' ', '_').lower()}.json"
        print(f"  📊 {train_file}")
        print(f"  📊 {val_file}")


if __name__ == "__main__":
    main()
