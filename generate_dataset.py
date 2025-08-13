"""
文件名: generate_dataset.py
数据集生成脚本
生成用于训练的命题逻辑数据集
"""

import json
import random
from logic_utils import (
    generate_simple_proposition, 
    generate_complex_proposition,
    to_contrapositive, 
    add_noise,
    verify_equivalence
)


def generate_training_sample(complexity='simple', noise_types=['type1']):
    """生成一个训练样本"""
    
    # 生成原始命题
    if complexity == 'simple':
        original_prop = generate_simple_proposition()
    else:
        original_prop = generate_complex_proposition()
    
    # 生成逆否命题
    target_contrapositive = to_contrapositive(original_prop)
    
    # 添加噪声
    noisy_prop = add_noise(original_prop, noise_types)
    
    # 验证逆否命题的正确性（可选，用于调试）
    # is_valid = verify_equivalence(original_prop, target_contrapositive)
    
    return {
        'original_prop': original_prop,
        'noisy_prop': noisy_prop,
        'target_contrapositive': target_contrapositive,
        'complexity': complexity
    }


def generate_dataset(num_samples, complexity='simple', noise_types=['type1']):
    """生成完整的数据集"""
    
    dataset = []
    successful_samples = 0
    attempts = 0
    max_attempts = num_samples * 3  # 防止无限循环
    
    while successful_samples < num_samples and attempts < max_attempts:
        attempts += 1
        
        try:
            sample = generate_training_sample(complexity, noise_types)
            
            # 基本验证：确保生成的样本不为空
            if (sample['noisy_prop'].strip() and 
                sample['target_contrapositive'].strip() and
                sample['noisy_prop'] != sample['target_contrapositive']):
                
                dataset.append(sample)
                successful_samples += 1
                
                # 每生成1000个样本打印进度
                if successful_samples % 1000 == 0:
                    print(f"已生成 {successful_samples} 个样本...")
                    
        except Exception as e:
            print(f"生成样本时出错: {e}")
            continue
    
    print(f"成功生成 {successful_samples} 个样本，总尝试次数: {attempts}")
    return dataset


def save_dataset(dataset, filename):
    """保存数据集到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in dataset:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"数据集已保存到 {filename}")


def load_dataset(filename):
    """从JSON文件加载数据集"""
    dataset = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def analyze_dataset(dataset):
    """分析数据集的基本统计信息"""
    print(f"\n数据集分析:")
    print(f"总样本数: {len(dataset)}")
    
    # 分析复杂度分布
    complexity_count = {}
    for sample in dataset:
        complexity = sample.get('complexity', 'unknown')
        complexity_count[complexity] = complexity_count.get(complexity, 0) + 1
    
    print(f"复杂度分布: {complexity_count}")
    
    # 分析长度分布
    noisy_lengths = [len(sample['noisy_prop']) for sample in dataset]
    target_lengths = [len(sample['target_contrapositive']) for sample in dataset]
    
    print(f"噪声命题平均长度: {sum(noisy_lengths) / len(noisy_lengths):.2f}")
    print(f"目标命题平均长度: {sum(target_lengths) / len(target_lengths):.2f}")
    
    # 显示几个样本
    print(f"\n样本示例:")
    for i, sample in enumerate(dataset[:5]):
        print(f"样本 {i+1}:")
        print(f"  原始: {sample['original_prop']}")
        print(f"  噪声: {sample['noisy_prop']}")
        print(f"  目标: {sample['target_contrapositive']}")
        print()


def main():
    """主函数：生成训练和验证数据集"""
    
    print("开始生成自偏移推理训练数据集...")
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    # 生成训练集 (简单命题，仅类型1噪声)
    print("\n生成训练集...")
    train_dataset = generate_dataset(
        num_samples=10000,
        complexity='simple',
        noise_types=['type1']
    )
    
    # 生成验证集
    print("\n生成验证集...")
    val_dataset = generate_dataset(
        num_samples=2000,
        complexity='simple',
        noise_types=['type1']
    )
    
    # 保存数据集
    save_dataset(train_dataset, 'data/train.json')
    save_dataset(val_dataset, 'data/val.json')
    
    # 分析数据集
    print("\n=== 训练集分析 ===")
    analyze_dataset(train_dataset)
    
    print("\n=== 验证集分析 ===")
    analyze_dataset(val_dataset)
    
    print("\n数据集生成完成！")
    print("文件位置:")
    print("  训练集: data/train.json")
    print("  验证集: data/val.json")


if __name__ == "__main__":
    main()
