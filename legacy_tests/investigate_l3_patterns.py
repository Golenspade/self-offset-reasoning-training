"""
文件名: investigate_l3_patterns.py
深入调查Level 3数据中的潜在作弊模式
寻找noisy_prop和target_contrapositive之间的简单对应关系
"""

import json
import random
import re


def load_and_sample_l3_data(filename='data/train_L3_complex.json', num_samples=10):
    """加载并随机采样Level 3数据"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                data.append(json.loads(line.strip()))
        
        # 随机采样
        random.seed(42)  # 确保可重现
        samples = random.sample(data, min(num_samples, len(data)))
        return samples
    except Exception as e:
        print(f"❌ 无法加载数据: {e}")
        return []


def analyze_pattern_complexity(samples):
    """分析样本的模式复杂度"""
    print("🔍 分析Level 3数据的模式复杂度...")
    print("=" * 80)
    
    for i, sample in enumerate(samples):
        noisy = sample.get('noisy_prop', '')
        target = sample.get('target_contrapositive', '')
        original = sample.get('original_prop', '')
        
        print(f"\n📝 样本 {i+1}:")
        print(f"  原始命题: {original}")
        print(f"  噪声命题: {noisy}")
        print(f"  目标输出: {target}")
        
        # 分析长度关系
        print(f"  长度比较: 原始({len(original)}) -> 噪声({len(noisy)}) -> 目标({len(target)})")
        
        # 寻找简单的字符串对应关系
        analyze_string_patterns(noisy, target, i+1)


def analyze_string_patterns(noisy, target, sample_num):
    """分析两个字符串之间的简单对应关系"""
    
    # 1. 检查是否存在直接的子字符串复制
    common_substrings = find_common_substrings(noisy, target, min_length=5)
    if common_substrings:
        print(f"  ⚠️  发现共同子字符串: {common_substrings}")
    
    # 2. 检查是否存在简单的变换规律
    check_simple_transformations(noisy, target)
    
    # 3. 检查变量出现模式
    check_variable_patterns(noisy, target)


def find_common_substrings(str1, str2, min_length=5):
    """找到两个字符串的共同子字符串"""
    common = []
    for i in range(len(str1) - min_length + 1):
        for j in range(min_length, len(str1) - i + 1):
            substring = str1[i:i+j]
            if substring in str2 and len(substring) >= min_length:
                common.append(substring)
    
    # 去重并按长度排序
    common = list(set(common))
    common.sort(key=len, reverse=True)
    return common[:3]  # 只返回前3个最长的


def check_simple_transformations(noisy, target):
    """检查简单的变换规律"""
    
    # 检查是否存在简单的前缀/后缀关系
    if target.startswith('~') and noisy.endswith(')'):
        # 检查是否是 ~A -> B 的模式
        if ' -> ' in target:
            parts = target.split(' -> ', 1)
            if len(parts) == 2:
                neg_consequent = parts[0]  # ~B
                neg_antecedent = parts[1]  # ~A
                
                # 检查neg_consequent是否能从noisy的某部分简单得到
                if '|' in noisy:
                    noisy_parts = noisy.split('|')
                    if len(noisy_parts) >= 2:
                        last_part = noisy_parts[-1].strip().rstrip(')')
                        if neg_consequent == f"~{last_part}" or neg_consequent == f"~({last_part})":
                            print(f"  🚨 发现作弊模式: 目标开头'{neg_consequent}'可能直接来自噪声结尾'{last_part}'")


def check_variable_patterns(noisy, target):
    """检查变量出现模式"""
    
    # 提取所有变量
    noisy_vars = set(re.findall(r'\b[pqrst]\b', noisy))
    target_vars = set(re.findall(r'\b[pqrst]\b', target))
    
    print(f"  变量分析: 噪声中有{noisy_vars}, 目标中有{target_vars}")
    
    if noisy_vars == target_vars:
        print(f"  ✓ 变量集合相同")
    else:
        print(f"  ⚠️  变量集合不同")


def analyze_noise_effectiveness(samples):
    """分析噪声的有效性"""
    print(f"\n🎯 分析噪声有效性...")
    print("=" * 50)
    
    noise_types_found = {
        'type1_implication_to_disjunction': 0,
        'type2_double_negation': 0,
        'type3_redundant_parentheses': 0,
        'minimal_change': 0,
        'no_change': 0
    }
    
    for i, sample in enumerate(samples):
        original = sample.get('original_prop', '')
        noisy = sample.get('noisy_prop', '')
        
        # 检查噪声类型
        if '->' in original and '|' in noisy and '->' not in noisy:
            noise_types_found['type1_implication_to_disjunction'] += 1
        elif '~~' in noisy:
            noise_types_found['type2_double_negation'] += 1
        elif noisy.count('(') > original.count('('):
            noise_types_found['type3_redundant_parentheses'] += 1
        elif len(noisy) - len(original) <= 2:
            noise_types_found['minimal_change'] += 1
        elif noisy == original:
            noise_types_found['no_change'] += 1
    
    print("噪声类型分布:")
    for noise_type, count in noise_types_found.items():
        percentage = count / len(samples) * 100
        print(f"  {noise_type}: {count} ({percentage:.1f}%)")
    
    # 评估噪声多样性
    total_effective = sum(noise_types_found.values()) - noise_types_found['no_change']
    if noise_types_found['no_change'] > len(samples) * 0.1:
        print(f"  ⚠️  警告: {noise_types_found['no_change']} 个样本没有应用噪声")
    
    if noise_types_found['minimal_change'] > len(samples) * 0.3:
        print(f"  ⚠️  警告: {noise_types_found['minimal_change']} 个样本的噪声变化很小")


def suggest_improvements(samples):
    """基于分析结果提出改进建议"""
    print(f"\n💡 改进建议...")
    print("=" * 50)
    
    print("基于分析结果，建议以下改进措施:")
    print("1. 增加噪声应用次数: 从1次增加到2-3次")
    print("2. 组合多种噪声类型: 同时应用type1, type2, type3")
    print("3. 增加结构多样性: 确保主蕴含符不总是在最外层")
    print("4. 添加更多噪声类型: 如交换律、结合律变换")
    print("5. 增加随机性: 在噪声应用中引入更多随机因素")


def cross_evaluation_test():
    """交叉评估测试建议"""
    print(f"\n🧪 交叉评估测试建议...")
    print("=" * 50)
    
    print("为了验证Level 3模型是否学到了真正的逻辑:")
    print("1. 加载Level 3训练好的模型")
    print("2. 用它评估Level 1的验证集")
    print("3. 如果准确率很低(接近0%)，证明模型确实在作弊")
    print("4. 如果准确率合理(>30%)，说明模型学到了一些通用规律")


def main():
    """主函数"""
    print("🕵️ Level 3数据模式调查报告")
    print("=" * 80)
    
    # 加载样本数据
    samples = load_and_sample_l3_data(num_samples=8)
    
    if not samples:
        print("❌ 无法加载数据，请确保数据文件存在")
        return
    
    print(f"✅ 成功加载 {len(samples)} 个样本进行分析")
    
    # 进行各种分析
    analyze_pattern_complexity(samples)
    analyze_noise_effectiveness(samples)
    suggest_improvements(samples)
    cross_evaluation_test()
    
    print(f"\n🎯 调查总结:")
    print("如果发现了明显的作弊模式，需要:")
    print("1. 重新设计数据生成策略")
    print("2. 增加噪声的复杂度和随机性") 
    print("3. 重新生成Level 3数据集")
    print("4. 重新训练并验证结果")


if __name__ == "__main__":
    main()
