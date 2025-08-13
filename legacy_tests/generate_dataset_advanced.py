"""
高级数据集生成脚本
实现配置文件驱动、课程学习和复杂度递增的数据生成策略
"""

import json
import random
import os
from typing import Dict, List, Callable
from logic_utils import (
    generate_simple_proposition, 
    generate_complex_proposition,
    generate_recursive_implication,
    to_contrapositive, 
    add_noise,
    verify_equivalence
)


def load_config(config_path: str = 'configs/dataset_config.json') -> Dict:
    """加载数据集配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None


def get_generator_function(func_name: str) -> Callable:
    """根据函数名获取生成器函数"""
    generators = {
        'generate_simple_proposition': generate_simple_proposition,
        'generate_complex_proposition': generate_complex_proposition,
        'generate_recursive_proposition': lambda: generate_recursive_implication(max_depth=3),
    }
    
    if func_name not in generators:
        raise ValueError(f"未知的生成器函数: {func_name}")
    
    return generators[func_name]


def generate_training_sample(generator_func: Callable, noise_types: List[str], 
                           num_applications: int = 1, **kwargs) -> Dict:
    """
    生成一个训练样本
    支持不同的生成器函数和噪声配置
    """
    # 生成原始命题
    if 'max_depth' in kwargs:
        # 对于递归生成器，传递max_depth参数
        original_prop = generate_recursive_implication(max_depth=kwargs['max_depth'])
    else:
        original_prop = generator_func()
    
    # 生成逆否命题
    target_contrapositive = to_contrapositive(original_prop)
    
    # 添加噪声
    noisy_prop = add_noise(original_prop, noise_types, num_applications)
    
    return {
        'original_prop': original_prop,
        'noisy_prop': noisy_prop,
        'target_contrapositive': target_contrapositive,
        'complexity': 'recursive' if 'max_depth' in kwargs else 'simple',
        'noise_applications': num_applications,
        'noise_types': noise_types
    }


def generate_dataset_from_config(dataset_config: Dict) -> List[Dict]:
    """
    根据配置生成数据集
    实现您建议的配置驱动方法
    """
    print(f"生成数据集: {dataset_config.get('description', '未知')}")
    
    # 获取生成器函数
    generator_func = get_generator_function(dataset_config['generator_func'])
    
    # 提取参数
    num_samples = dataset_config['num_samples']
    noise_types = dataset_config['noise_types']
    num_applications = dataset_config.get('noise_applications', 1)
    max_depth = dataset_config.get('max_depth', 3)
    
    dataset = []
    successful_samples = 0
    attempts = 0
    max_attempts = num_samples * 3
    
    while successful_samples < num_samples and attempts < max_attempts:
        attempts += 1
        
        try:
            sample = generate_training_sample(
                generator_func=generator_func,
                noise_types=noise_types,
                num_applications=num_applications,
                max_depth=max_depth
            )
            
            # 基本验证：确保生成的样本不为空且不相同
            if (sample['noisy_prop'].strip() and 
                sample['target_contrapositive'].strip() and
                sample['noisy_prop'] != sample['target_contrapositive']):
                
                dataset.append(sample)
                successful_samples += 1
                
                # 每生成1000个样本打印进度
                if successful_samples % 1000 == 0:
                    print(f"  已生成 {successful_samples}/{num_samples} 个样本...")
                    
        except Exception as e:
            # print(f"生成样本时出错: {e}")
            continue
    
    print(f"  ✅ 成功生成 {successful_samples} 个样本，总尝试次数: {attempts}")
    return dataset


def save_dataset_optimized(dataset: List[Dict], filename: str):
    """
    优化的数据集保存函数
    实现您建议的批量写入优化
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 批量准备JSON字符串
    json_lines = [json.dumps(sample, ensure_ascii=False) + '\n' for sample in dataset]
    
    # 一次性写入
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(json_lines)
    
    print(f"  ✅ 数据集已保存到 {filename}")


def analyze_dataset_advanced(dataset: List[Dict], name: str = "数据集") -> Dict:
    """
    高级数据集分析
    包含复杂度和噪声类型的详细统计
    """
    if not dataset:
        return {"error": "数据集为空"}
    
    total_samples = len(dataset)
    
    # 复杂度分布
    complexity_count = {}
    for sample in dataset:
        complexity = sample.get('complexity', 'unknown')
        complexity_count[complexity] = complexity_count.get(complexity, 0) + 1
    
    # 噪声应用次数分布
    noise_app_count = {}
    for sample in dataset:
        apps = sample.get('noise_applications', 1)
        noise_app_count[apps] = noise_app_count.get(apps, 0) + 1
    
    # 噪声类型分布
    noise_type_count = {}
    for sample in dataset:
        types = sample.get('noise_types', ['type1'])
        for noise_type in types:
            noise_type_count[noise_type] = noise_type_count.get(noise_type, 0) + 1
    
    # 长度分布
    input_lengths = [len(sample['noisy_prop']) for sample in dataset]
    target_lengths = [len(sample['target_contrapositive']) for sample in dataset]
    
    stats = {
        "name": name,
        "total_samples": total_samples,
        "complexity_distribution": complexity_count,
        "noise_applications_distribution": noise_app_count,
        "noise_types_distribution": noise_type_count,
        "avg_input_length": round(sum(input_lengths) / len(input_lengths), 2),
        "avg_target_length": round(sum(target_lengths) / len(target_lengths), 2),
        "max_input_length": max(input_lengths),
        "max_target_length": max(target_lengths)
    }
    
    print(f"\n📊 {name} 详细分析:")
    print(f"  总样本数: {total_samples}")
    print(f"  复杂度分布: {complexity_count}")
    print(f"  噪声应用次数: {noise_app_count}")
    print(f"  噪声类型分布: {noise_type_count}")
    print(f"  平均输入长度: {stats['avg_input_length']}")
    print(f"  平均目标长度: {stats['avg_target_length']}")
    
    return stats


def show_sample_examples(dataset: List[Dict], num_examples: int = 3):
    """显示数据集样本示例"""
    print(f"\n📝 样本示例 (前{num_examples}个):")
    
    for i, sample in enumerate(dataset[:num_examples]):
        print(f"\n  样本 {i+1}:")
        print(f"    原始命题: {sample['original_prop']}")
        print(f"    噪声命题: {sample['noisy_prop']}")
        print(f"    目标输出: {sample['target_contrapositive']}")
        print(f"    复杂度: {sample.get('complexity', 'unknown')}")
        print(f"    噪声次数: {sample.get('noise_applications', 1)}")


def generate_curriculum_datasets(config: Dict):
    """
    生成课程学习数据集
    实现您建议的从易到难的学习策略
    """
    print("🎓 开始生成课程学习数据集...")
    print("=" * 60)
    
    datasets_config = config['datasets']
    curriculum_config = config.get('curriculum_learning', {})
    
    if curriculum_config.get('enabled', False):
        print("📚 课程学习模式已启用")
        stages = curriculum_config.get('stages', [])
        
        for stage in stages:
            print(f"\n🎯 {stage['name']}:")
            for dataset_name in stage['datasets']:
                if dataset_name in datasets_config:
                    dataset_config = datasets_config[dataset_name]
                    dataset = generate_dataset_from_config(dataset_config)
                    save_dataset_optimized(dataset, dataset_config['output_file'])
                    analyze_dataset_advanced(dataset, dataset_name)
                    show_sample_examples(dataset, 2)
    else:
        print("📖 标准模式：生成所有数据集")
        for dataset_name, dataset_config in datasets_config.items():
            print(f"\n📁 生成 {dataset_name}:")
            dataset = generate_dataset_from_config(dataset_config)
            save_dataset_optimized(dataset, dataset_config['output_file'])
            analyze_dataset_advanced(dataset, dataset_name)


def main():
    """
    主函数：配置驱动的数据生成
    实现您建议的所有优化
    """
    print("🚀 高级自偏移推理训练数据集生成器")
    print("=" * 60)
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    # 加载配置
    config = load_config()
    if config is None:
        print("❌ 无法加载配置文件，退出程序")
        return
    
    # 生成数据集
    generate_curriculum_datasets(config)
    
    print("\n🎉 所有数据集生成完成！")
    print("\n📋 生成的文件:")
    for dataset_name, dataset_config in config['datasets'].items():
        output_file = dataset_config['output_file']
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024 / 1024  # MB
            print(f"  ✅ {output_file} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()
