"""
文件名: evaluate.py
评估脚本
用于全面评估训练好的模型性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from logic_utils import Tokenizer, verify_equivalence, to_contrapositive
from simple_model import load_data
from train import ImprovedSimpleModel


def analyze_predictions(model, data, tokenizer, num_samples=50):
    """
    分析模型预测的详细情况
    """
    print(f"\n=== 预测分析 (分析 {num_samples} 个样本) ===")
    
    categories = {
        'perfect_match': [],      # 完全匹配
        'logical_equivalent': [], # 逻辑等价但不完全匹配
        'partial_correct': [],    # 部分正确
        'completely_wrong': []    # 完全错误
    }
    
    for i, sample in enumerate(data[:num_samples]):
        predicted_tokens = model.predict(sample['input'], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()
        target_text = sample['target_text'].strip()
        input_text = sample['input_text'].strip()
        
        # 分类预测结果
        if predicted_text == target_text:
            categories['perfect_match'].append((input_text, target_text, predicted_text))
        elif verify_equivalence(predicted_text, target_text):
            categories['logical_equivalent'].append((input_text, target_text, predicted_text))
        elif any(token in predicted_text for token in ['~', '->', '&', '|']):
            categories['partial_correct'].append((input_text, target_text, predicted_text))
        else:
            categories['completely_wrong'].append((input_text, target_text, predicted_text))
    
    # 打印分析结果
    for category, samples in categories.items():
        print(f"\n{category.replace('_', ' ').title()}: {len(samples)} 样本")
        
        # 显示每个类别的前几个例子
        for j, (inp, target, pred) in enumerate(samples[:3]):
            print(f"  例子 {j+1}:")
            print(f"    输入: {inp}")
            print(f"    目标: {target}")
            print(f"    预测: {pred}")
    
    return categories


def test_specific_patterns(model, tokenizer):
    """
    测试模型对特定逻辑模式的处理能力
    """
    print(f"\n=== 特定模式测试 ===")
    
    test_cases = [
        # 简单否定
        ("(~p | q)", "p -> q", "~q -> ~p"),
        ("(~q | r)", "q -> r", "~r -> ~q"),
        
        # 双重否定
        ("(p | ~~q)", "~p -> ~~q", "~q -> p"),
        
        # 复杂表达式
        ("(~p | ~q)", "p -> ~q", "q -> ~p"),
        ("(~~p | q)", "~p -> q", "~q -> p"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, (noisy_input, original, expected_output) in enumerate(test_cases):
        input_tokens = tokenizer.encode(noisy_input)
        predicted_tokens = model.predict(input_tokens, tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()
        
        is_correct = (predicted_text == expected_output or 
                     verify_equivalence(predicted_text, expected_output))
        
        if is_correct:
            correct += 1
        
        print(f"\n测试 {i+1}:")
        print(f"  噪声输入: {noisy_input}")
        print(f"  原始命题: {original}")
        print(f"  期望输出: {expected_output}")
        print(f"  模型预测: {predicted_text}")
        print(f"  结果: {'✓' if is_correct else '✗'}")
    
    accuracy = correct / total
    print(f"\n特定模式测试准确率: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


def error_analysis(model, data, tokenizer, num_samples=100):
    """
    错误分析：找出模型常犯的错误类型
    """
    print(f"\n=== 错误分析 ===")
    
    error_types = {
        'wrong_negation': 0,      # 否定错误
        'wrong_direction': 0,     # 方向错误 (A->B vs B->A)
        'missing_symbols': 0,     # 缺少符号
        'extra_symbols': 0,       # 多余符号
        'format_error': 0,        # 格式错误
        'other': 0               # 其他错误
    }
    
    total_errors = 0
    
    for sample in data[:num_samples]:
        predicted_tokens = model.predict(sample['input'], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()
        target_text = sample['target_text'].strip()
        
        if predicted_text != target_text:
            total_errors += 1
            
            # 分析错误类型
            if '~' in target_text and '~' not in predicted_text:
                error_types['missing_symbols'] += 1
            elif '~' not in target_text and '~' in predicted_text:
                error_types['extra_symbols'] += 1
            elif '->' not in predicted_text:
                error_types['format_error'] += 1
            else:
                error_types['other'] += 1
    
    print(f"总错误数: {total_errors}/{num_samples}")
    print("错误类型分布:")
    for error_type, count in error_types.items():
        if total_errors > 0:
            percentage = count / total_errors * 100
            print(f"  {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return error_types


# 删除了装模作样的 benchmark_against_baseline 函数
# 原函数硬编码禁用了规则基线 (original_prop = "")，永远返回0%，完全无效
# 如果需要真实的基线比较，请使用 clean_evaluation_system.py

def benchmark_against_baseline_REMOVED():
    """
    这个函数已被删除，因为它是装模作样的代码：
    - 硬编码 original_prop = ""，规则基线永远是0%
    - 没有提供任何有用的比较信息
    - 请使用 clean_evaluation_system.py 进行真实的基线比较
    """
    print("❌ 此函数已被删除，因为它是装模作样的代码")
    print("💡 请使用 clean_evaluation_system.py 进行真实的评估")
    return 0.0, 0.0


def visualize_training_history():
    """
    可视化训练历史
    """
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses, 'b-', linewidth=2)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 如果有验证准确率数据
        val_accuracies = [h.get('val_accuracy') for h in history if h.get('val_accuracy') is not None]
        val_epochs = [h['epoch'] for h in history if h.get('val_accuracy') is not None]
        
        if val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(val_epochs, val_accuracies, 'r-', linewidth=2)
            plt.title('验证准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练曲线已保存为 training_curves.png")
        
    except FileNotFoundError:
        print("未找到训练历史文件")


def comprehensive_evaluation():
    """
    综合评估函数
    """
    print("=== 自偏移推理训练 - 综合评估 ===\n")
    
    # 初始化
    tokenizer = Tokenizer()
    
    # 创建模型并加载训练好的权重
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        max_length=50,
        learning_rate=0.005
    )

    # 尝试加载训练好的模型权重
    if not model.load_model('trained_model.npz'):
        print("警告: 无法加载训练好的模型，使用随机初始化的权重")
        print("请先运行 train.py 来训练模型")
    
    # 加载数据
    print("加载评估数据...")
    val_data = load_data('data/val.json', tokenizer, max_samples=200)
    print(f"评估数据: {len(val_data)} 样本")
    
    # 1. 基本性能评估
    print("\n1. 基本性能评估")
    correct_exact = 0
    correct_logical = 0
    
    for sample in val_data[:100]:
        predicted_tokens = model.predict(sample['input'], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()
        target_text = sample['target_text'].strip()
        
        if predicted_text == target_text:
            correct_exact += 1
            correct_logical += 1
        elif verify_equivalence(predicted_text, target_text):
            correct_logical += 1
    
    exact_accuracy = correct_exact / 100
    logical_accuracy = correct_logical / 100
    
    print(f"精确匹配准确率: {exact_accuracy:.2%}")
    print(f"逻辑等价准确率: {logical_accuracy:.2%}")
    
    # 2. 预测分析
    categories = analyze_predictions(model, val_data, tokenizer, 50)
    
    # 3. 特定模式测试
    pattern_accuracy = test_specific_patterns(model, tokenizer)
    
    # 4. 错误分析
    error_types = error_analysis(model, val_data, tokenizer, 100)
    
    # 5. 基线比较 (已删除装模作样的函数)
    print("\n⚠️ 原基线比较函数已删除 (装模作样的代码)")
    print("💡 如需真实基线比较，请使用: python3 clean_evaluation_system.py")
    random_acc, rule_acc = 0.0, 0.0  # 占位符
    
    # 6. 可视化训练历史
    visualize_training_history()
    
    # 生成评估报告
    report = {
        'exact_accuracy': exact_accuracy,
        'logical_accuracy': logical_accuracy,
        'pattern_accuracy': pattern_accuracy,
        'random_baseline': random_acc,
        'rule_baseline': rule_acc,
        'error_analysis': error_types,
        'prediction_categories': {k: len(v) for k, v in categories.items()}
    }
    
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== 评估总结 ===")
    print(f"模型在逻辑等价准确率上达到了 {logical_accuracy:.2%}")
    print(f"这比随机基线 ({random_acc:.2%}) 高出 {logical_accuracy - random_acc:.2%}")
    print(f"评估报告已保存到 evaluation_report.json")
    
    return report


if __name__ == "__main__":
    comprehensive_evaluation()
