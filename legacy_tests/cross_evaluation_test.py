"""
交叉评估测试：验证Level 3模型是否真的学会了逻辑推理
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel


def load_trained_model(model_path, tokenizer):
    """加载训练好的模型"""
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        max_length=50,
        learning_rate=0.005
    )
    
    if model.load_model(model_path):
        print(f"✅ 成功加载模型: {model_path}")
        return model
    else:
        print(f"❌ 无法加载模型: {model_path}")
        return None


def evaluate_cross_performance(model, data, tokenizer, data_name):
    """评估模型在不同数据集上的性能"""
    if not data:
        return 0, 0
    
    correct_exact = 0
    correct_logical = 0
    total = len(data)
    
    print(f"\n🧪 在{data_name}上评估模型性能...")
    print(f"样本数量: {total}")
    
    # 显示前几个预测示例
    print(f"\n📝 预测示例:")
    for i, sample in enumerate(data[:5]):
        try:
            predicted_tokens = model.predict(sample['input'], tokenizer)
            predicted_text = tokenizer.decode(predicted_tokens).strip()
            target_text = sample['target_text'].strip()
            
            print(f"\n  样本 {i+1}:")
            print(f"    输入: {sample['input_text']}")
            print(f"    目标: {target_text}")
            print(f"    预测: {predicted_text}")
            print(f"    匹配: {'✓' if predicted_text == target_text else '✗'}")
            
        except Exception as e:
            print(f"    预测出错: {e}")
    
    # 计算整体准确率
    for sample in data:
        try:
            predicted_tokens = model.predict(sample['input'], tokenizer)
            predicted_text = tokenizer.decode(predicted_tokens).strip()
            target_text = sample['target_text'].strip()
            
            # 精确匹配
            if predicted_text == target_text:
                correct_exact += 1
                correct_logical += 1
            else:
                # 简化的逻辑等价检查
                if len(predicted_text) > 0 and '->' in predicted_text:
                    correct_logical += 1
        except:
            continue
    
    exact_acc = correct_exact / total if total > 0 else 0
    logical_acc = correct_logical / total if total > 0 else 0
    
    return exact_acc, logical_acc


def run_cross_evaluation():
    """运行交叉评估实验"""
    print("🔬 Level 3模型交叉评估实验")
    print("=" * 60)
    
    # 初始化tokenizer
    tokenizer = Tokenizer()
    
    # 加载Level 3训练的模型
    l3_model_path = "outputs/trained_models/model_Level_3_复杂结构.npz"
    l3_model = load_trained_model(l3_model_path, tokenizer)
    
    if l3_model is None:
        print("❌ 无法加载Level 3模型，退出测试")
        return
    
    # 加载不同级别的验证数据
    datasets = {
        "Level 1 (简单命题)": "data/val_L1_simple.json",
        "Level 2 (多噪声)": "data/val_L2_multi_noise.json", 
        "Level 3 (复杂结构)": "data/val_L3_complex.json"
    }
    
    results = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"📊 评估 {dataset_name}")
        print(f"{'='*60}")
        
        # 加载数据
        data = load_dataset(dataset_path, tokenizer, 100)  # 限制100个样本
        
        if not data:
            print(f"❌ 无法加载数据: {dataset_path}")
            continue
        
        # 评估性能
        exact_acc, logical_acc = evaluate_cross_performance(l3_model, data, tokenizer, dataset_name)
        
        results[dataset_name] = {
            'exact_accuracy': exact_acc,
            'logical_accuracy': logical_acc,
            'sample_count': len(data)
        }
        
        print(f"\n📈 {dataset_name} 结果:")
        print(f"  精确匹配准确率: {exact_acc:.2%}")
        print(f"  逻辑准确率: {logical_acc:.2%}")
    
    return results


def analyze_cross_evaluation_results(results):
    """分析交叉评估结果"""
    print(f"\n🎯 交叉评估结果分析")
    print("=" * 60)
    
    if not results:
        print("❌ 没有可分析的结果")
        return
    
    print(f"{'数据集':<20} {'精确准确率':<12} {'逻辑准确率':<12} {'样本数':<8}")
    print("-" * 60)
    
    for dataset_name, result in results.items():
        print(f"{dataset_name:<20} {result['exact_accuracy']:<12.1%} "
              f"{result['logical_accuracy']:<12.1%} {result['sample_count']:<8}")
    
    # 分析结论
    print(f"\n🔍 分析结论:")
    
    l3_acc = results.get("Level 3 (复杂结构)", {}).get('exact_accuracy', 0)
    l1_acc = results.get("Level 1 (简单命题)", {}).get('exact_accuracy', 0)
    l2_acc = results.get("Level 2 (多噪声)", {}).get('exact_accuracy', 0)
    
    if l3_acc > 0.8 and l1_acc < 0.1:
        print("🚨 **确认作弊行为！**")
        print("   Level 3模型在复杂数据上表现完美，但在简单数据上完全失败")
        print("   这证明模型学到的是特定于Level 3数据的捷径，而非通用逻辑")
        
    elif l3_acc > 0.8 and l1_acc > 0.3:
        print("✅ **模型可能学到了一些通用规律**")
        print("   虽然在Level 3上表现最好，但在其他数据上也有合理表现")
        
    elif l1_acc > l3_acc:
        print("🤔 **意外结果**")
        print("   模型在简单数据上表现更好，这可能表明训练过程有问题")
        
    else:
        print("📊 **需要更多分析**")
        print("   结果不够明确，建议增加样本数量或检查其他因素")
    
    # 给出具体建议
    print(f"\n💡 下一步建议:")
    if l3_acc > 0.8 and l1_acc < 0.1:
        print("1. 立即重新设计Level 3数据生成策略")
        print("2. 增加噪声的多样性和复杂度")
        print("3. 确保不同复杂度数据之间的一致性")
        print("4. 重新生成数据并重新训练")
    else:
        print("1. 继续训练更多轮次")
        print("2. 调整学习率和训练策略")
        print("3. 考虑课程学习方法")


def main():
    """主函数"""
    print("🔬 开始Level 3模型的交叉评估测试...")
    
    # 运行交叉评估
    results = run_cross_evaluation()
    
    # 分析结果
    analyze_cross_evaluation_results(results)
    
    print(f"\n🎉 交叉评估测试完成！")


if __name__ == "__main__":
    main()
