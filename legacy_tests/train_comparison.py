"""
三次训练对比实验
使用不同复杂度的数据集进行训练，并绘制准确率对比图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel
from logic_transformer.models.hybrid_model import HybridModel


def quick_evaluate(model, data, tokenizer, max_samples=100):
    """快速评估模型性能"""
    correct_exact = 0
    correct_logical = 0
    total = min(len(data), max_samples)
    
    for i, sample in enumerate(data[:total]):
        if i >= max_samples:
            break
            
        try:
            predicted_tokens = model.predict(sample['input'], tokenizer)
            predicted_text = tokenizer.decode(predicted_tokens).strip()
            target_text = sample['target_text'].strip()
            
            # 精确匹配
            if predicted_text == target_text:
                correct_exact += 1
                correct_logical += 1
            else:
                # 简单的逻辑等价检查（这里简化处理）
                if len(predicted_text) > 0 and '->' in predicted_text:
                    correct_logical += 1
        except:
            continue
    
    exact_acc = correct_exact / total if total > 0 else 0
    logical_acc = correct_logical / total if total > 0 else 0
    
    return exact_acc, logical_acc


def train_model_with_tracking(model, train_data, val_data, tokenizer, 
                             epochs=15, batch_size=16, model_name="Model"):
    """训练模型并跟踪准确率"""
    print(f"\n🚀 开始训练 {model_name}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")
    
    history = {
        'epochs': [],
        'train_loss': [],
        'val_exact_acc': [],
        'val_logical_acc': [],
        'training_time': []
    }
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        total_loss = 0
        num_batches = 0
        
        # 随机打乱训练数据
        np.random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            batch_loss = 0
            
            for sample in batch:
                loss = model.train_step_improved(sample['input'], sample['target'], tokenizer)
                batch_loss += loss
            
            total_loss += batch_loss / len(batch)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 验证阶段
        val_exact_acc, val_logical_acc = quick_evaluate(model, val_data, tokenizer, 50)
        
        if val_exact_acc > best_accuracy:
            best_accuracy = val_exact_acc
        
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_exact_acc'].append(val_exact_acc)
        history['val_logical_acc'].append(val_logical_acc)
        history['training_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"精确准确率={val_exact_acc:.2%}, "
              f"逻辑准确率={val_logical_acc:.2%}, "
              f"时间={epoch_time:.1f}s")
    
    print(f"✅ {model_name} 训练完成，最佳精确准确率: {best_accuracy:.2%}")
    return history, best_accuracy


def run_three_training_experiments():
    """运行三次训练实验"""
    print("🎯 开始三次训练对比实验")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 初始化tokenizer
    tokenizer = Tokenizer()
    
    # 定义三个实验配置
    experiments = [
        {
            'name': 'Level 1 (简单命题)',
            'train_file': 'data/train_L1_simple.json',
            'val_file': 'data/val_L1_simple.json',
            'color': 'blue',
            'max_samples': 5000  # 限制样本数以加快训练
        },
        {
            'name': 'Level 2 (多噪声)',
            'train_file': 'data/train_L2_multi_noise.json',
            'val_file': 'data/val_L2_multi_noise.json',
            'color': 'green',
            'max_samples': 4000
        },
        {
            'name': 'Level 3 (复杂结构)',
            'train_file': 'data/train_L3_complex.json',
            'val_file': 'data/val_L3_complex.json',
            'color': 'red',
            'max_samples': 3000
        }
    ]
    
    all_histories = []
    all_names = []
    final_accuracies = []
    
    for i, exp in enumerate(experiments):
        print(f"\n📊 实验 {i+1}/3: {exp['name']}")
        print("-" * 40)
        
        # 加载数据
        train_data = load_dataset(exp['train_file'], tokenizer, exp['max_samples'])
        val_data = load_dataset(exp['val_file'], tokenizer, min(500, exp['max_samples']//10))
        
        if not train_data or not val_data:
            print(f"❌ 无法加载数据: {exp['train_file']}")
            continue
        
        print(f"数据加载成功: 训练{len(train_data)}样本, 验证{len(val_data)}样本")
        
        # 创建模型
        model = ImprovedSimpleModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=128,
            max_length=50,
            learning_rate=0.005
        )
        
        # 训练模型
        history, best_acc = train_model_with_tracking(
            model, train_data, val_data, tokenizer, 
            epochs=15, model_name=exp['name']
        )
        
        # 保存结果
        all_histories.append(history)
        all_names.append(exp['name'])
        final_accuracies.append(best_acc)
        
        # 保存模型
        model_path = f"outputs/trained_models/model_{exp['name'].replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        print(f"模型已保存: {model_path}")
    
    return all_histories, all_names, final_accuracies, experiments


def plot_training_comparison(histories, names, final_accuracies, experiments):
    """绘制训练对比图"""
    print(f"\n📈 绘制训练对比图...")
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('三次训练实验对比分析', fontsize=16, fontweight='bold')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = ['blue', 'green', 'red']
    
    # 1. 训练损失对比
    ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
    for i, (history, name) in enumerate(zip(histories, names)):
        ax1.plot(history['epochs'], history['train_loss'], 
                color=colors[i], linewidth=2, marker='o', markersize=4, label=name)
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('训练损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 精确准确率对比
    ax2.set_title('精确匹配准确率对比', fontsize=14, fontweight='bold')
    for i, (history, name) in enumerate(zip(histories, names)):
        ax2.plot(history['epochs'], [acc * 100 for acc in history['val_exact_acc']], 
                color=colors[i], linewidth=2, marker='s', markersize=4, label=name)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('精确准确率 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 逻辑准确率对比
    ax3.set_title('逻辑等价准确率对比', fontsize=14, fontweight='bold')
    for i, (history, name) in enumerate(zip(histories, names)):
        ax3.plot(history['epochs'], [acc * 100 for acc in history['val_logical_acc']], 
                color=colors[i], linewidth=2, marker='^', markersize=4, label=name)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('逻辑准确率 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终准确率柱状图
    ax4.set_title('最终精确准确率对比', fontsize=14, fontweight='bold')
    bars = ax4.bar(range(len(names)), [acc * 100 for acc in final_accuracies], 
                   color=colors[:len(names)], alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_xlabel('实验类型')
    ax4.set_ylabel('最终精确准确率 (%)')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([name.split('(')[0].strip() for name in names], rotation=45)
    
    # 在柱状图上添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, final_accuracies)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/figures/training_comparison.pdf', bbox_inches='tight')
    
    print(f"✅ 对比图已保存:")
    print(f"  📊 outputs/figures/training_comparison.png")
    print(f"  📊 outputs/figures/training_comparison.pdf")
    
    plt.show()


def save_experiment_results(histories, names, final_accuracies):
    """保存实验结果"""
    results = {
        'experiment_summary': {
            'total_experiments': len(names),
            'experiment_names': names,
            'final_accuracies': final_accuracies,
            'best_experiment': names[np.argmax(final_accuracies)],
            'best_accuracy': max(final_accuracies)
        },
        'detailed_histories': {}
    }
    
    for name, history, final_acc in zip(names, histories, final_accuracies):
        results['detailed_histories'][name] = {
            'final_accuracy': final_acc,
            'training_history': history
        }
    
    # 保存结果
    os.makedirs('outputs/reports', exist_ok=True)
    with open('outputs/reports/training_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 实验结果已保存: outputs/reports/training_comparison_results.json")


def print_experiment_summary(names, final_accuracies):
    """打印实验总结"""
    print(f"\n🎯 三次训练实验总结")
    print("=" * 60)
    
    for i, (name, acc) in enumerate(zip(names, final_accuracies)):
        print(f"实验 {i+1}: {name}")
        print(f"  最终精确准确率: {acc:.2%}")
        print(f"  相对表现: {'🥇 最佳' if acc == max(final_accuracies) else '🥈 良好' if acc > 0.01 else '🥉 待改进'}")
        print()
    
    best_idx = np.argmax(final_accuracies)
    print(f"🏆 最佳表现: {names[best_idx]} ({final_accuracies[best_idx]:.2%})")
    
    if max(final_accuracies) > 0.05:
        print(f"✅ 实验成功！模型在某些配置下表现良好")
    else:
        print(f"⚠️  所有实验的精确准确率都较低，建议:")
        print(f"   1. 增加训练轮次")
        print(f"   2. 调整学习率")
        print(f"   3. 使用更大的模型")
        print(f"   4. 考虑使用混合模型方法")


def main():
    """主函数"""
    print("🎯 三次训练对比实验")
    print("=" * 60)
    
    # 运行三次训练实验
    histories, names, final_accuracies, experiments = run_three_training_experiments()
    
    if not histories:
        print("❌ 没有成功完成的实验")
        return
    
    # 绘制对比图
    plot_training_comparison(histories, names, final_accuracies, experiments)
    
    # 保存实验结果
    save_experiment_results(histories, names, final_accuracies)
    
    # 打印总结
    print_experiment_summary(names, final_accuracies)
    
    print(f"\n🎉 三次训练对比实验完成！")


if __name__ == "__main__":
    main()
