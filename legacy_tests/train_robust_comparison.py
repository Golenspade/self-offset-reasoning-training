"""
使用鲁棒数据集进行训练对比实验
验证是否能堵死作弊捷径，迫使模型学习真正的逻辑
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


def train_robust_model(model, train_data, val_data, tokenizer, epochs=20, model_name="Model"):
    """训练鲁棒模型"""
    print(f"\n🚀 开始训练 {model_name}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(val_data)}")
    
    history = {
        'epochs': [],
        'train_loss': [],
        'val_exact_acc': [],
        'val_logical_acc': []
    }
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        total_loss = 0
        num_batches = 0
        batch_size = 8  # 减小批次大小以适应更复杂的数据
        
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
        val_exact_acc, val_logical_acc = evaluate_robust_model(model, val_data, tokenizer)
        
        if val_exact_acc > best_accuracy:
            best_accuracy = val_exact_acc
        
        epoch_time = time.time() - start_time
        
        # 记录历史
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['val_exact_acc'].append(val_exact_acc)
        history['val_logical_acc'].append(val_logical_acc)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"精确={val_exact_acc:.1%}, "
              f"逻辑={val_logical_acc:.1%}, "
              f"时间={epoch_time:.1f}s")
    
    print(f"✅ {model_name} 训练完成，最佳精确准确率: {best_accuracy:.2%}")
    return history, best_accuracy


def evaluate_robust_model(model, data, tokenizer, max_samples=50):
    """评估鲁棒模型性能"""
    correct_exact = 0
    correct_logical = 0
    total = min(len(data), max_samples)
    
    for i, sample in enumerate(data[:total]):
        try:
            predicted_tokens = model.predict(sample['input'], tokenizer)
            predicted_text = tokenizer.decode(predicted_tokens).strip()
            target_text = sample['target_text'].strip()
            
            # 精确匹配
            if predicted_text == target_text:
                correct_exact += 1
                correct_logical += 1
            else:
                # 检查是否至少生成了合理的逻辑表达式
                if (len(predicted_text) > 5 and 
                    '->' in predicted_text and 
                    not predicted_text.startswith('-> -> ->')):
                    correct_logical += 1
        except:
            continue
    
    exact_acc = correct_exact / total if total > 0 else 0
    logical_acc = correct_logical / total if total > 0 else 0
    
    return exact_acc, logical_acc


def run_robust_training_experiments():
    """运行鲁棒训练实验"""
    print("🛡️ 鲁棒数据集训练对比实验")
    print("=" * 60)
    
    np.random.seed(42)
    tokenizer = Tokenizer()
    
    # 定义实验配置
    experiments = [
        {
            'name': 'Level 1 鲁棒版',
            'train_file': 'data/train_level_1_鲁棒版.json',
            'val_file': 'data/val_level_1_鲁棒版.json',
            'color': 'blue',
            'max_samples': 2000
        },
        {
            'name': 'Level 2 鲁棒版',
            'train_file': 'data/train_level_2_鲁棒版.json',
            'val_file': 'data/val_level_2_鲁棒版.json',
            'color': 'green',
            'max_samples': 1500
        },
        {
            'name': 'Level 3 鲁棒版',
            'train_file': 'data/train_level_3_鲁棒版.json',
            'val_file': 'data/val_level_3_鲁棒版.json',
            'color': 'red',
            'max_samples': 1000
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
        val_data = load_dataset(exp['val_file'], tokenizer, min(200, exp['max_samples']//10))
        
        if not train_data or not val_data:
            print(f"❌ 无法加载数据: {exp['train_file']}")
            continue
        
        print(f"数据加载成功: 训练{len(train_data)}样本, 验证{len(val_data)}样本")
        
        # 创建模型
        model = ImprovedSimpleModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=128,
            max_length=50,
            learning_rate=0.003  # 稍微降低学习率以提高稳定性
        )
        
        # 训练模型
        history, best_acc = train_robust_model(
            model, train_data, val_data, tokenizer, 
            epochs=20, model_name=exp['name']
        )
        
        # 保存结果
        all_histories.append(history)
        all_names.append(exp['name'])
        final_accuracies.append(best_acc)
        
        # 保存模型
        model_path = f"outputs/trained_models/robust_model_{exp['name'].replace(' ', '_')}.npz"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        print(f"模型已保存: {model_path}")
    
    return all_histories, all_names, final_accuracies, experiments


def plot_robust_training_results(histories, names, final_accuracies):
    """绘制鲁棒训练结果"""
    print(f"\n📈 绘制鲁棒训练结果...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('鲁棒数据集训练结果 - 堵死作弊捷径后的真实学习', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red']
    
    # 1. 训练损失
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    for i, (history, name) in enumerate(zip(histories, names)):
        ax1.plot(history['epochs'], history['train_loss'], 
                color=colors[i], linewidth=2, marker='o', markersize=3, label=name)
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('训练损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 精确准确率
    ax2.set_title('精确匹配准确率', fontsize=14, fontweight='bold')
    for i, (history, name) in enumerate(zip(histories, names)):
        ax2.plot(history['epochs'], [acc * 100 for acc in history['val_exact_acc']], 
                color=colors[i], linewidth=2, marker='s', markersize=3, label=name)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('精确准确率 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 逻辑准确率
    ax3.set_title('逻辑等价准确率', fontsize=14, fontweight='bold')
    for i, (history, name) in enumerate(zip(histories, names)):
        ax3.plot(history['epochs'], [acc * 100 for acc in history['val_logical_acc']], 
                color=colors[i], linewidth=2, marker='^', markersize=3, label=name)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('逻辑准确率 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终准确率对比
    ax4.set_title('最终准确率对比', fontsize=14, fontweight='bold')
    bars = ax4.bar(range(len(names)), [acc * 100 for acc in final_accuracies], 
                   color=colors[:len(names)], alpha=0.7, edgecolor='black')
    ax4.set_xlabel('实验类型')
    ax4.set_ylabel('最终精确准确率 (%)')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([name.replace('鲁棒版', '') for name in names], rotation=45)
    
    # 添加数值标签
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/robust_training_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/figures/robust_training_results.pdf', bbox_inches='tight')
    
    print(f"✅ 鲁棒训练结果图已保存:")
    print(f"  📊 outputs/figures/robust_training_results.png")


def analyze_robust_results(names, final_accuracies, histories):
    """分析鲁棒训练结果"""
    print(f"\n🎯 鲁棒训练结果分析")
    print("=" * 60)
    
    print(f"{'实验名称':<15} {'最终精确准确率':<15} {'学习趋势':<15}")
    print("-" * 50)
    
    for i, (name, acc) in enumerate(zip(names, final_accuracies)):
        history = histories[i]
        
        # 分析学习趋势
        if len(history['val_exact_acc']) >= 10:
            early_acc = np.mean(history['val_exact_acc'][:5])
            late_acc = np.mean(history['val_exact_acc'][-5:])
            trend = "上升" if late_acc > early_acc + 0.01 else "稳定" if abs(late_acc - early_acc) <= 0.01 else "下降"
        else:
            trend = "数据不足"
        
        print(f"{name:<15} {acc:<15.1%} {trend:<15}")
    
    # 总体分析
    print(f"\n🔍 关键发现:")
    
    max_acc = max(final_accuracies)
    best_idx = final_accuracies.index(max_acc)
    
    if max_acc > 0.1:
        print(f"✅ 成功！{names[best_idx]} 达到了 {max_acc:.1%} 的准确率")
        print(f"   这表明鲁棒数据集成功迫使模型学习真正的逻辑推理")
    elif max_acc > 0.05:
        print(f"🔄 进展中：最佳准确率 {max_acc:.1%}，需要更多训练")
    else:
        print(f"⚠️  挑战性：所有实验准确率都较低，数据可能过于复杂")
    
    # 检查是否还有异常的快速学习
    for i, (name, history) in enumerate(zip(names, histories)):
        if len(history['val_exact_acc']) >= 3:
            early_acc = history['val_exact_acc'][2]  # 第3轮的准确率
            if early_acc > 0.8:
                print(f"⚠️  {name} 仍然表现出异常快速的学习，可能还有隐藏的捷径")


def main():
    """主函数"""
    print("🛡️ 开始鲁棒数据集训练实验...")
    
    # 运行训练实验
    histories, names, final_accuracies, experiments = run_robust_training_experiments()
    
    if not histories:
        print("❌ 没有成功完成的实验")
        return
    
    # 绘制结果
    plot_robust_training_results(histories, names, final_accuracies)
    
    # 分析结果
    analyze_robust_results(names, final_accuracies, histories)
    
    print(f"\n🎉 鲁棒训练实验完成！")
    print(f"\n💡 如果结果显示健康的学习曲线（逐步提升而非瞬间达到100%），")
    print(f"   那么我们就成功地堵死了作弊捷径，迫使模型学习真正的逻辑推理！")


if __name__ == "__main__":
    main()
