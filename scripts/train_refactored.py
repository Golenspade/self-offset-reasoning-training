"""
重构后的训练脚本
使用配置文件和新的包结构
"""

import json
import numpy as np
import time
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset, analyze_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel


def load_config(config_path='configs/default_config.json'):
    """加载配置文件"""
    config_file = project_root / config_path
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 配置文件加载成功: {config_file}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None


def train_model(model, train_data, val_data, tokenizer, config):
    """
    训练模型的主函数
    """
    training_config = config['training']
    
    epochs = training_config['epochs']
    batch_size = training_config['batch_size']
    validation_frequency = training_config['validation_frequency']
    
    print(f"开始训练模型...")
    print(f"训练样本数: {len(train_data)}")
    print(f"验证样本数: {len(val_data)}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    
    best_accuracy = 0
    training_history = []
    
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
        val_accuracy = None
        if epoch % validation_frequency == 0:
            val_accuracy = evaluate_model_quick(model, val_data[:50], tokenizer)
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"*** 新的最佳准确率: {val_accuracy:.2%} ***")
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"损失={avg_loss:.4f}, "
              f"验证准确率={val_accuracy:.2%} " if val_accuracy is not None else f"损失={avg_loss:.4f}, " +
              f"时间={epoch_time:.1f}s")
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'val_accuracy': val_accuracy,
            'time': epoch_time
        })
    
    return training_history, best_accuracy


def evaluate_model_quick(model, data, tokenizer):
    """快速评估模型（用于训练过程中的验证）"""
    correct = 0
    total = len(data)
    
    for sample in data:
        predicted_tokens = model.predict(sample['input'], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()
        target_text = sample['target_text'].strip()
        
        # 简单的字符串匹配
        if predicted_text == target_text:
            correct += 1
    
    return correct / total if total > 0 else 0


def save_outputs(model, training_history, config):
    """保存训练输出"""
    paths = config['paths']
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(paths['model_save_path']), exist_ok=True)
    os.makedirs(os.path.dirname(paths['training_history_path']), exist_ok=True)
    
    # 保存模型
    model_path = project_root / paths['model_save_path']
    model.save_model(str(model_path))
    
    # 保存训练历史
    history_path = project_root / paths['training_history_path']
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练历史已保存到: {history_path}")
    print(f"✅ 模型已保存到: {model_path}")


def main():
    """主训练函数"""
    print("=== 自偏移推理训练 - 重构版训练流程 ===\n")
    
    # 加载配置
    config = load_config()
    if config is None:
        print("❌ 无法加载配置文件，退出训练")
        return
    
    # 设置随机种子
    np.random.seed(42)
    
    # 初始化tokenizer
    tokenizer = Tokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 加载数据
    print("\n加载数据...")
    data_config = config['data']
    
    train_path = project_root / data_config['train_path']
    val_path = project_root / data_config['val_path']
    
    train_data = load_dataset(str(train_path), tokenizer, data_config['max_train_samples'])
    val_data = load_dataset(str(val_path), tokenizer, data_config['max_val_samples'])
    
    if not train_data or not val_data:
        print("❌ 数据加载失败，退出训练")
        return
    
    # 分析数据
    analyze_dataset(train_data, "训练集")
    analyze_dataset(val_data, "验证集")
    
    # 创建模型
    print("\n创建模型...")
    model_config = config['model']
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_config['hidden_size'],
        max_length=model_config['max_length'],
        learning_rate=model_config['learning_rate']
    )
    
    # 训练模型
    print("\n=== 开始训练 ===")
    history, best_accuracy = train_model(model, train_data, val_data, tokenizer, config)
    
    # 保存输出
    print("\n=== 保存训练结果 ===")
    save_outputs(model, history, config)
    
    # 训练总结
    print(f"\n=== 训练总结 ===")
    print(f"最佳验证准确率: {best_accuracy:.2%}")
    print(f"训练完成！")


if __name__ == "__main__":
    main()
