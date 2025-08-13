"""
文件名: train.py
训练脚本
实现完整的训练循环，支持简化模型和PyTorch模型
"""

import json
import numpy as np
import time
from logic_utils import Tokenizer, verify_equivalence
from simple_model import SimpleSeq2SeqModel, load_data, evaluate_model


class ImprovedSimpleModel(SimpleSeq2SeqModel):
    """
    改进版的简化模型
    添加了更好的训练机制和梯度更新
    """
    
    def __init__(self, vocab_size, hidden_size=64, max_length=50, learning_rate=0.01):
        super().__init__(vocab_size, hidden_size, max_length)
        self.learning_rate = learning_rate
        
        # 添加动量项
        self.embedding_momentum = np.zeros_like(self.embedding)
        self.encoder_weights_momentum = np.zeros_like(self.encoder_weights)
        self.decoder_weights_momentum = np.zeros_like(self.decoder_weights)
        self.output_weights_momentum = np.zeros_like(self.output_weights)
        
        self.momentum = 0.9

    def predict(self, input_sequence, tokenizer, max_length=50):
        """
        预测函数：给定输入序列，生成输出序列
        这是修复后的解码循环实现
        """
        # 编码输入
        encoded = self.encode(input_sequence)

        # 初始化解码序列
        output_sequence = []
        current_token = tokenizer.START_TOKEN

        for step in range(max_length):
            # 解码步骤
            _, output_probs = self.decode_step(encoded, current_token)

            # 选择下一个token (贪婪解码)
            next_token = int(np.argmax(output_probs))

            # 检查终止条件
            if next_token == tokenizer.END_TOKEN:
                break

            # 检查有效性
            if next_token >= tokenizer.vocab_size or next_token < 0:
                next_token = tokenizer.PAD_TOKEN

            # 添加到序列
            output_sequence.append(next_token)
            current_token = next_token

        # 清理输出序列，移除PAD tokens
        cleaned_sequence = [token for token in output_sequence if token != tokenizer.PAD_TOKEN]

        return cleaned_sequence

    def save_model(self, filepath):
        """保存模型权重"""
        model_state = {
            'embedding': self.embedding,
            'encoder_weights': self.encoder_weights,
            'decoder_weights': self.decoder_weights,
            'output_weights': self.output_weights,
            'encoder_bias': self.encoder_bias,
            'decoder_bias': self.decoder_bias,
            'output_bias': self.output_bias,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate
        }

        np.savez(filepath, **model_state)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型权重"""
        try:
            model_state = np.load(filepath)

            self.embedding = model_state['embedding']
            self.encoder_weights = model_state['encoder_weights']
            self.decoder_weights = model_state['decoder_weights']
            self.output_weights = model_state['output_weights']
            self.encoder_bias = model_state['encoder_bias']
            self.decoder_bias = model_state['decoder_bias']
            self.output_bias = model_state['output_bias']

            print(f"模型已从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def train_step_improved(self, input_sequence, target_sequence, tokenizer):
        """
        改进的训练步骤，包含简单的梯度更新
        """
        # 前向传播
        encoded = self.encode(input_sequence)
        
        # 计算损失和简单的梯度
        total_loss = 0
        hidden_state = encoded
        
        # 简单的梯度累积
        embedding_grad = np.zeros_like(self.embedding)
        output_weights_grad = np.zeros_like(self.output_weights)
        
        for i, target_token in enumerate(target_sequence):
            if i == 0:
                current_token = tokenizer.START_TOKEN
            else:
                current_token = target_sequence[i-1]
            
            hidden_state, output_probs = self.decode_step(hidden_state, current_token)
            
            # 计算交叉熵损失
            if target_token < self.vocab_size:
                loss = -np.log(output_probs[target_token] + 1e-8)
                total_loss += loss
                
                # 简单的梯度计算（近似）
                grad = output_probs.copy()
                grad[target_token] -= 1
                
                # 更新输出权重（简化的梯度下降）
                output_weights_grad += np.outer(hidden_state, grad)
        
        # 应用梯度更新
        if len(target_sequence) > 0:
            # 使用动量更新
            self.output_weights_momentum = (self.momentum * self.output_weights_momentum + 
                                          self.learning_rate * output_weights_grad / len(target_sequence))
            self.output_weights -= self.output_weights_momentum
            
            # 添加一些随机性来帮助探索
            if np.random.random() < 0.1:
                self.output_weights += np.random.randn(*self.output_weights.shape) * 0.001
        
        return total_loss / len(target_sequence) if target_sequence else 0


def train_model(model, train_data, val_data, tokenizer, epochs=10, batch_size=32):
    """
    训练模型的主函数
    """
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
                if hasattr(model, 'train_step_improved'):
                    loss = model.train_step_improved(sample['input'], sample['target'], tokenizer)
                else:
                    loss = model.train_step(sample['input'], sample['target'], tokenizer)
                batch_loss += loss
            
            total_loss += batch_loss / len(batch)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 验证阶段
        if epoch % 2 == 0:  # 每2个epoch验证一次
            val_accuracy = evaluate_model_quick(model, val_data[:50], tokenizer)  # 快速验证
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"*** 新的最佳准确率: {val_accuracy:.2%} ***")
        else:
            val_accuracy = None
        
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


def evaluate_model_detailed(model, data, tokenizer, max_samples=100):
    """详细评估模型，包括逻辑等价性检查"""
    correct_exact = 0
    correct_logical = 0
    total = min(len(data), max_samples)
    
    print(f"\n开始详细评估 (评估 {total} 个样本)...")
    
    for i, sample in enumerate(data[:total]):
        predicted_tokens = model.predict(sample['input'], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens).strip()
        target_text = sample['target_text'].strip()
        
        # 精确匹配
        if predicted_text == target_text:
            correct_exact += 1
            correct_logical += 1
        else:
            # 逻辑等价性检查
            try:
                if verify_equivalence(predicted_text, target_text):
                    correct_logical += 1
            except:
                pass  # 如果验证失败，认为不等价
        
        # 显示前几个详细结果
        if i < 10:
            print(f"\n样本 {i+1}:")
            print(f"  输入: {sample['input_text']}")
            print(f"  目标: {target_text}")
            print(f"  预测: {predicted_text}")
            print(f"  精确匹配: {'✓' if predicted_text == target_text else '✗'}")
    
    exact_accuracy = correct_exact / total
    logical_accuracy = correct_logical / total
    
    print(f"\n=== 详细评估结果 ===")
    print(f"精确匹配准确率: {exact_accuracy:.2%} ({correct_exact}/{total})")
    print(f"逻辑等价准确率: {logical_accuracy:.2%} ({correct_logical}/{total})")
    
    return exact_accuracy, logical_accuracy


def main():
    """主训练函数"""
    print("=== 自偏移推理训练 - 完整训练流程 ===\n")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 初始化
    tokenizer = Tokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 加载数据
    print("\n加载数据...")
    train_data = load_data('data/train.json', tokenizer, max_samples=2000)
    val_data = load_data('data/val.json', tokenizer, max_samples=500)
    
    print(f"训练数据: {len(train_data)} 样本")
    print(f"验证数据: {len(val_data)} 样本")
    
    # 创建改进的模型
    print("\n创建模型...")
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,  # 增大隐藏层
        max_length=50,
        learning_rate=0.005  # 较小的学习率
    )
    
    # 初始评估
    print("\n=== 训练前评估 ===")
    initial_exact, initial_logical = evaluate_model_detailed(model, val_data, tokenizer, max_samples=50)
    
    # 训练模型
    print("\n=== 开始训练 ===")
    history, best_accuracy = train_model(
        model, train_data, val_data, tokenizer,
        epochs=20,
        batch_size=16
    )
    
    # 最终评估
    print("\n=== 训练后评估 ===")
    final_exact, final_logical = evaluate_model_detailed(model, val_data, tokenizer, max_samples=100)
    
    # 训练总结
    print(f"\n=== 训练总结 ===")
    print(f"训练前精确准确率: {initial_exact:.2%}")
    print(f"训练后精确准确率: {final_exact:.2%}")
    print(f"精确准确率改进: {final_exact - initial_exact:.2%}")
    print(f"")
    print(f"训练前逻辑准确率: {initial_logical:.2%}")
    print(f"训练后逻辑准确率: {final_logical:.2%}")
    print(f"逻辑准确率改进: {final_logical - initial_logical:.2%}")
    print(f"")
    print(f"最佳验证准确率: {best_accuracy:.2%}")
    
    # 保存训练历史
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存到 training_history.json")

    # 保存训练好的模型
    model.save_model('trained_model.npz')
    print(f"训练好的模型已保存到 trained_model.npz")


if __name__ == "__main__":
    main()
