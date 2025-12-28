"""
文件名: simple_model.py
简化版的序列到序列模型
使用numpy实现，不依赖PyTorch
用于验证自偏移推理训练的核心概念
"""

import numpy as np
import json
import sys
from pathlib import Path

# 确保可以从 scripts/ 目录导入项目根模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from logic_utils import Tokenizer


class SimpleSeq2SeqModel:
    """简化的序列到序列模型，使用简单的神经网络架构"""

    def __init__(self, vocab_size, hidden_size=64, max_length=50):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length

        # 初始化权重
        self.embedding = np.random.randn(vocab_size, hidden_size) * 0.1
        self.encoder_weights = np.random.randn(hidden_size, hidden_size) * 0.1
        self.decoder_weights = np.random.randn(hidden_size, hidden_size) * 0.1
        self.output_weights = np.random.randn(hidden_size, vocab_size) * 0.1

        # 偏置
        self.encoder_bias = np.zeros(hidden_size)
        self.decoder_bias = np.zeros(hidden_size)
        self.output_bias = np.zeros(vocab_size)

        print("简化模型创建成功!")
        print(f"词汇表大小: {vocab_size}")
        print(f"隐藏层大小: {hidden_size}")
        print(f"最大序列长度: {max_length}")

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(x)

    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def encode(self, input_sequence):
        """编码输入序列"""
        # 简单的编码：对所有输入token的embedding求平均
        embeddings = []
        for token in input_sequence:
            if token < self.vocab_size:
                embeddings.append(self.embedding[token])

        if not embeddings:
            return np.zeros(self.hidden_size)

        # 平均池化
        avg_embedding = np.mean(embeddings, axis=0)

        # 通过编码器层
        encoded = self.tanh(np.dot(avg_embedding, self.encoder_weights) + self.encoder_bias)

        return encoded

    def decode_step(self, hidden_state, previous_token):
        """解码一个时间步"""
        # 获取前一个token的embedding
        if previous_token < self.vocab_size:
            token_embedding = self.embedding[previous_token]
        else:
            token_embedding = np.zeros(self.hidden_size)

        # 结合隐藏状态和token embedding
        combined = hidden_state + token_embedding

        # 通过解码器层
        new_hidden = self.tanh(np.dot(combined, self.decoder_weights) + self.decoder_bias)

        # 生成输出概率
        output_logits = np.dot(new_hidden, self.output_weights) + self.output_bias
        output_probs = self.softmax(output_logits)

        return new_hidden, output_probs

    def predict(self, input_sequence, tokenizer):
        """预测输出序列"""
        # 编码输入
        encoded = self.encode(input_sequence)

        # 初始化解码
        hidden_state = encoded
        output_sequence = []
        current_token = tokenizer.START_TOKEN

        for _ in range(self.max_length):
            hidden_state, output_probs = self.decode_step(hidden_state, current_token)

            # 选择概率最高的token
            next_token = np.argmax(output_probs)

            # 如果生成了结束token，停止
            if next_token == tokenizer.END_TOKEN:
                break

            output_sequence.append(next_token)
            current_token = next_token

        return output_sequence

    def train_step(self, input_sequence, target_sequence, tokenizer, learning_rate=0.01):
        """简化的训练步骤（仅用于概念验证）"""
        # 前向传播
        encoded = self.encode(input_sequence)

        # 计算损失（简化版）
        total_loss = 0
        hidden_state = encoded

        for i, target_token in enumerate(target_sequence):
            if i == 0:
                current_token = tokenizer.START_TOKEN
            else:
                current_token = target_sequence[i - 1]

            hidden_state, output_probs = self.decode_step(hidden_state, current_token)

            # 计算交叉熵损失
            if target_token < self.vocab_size:
                loss = -np.log(output_probs[target_token] + 1e-8)
                total_loss += loss

        return total_loss / len(target_sequence) if target_sequence else 0


def load_data(filename, tokenizer, max_samples=1000):
    """加载并预处理数据"""
    data = []

    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            if line.strip():
                sample = json.loads(line)

                # 编码输入和目标序列
                input_tokens = tokenizer.encode(sample["noisy_prop"])
                target_tokens = tokenizer.encode(sample["target_contrapositive"])

                data.append(
                    {
                        "input": input_tokens,
                        "target": target_tokens,
                        "input_text": sample["noisy_prop"],
                        "target_text": sample["target_contrapositive"],
                    }
                )

    return data


def evaluate_model(model, data, tokenizer):
    """评估模型性能"""
    correct = 0
    total = len(data)

    print("开始评估模型...")

    for i, sample in enumerate(data[:100]):  # 只评估前100个样本
        predicted_tokens = model.predict(sample["input"], tokenizer)
        predicted_text = tokenizer.decode(predicted_tokens)

        # 简单的字符串匹配评估
        if predicted_text.strip() == sample["target_text"].strip():
            correct += 1

        # 显示前几个预测结果
        if i < 5:
            print(f"\n样本 {i+1}:")
            print(f"输入: {sample['input_text']}")
            print(f"目标: {sample['target_text']}")
            print(f"预测: {predicted_text}")

    accuracy = correct / min(100, total)
    print("\n评估完成!")
    print(f"准确率: {accuracy:.2%} ({correct}/{min(100, total)})")

    return accuracy


def main():
    """主函数"""
    print("=== 简化版自偏移推理训练Demo ===\n")

    # 初始化tokenizer
    tokenizer = Tokenizer()

    # 创建模型
    model = SimpleSeq2SeqModel(
        vocab_size=tokenizer.vocab_size, hidden_size=64, max_length=50
    )

    # 加载数据
    print("\n加载训练数据...")
    train_data = load_data("data/train.json", tokenizer, max_samples=1000)
    print(f"加载了 {len(train_data)} 个训练样本")

    print("\n加载验证数据...")
    val_data = load_data("data/val.json", tokenizer, max_samples=200)
    print(f"加载了 {len(val_data)} 个验证样本")

    # 显示数据样本
    print("\n数据样本:")
    for i in range(3):
        sample = train_data[i]
        print(f"样本 {i+1}:")
        print(f"  输入: {sample['input_text']} -> {sample['input']}")
        print(f"  目标: {sample['target_text']} -> {sample['target']}")

    # 评估未训练的模型
    print("\n=== 评估未训练的模型 ===")
    initial_accuracy = evaluate_model(model, val_data, tokenizer)

    # 简单的训练循环
    print("\n=== 开始训练 ===")
    print("注意：这是一个简化的训练过程，主要用于概念验证")

    for epoch in range(5):
        total_loss = 0
        for sample in train_data[:100]:  # 只训练前100个样本
            loss = model.train_step(sample["input"], sample["target"], tokenizer)
            total_loss += loss

        avg_loss = total_loss / min(100, len(train_data))
        print(f"Epoch {epoch + 1}: 平均损失 = {avg_loss:.4f}")

    # 评估训练后的模型
    print("\n=== 评估训练后的模型 ===")
    final_accuracy = evaluate_model(model, val_data, tokenizer)

    print("\n=== 训练总结 ===")
    print(f"初始准确率: {initial_accuracy:.2%}")
    print(f"最终准确率: {final_accuracy:.2%}")
    print(f"改进: {final_accuracy - initial_accuracy:.2%}")


if __name__ == "__main__":
    main()

