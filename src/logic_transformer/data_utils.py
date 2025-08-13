"""
文件名: src/logic_transformer/data_utils.py
数据工具模块
包含Tokenizer和数据加载功能
"""

import json
from typing import List, Dict, Tuple


class Tokenizer:
    """简单的字符级tokenizer，用于命题逻辑符号"""
    
    def __init__(self):
        # 定义所有可能的符号
        self.symbols = ['p', 'q', 'r', 's', 't', '~', '&', '|', '-', '>', '(', ')', ' ']
        self.char_to_int = {char: i for i, char in enumerate(self.symbols)}
        self.int_to_char = {i: char for i, char in enumerate(self.symbols)}
        self.vocab_size = len(self.symbols)
        
        # 特殊token
        self.PAD_TOKEN = len(self.symbols)
        self.START_TOKEN = len(self.symbols) + 1
        self.END_TOKEN = len(self.symbols) + 2
        
        self.char_to_int['<PAD>'] = self.PAD_TOKEN
        self.char_to_int['<START>'] = self.START_TOKEN
        self.char_to_int['<END>'] = self.END_TOKEN
        
        self.int_to_char[self.PAD_TOKEN] = '<PAD>'
        self.int_to_char[self.START_TOKEN] = '<START>'
        self.int_to_char[self.END_TOKEN] = '<END>'
        
        self.vocab_size += 3
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为整数序列"""
        return [self.char_to_int.get(char, self.PAD_TOKEN) for char in text]
    
    def decode(self, tokens: List[int]) -> str:
        """将整数序列解码为文本"""
        return ''.join([self.int_to_char.get(token, '') for token in tokens])


def load_dataset(filename: str, tokenizer: Tokenizer, max_samples: int = None) -> List[Dict]:
    """
    加载并预处理数据集
    
    Args:
        filename: 数据文件路径
        tokenizer: tokenizer实例
        max_samples: 最大样本数量
    
    Returns:
        处理后的数据列表
    """
    data = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                if line.strip():
                    sample = json.loads(line)
                    
                    # 编码输入和目标序列
                    input_tokens = tokenizer.encode(sample['noisy_prop'])
                    target_tokens = tokenizer.encode(sample['target_contrapositive'])
                    
                    data.append({
                        'input': input_tokens,
                        'target': target_tokens,
                        'input_text': sample['noisy_prop'],
                        'target_text': sample['target_contrapositive'],
                        'original_prop': sample.get('original_prop', ''),
                        'complexity': sample.get('complexity', 'simple')
                    })
    except FileNotFoundError:
        print(f"警告: 数据文件 {filename} 不存在")
    except Exception as e:
        print(f"加载数据时出错: {e}")
    
    return data


def create_data_splits(data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    将数据分割为训练集和验证集
    
    Args:
        data: 原始数据
        train_ratio: 训练集比例
    
    Returns:
        (训练集, 验证集)
    """
    import random
    
    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算分割点
    split_point = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_point]
    val_data = shuffled_data[split_point:]
    
    return train_data, val_data


def analyze_dataset(data: List[Dict], name: str = "数据集") -> Dict:
    """
    分析数据集的基本统计信息
    
    Args:
        data: 数据集
        name: 数据集名称
    
    Returns:
        统计信息字典
    """
    if not data:
        return {"error": "数据集为空"}
    
    # 基本统计
    total_samples = len(data)
    
    # 复杂度分布
    complexity_count = {}
    for sample in data:
        complexity = sample.get('complexity', 'unknown')
        complexity_count[complexity] = complexity_count.get(complexity, 0) + 1
    
    # 长度分布
    input_lengths = [len(sample['input_text']) for sample in data]
    target_lengths = [len(sample['target_text']) for sample in data]
    
    avg_input_length = sum(input_lengths) / len(input_lengths)
    avg_target_length = sum(target_lengths) / len(target_lengths)
    
    stats = {
        "name": name,
        "total_samples": total_samples,
        "complexity_distribution": complexity_count,
        "avg_input_length": round(avg_input_length, 2),
        "avg_target_length": round(avg_target_length, 2),
        "min_input_length": min(input_lengths),
        "max_input_length": max(input_lengths),
        "min_target_length": min(target_lengths),
        "max_target_length": max(target_lengths)
    }
    
    print(f"\n{name}分析:")
    print(f"总样本数: {total_samples}")
    print(f"复杂度分布: {complexity_count}")
    print(f"平均输入长度: {avg_input_length:.2f}")
    print(f"平均目标长度: {avg_target_length:.2f}")
    
    return stats


if __name__ == "__main__":
    # 测试代码
    tokenizer = Tokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 测试编码解码
    test_text = "p -> q"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"测试: '{test_text}' -> {encoded} -> '{decoded}'")
