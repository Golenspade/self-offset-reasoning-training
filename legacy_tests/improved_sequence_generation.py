"""
改进的序列生成机制
实现多种惩罚策略来解决循环问题
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer


class ImprovedSequenceGenerator:
    """改进的序列生成器，包含多种惩罚机制"""
    
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # 惩罚参数
        self.repetition_penalty = 1.2  # 重复惩罚强度
        self.length_penalty = 0.1      # 长度惩罚强度
        self.end_token_boost = 2.0     # END_TOKEN提升倍数
        self.max_repeats = 2           # 最大重复次数
        self.max_length = 30           # 最大序列长度
        
        # 循环检测参数
        self.cycle_detection_window = 3  # 循环检测窗口
        self.cycle_penalty = 0.1         # 循环惩罚强度
        
    def apply_repetition_penalty(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """
        应用重复惩罚
        对已生成的token降低概率
        """
        if not generated_tokens:
            return logits
        
        # 统计token出现次数
        token_counts = {}
        for token in generated_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # 应用惩罚
        for token, count in token_counts.items():
            if token < len(logits):
                # 重复次数越多，惩罚越重
                penalty = self.repetition_penalty ** count
                logits[token] /= penalty
        
        return logits
    
    def apply_cycle_penalty(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """
        应用循环检测惩罚
        检测并惩罚循环模式
        """
        if len(generated_tokens) < self.cycle_detection_window * 2:
            return logits
        
        # 检测最近的循环模式
        window_size = self.cycle_detection_window
        recent_tokens = generated_tokens[-window_size:]
        
        # 检查是否与之前的窗口重复
        for i in range(len(generated_tokens) - window_size * 2, -1, -1):
            if i < 0:
                break
            
            prev_window = generated_tokens[i:i + window_size]
            if prev_window == recent_tokens:
                # 发现循环，惩罚循环中的所有token
                for token in set(recent_tokens):
                    if token < len(logits):
                        logits[token] *= self.cycle_penalty
                break
        
        return logits
    
    def apply_length_penalty(self, logits: np.ndarray, current_length: int) -> np.ndarray:
        """
        应用长度惩罚
        随着序列变长，增加END_TOKEN的概率
        """
        if current_length > 5:  # 最小长度阈值
            # 长度惩罚：序列越长，END_TOKEN概率越高
            length_factor = 1 + (current_length - 5) * self.length_penalty
            logits[self.tokenizer.END_TOKEN] *= length_factor
        
        return logits
    
    def apply_end_token_boost(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """
        应用END_TOKEN提升
        在合适的时机提升END_TOKEN概率
        """
        # 如果已经生成了基本的逆否命题结构，提升END_TOKEN
        if len(generated_tokens) >= 5:
            # 检查是否包含基本结构：~ 变量 -> 
            has_negation = self.tokenizer.char_to_int['~'] in generated_tokens
            has_arrow = (self.tokenizer.char_to_int['-'] in generated_tokens and 
                        self.tokenizer.char_to_int['>'] in generated_tokens)
            has_variable = any(token in [0, 1, 2, 3, 4] for token in generated_tokens)  # p,q,r,s,t
            
            if has_negation and has_arrow and has_variable:
                logits[self.tokenizer.END_TOKEN] *= self.end_token_boost
        
        return logits
    
    def apply_structural_guidance(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """
        应用结构化指导
        根据逻辑命题的结构规律调整概率
        """
        if not generated_tokens:
            return logits
        
        last_token = generated_tokens[-1]
        
        # 结构化规则
        if last_token == self.tokenizer.char_to_int['~']:
            # 否定符后应该跟变量或括号
            for token in [0, 1, 2, 3, 4, 10]:  # p,q,r,s,t,(
                if token < len(logits):
                    logits[token] *= 2.0
        
        elif last_token == self.tokenizer.char_to_int['-']:
            # 破折号后应该跟大于号
            logits[self.tokenizer.char_to_int['>']] *= 3.0
            # 强烈抑制再次生成破折号
            logits[self.tokenizer.char_to_int['-']] *= 0.1
        
        elif last_token == self.tokenizer.char_to_int['>']:
            # 大于号后应该跟空格或变量
            logits[self.tokenizer.char_to_int[' ']] *= 2.0
            for token in [0, 1, 2, 3, 4]:  # p,q,r,s,t
                if token < len(logits):
                    logits[token] *= 1.5
            # 抑制立即生成另一个箭头
            logits[self.tokenizer.char_to_int['-']] *= 0.3
        
        elif last_token == self.tokenizer.char_to_int[' ']:
            # 空格后的规则
            if len(generated_tokens) >= 2:
                prev_token = generated_tokens[-2]
                if prev_token == self.tokenizer.char_to_int['>']:
                    # -> 后的空格，应该跟变量或否定
                    for token in [0, 1, 2, 3, 4, 5]:  # p,q,r,s,t,~
                        if token < len(logits):
                            logits[token] *= 2.0
                    # 强烈抑制再次生成箭头
                    logits[self.tokenizer.char_to_int['-']] *= 0.1
        
        return logits
    
    def detect_completion(self, generated_tokens: List[int]) -> bool:
        """
        检测序列是否应该完成
        """
        if len(generated_tokens) < 3:
            return False
        
        # 检查是否有完整的逆否命题结构
        text = self.tokenizer.decode(generated_tokens)
        
        # 基本完整性检查
        has_negation = '~' in text
        has_arrow = '->' in text
        has_variable = any(var in text for var in ['p', 'q', 'r', 's', 't'])
        
        # 如果有基本结构且长度合理，可以结束
        if has_negation and has_arrow and has_variable and len(generated_tokens) >= 5:
            return True
        
        # 如果序列过长，强制结束
        if len(generated_tokens) >= self.max_length:
            return True
        
        return False
    
    def generate_sequence(self, input_sequence: List[int], max_steps: int = 20) -> Tuple[List[int], str]:
        """
        生成改进的序列
        """
        # 编码输入
        encoded = self.model.encode(input_sequence)
        
        # 初始化
        generated_tokens = []
        current_token = self.tokenizer.START_TOKEN
        
        for step in range(max_steps):
            # 解码步骤
            hidden_state, raw_logits = self.model.decode_step(encoded, current_token)
            
            # 转换为概率（softmax）
            logits = raw_logits.copy()
            
            # 应用各种惩罚机制
            logits = self.apply_repetition_penalty(logits, generated_tokens)
            logits = self.apply_cycle_penalty(logits, generated_tokens)
            logits = self.apply_length_penalty(logits, len(generated_tokens))
            logits = self.apply_end_token_boost(logits, generated_tokens)
            logits = self.apply_structural_guidance(logits, generated_tokens)
            
            # 重新计算概率
            exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
            probabilities = exp_logits / np.sum(exp_logits)
            
            # 选择下一个token
            next_token = int(np.argmax(probabilities))
            
            # 检查终止条件
            if next_token == self.tokenizer.END_TOKEN:
                break
            
            # 检查是否应该强制完成
            if self.detect_completion(generated_tokens):
                break
            
            # 检查token有效性
            if next_token >= self.tokenizer.vocab_size or next_token < 0:
                next_token = self.tokenizer.END_TOKEN
                break
            
            # 添加到序列
            generated_tokens.append(next_token)
            current_token = next_token
            
            # 循环检测：如果检测到严重循环，强制结束
            if len(generated_tokens) >= 6:
                last_3 = generated_tokens[-3:]
                prev_3 = generated_tokens[-6:-3]
                if last_3 == prev_3:
                    print(f"检测到循环，强制结束: {last_3}")
                    break
        
        # 解码结果
        decoded_text = self.tokenizer.decode(generated_tokens)
        
        return generated_tokens, decoded_text


def test_improved_generation():
    """测试改进的生成机制"""
    print("🚀 测试改进的序列生成机制")
    print("=" * 60)
    
    # 加载模型和tokenizer
    tokenizer = Tokenizer()
    
    # 这里需要导入实际的模型类
    from logic_transformer.models.base_model import ImprovedSimpleModel
    
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        max_length=50,
        learning_rate=0.003
    )
    
    model_path = 'outputs/trained_models/robust_model_Level_1_鲁棒版.npz'
    if not model.load_model(model_path):
        print(f"❌ 无法加载模型: {model_path}")
        return
    
    # 创建改进的生成器
    generator = ImprovedSequenceGenerator(model, tokenizer)
    
    # 测试用例
    test_cases = [
        "p -> q",
        "~p -> r",
        "(p & q) -> s"
    ]
    
    for test_input in test_cases:
        print(f"\n🧪 测试输入: '{test_input}'")
        print("-" * 40)
        
        # 编码输入
        input_sequence = tokenizer.encode(test_input)
        
        # 生成序列
        generated_tokens, decoded_text = generator.generate_sequence(input_sequence)
        
        print(f"生成的tokens: {generated_tokens}")
        print(f"解码结果: '{decoded_text}'")
        
        # 分析结果
        if '-> -> ->' in decoded_text:
            print("❌ 仍然存在循环问题")
        elif '->' in decoded_text and len(decoded_text.strip()) > 3:
            print("✅ 生成了合理的逻辑表达式")
        else:
            print("🔄 生成结果需要进一步改进")


if __name__ == "__main__":
    test_improved_generation()
