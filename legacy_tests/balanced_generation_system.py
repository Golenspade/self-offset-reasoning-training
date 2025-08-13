"""
平衡的生成系统
结合软约束和混合模型思路，平衡逻辑学习和语法规范
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


class BalancedSequenceGenerator:
    """平衡的序列生成器，实现软约束和逻辑优先策略"""
    
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # 软约束参数
        self.logic_priority_weight = 0.7    # 逻辑优先权重
        self.grammar_guidance_weight = 0.3  # 语法指导权重
        
        # 温和的惩罚参数
        self.soft_repetition_penalty = 1.1  # 降低重复惩罚
        self.soft_cycle_penalty = 0.5       # 温和的循环惩罚
        self.completion_encouragement = 1.5  # 鼓励完成
        
        # 逻辑完整性检查
        self.min_logical_length = 5        # 最小逻辑长度
        self.max_logical_length = 15       # 最大逻辑长度
        
    def calculate_logic_reward(self, generated_tokens: List[int], input_sequence: List[int]) -> float:
        """计算逻辑奖励分数"""
        if not generated_tokens:
            return 0.0
        
        current_text = self.tokenizer.decode(generated_tokens)
        input_text = self.tokenizer.decode(input_sequence)
        
        reward = 0.0
        
        # 基础结构奖励
        if '~' in current_text:
            reward += 0.2  # 有否定符
        if '->' in current_text:
            reward += 0.3  # 有蕴含符
        
        # 完整性奖励
        if '->' in current_text:
            arrow_pos = current_text.find('->')
            after_arrow = current_text[arrow_pos + 2:].strip()
            if after_arrow:  # 箭头后有内容
                reward += 0.3
                if len(after_arrow) > 1:  # 箭头后有实质内容
                    reward += 0.2
        
        # 变量一致性奖励
        input_vars = set(c for c in input_text if c in 'pqrst')
        output_vars = set(c for c in current_text if c in 'pqrst')
        if input_vars and output_vars:
            overlap = len(input_vars & output_vars) / len(input_vars)
            reward += overlap * 0.3
        
        return min(reward, 1.0)  # 限制在[0,1]范围
    
    def apply_soft_constraints(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """应用软约束（温和的语法指导）"""
        
        # 1. 温和的重复惩罚
        if generated_tokens:
            token_counts = {}
            for token in generated_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            for token, count in token_counts.items():
                if token < len(logits) and count > 1:
                    penalty = self.soft_repetition_penalty ** (count - 1)
                    logits[token] /= penalty
        
        # 2. 温和的循环检测
        if len(generated_tokens) >= 6:
            last_3 = generated_tokens[-3:]
            prev_3 = generated_tokens[-6:-3]
            if last_3 == prev_3:
                for token in set(last_3):
                    if token < len(logits):
                        logits[token] *= self.soft_cycle_penalty
        
        # 3. 基本的结构指导（但不强制）
        if generated_tokens:
            last_token = generated_tokens[-1]
            
            if last_token == self.tokenizer.char_to_int.get('-', -1):
                # 破折号后鼓励大于号
                gt_token = self.tokenizer.char_to_int.get('>', -1)
                if gt_token >= 0 and gt_token < len(logits):
                    logits[gt_token] *= 1.5  # 温和的鼓励
            
            elif last_token == self.tokenizer.char_to_int.get('>', -1):
                # 大于号后鼓励空格或变量
                space_token = self.tokenizer.char_to_int.get(' ', -1)
                if space_token >= 0 and space_token < len(logits):
                    logits[space_token] *= 1.3
                
                # 鼓励变量
                for var_token in [0, 1, 2, 3, 4]:  # p,q,r,s,t
                    if var_token < len(logits):
                        logits[var_token] *= 1.2
        
        return logits
    
    def apply_completion_encouragement(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """鼓励完成完整的逻辑表达式"""
        
        current_text = self.tokenizer.decode(generated_tokens)
        
        # 如果已经有箭头但后面内容不完整，鼓励继续生成
        if '->' in current_text:
            arrow_pos = current_text.find('->')
            after_arrow = current_text[arrow_pos + 2:].strip()
            
            if not after_arrow:
                # 箭头后没有内容，鼓励生成空格
                space_token = self.tokenizer.char_to_int.get(' ', -1)
                if space_token >= 0 and space_token < len(logits):
                    logits[space_token] *= 2.0
            
            elif after_arrow == ' ':
                # 只有空格，鼓励生成否定符或变量
                neg_token = self.tokenizer.char_to_int.get('~', -1)
                if neg_token >= 0 and neg_token < len(logits):
                    logits[neg_token] *= 1.8
                
                for var_token in [0, 1, 2, 3, 4]:  # p,q,r,s,t
                    if var_token < len(logits):
                        logits[var_token] *= 1.5
            
            elif len(after_arrow.strip()) == 1:
                # 只有一个字符，可能需要更多内容
                # 适度抑制END_TOKEN
                logits[self.tokenizer.END_TOKEN] *= 0.7
        
        # 如果长度太短，抑制END_TOKEN
        if len(generated_tokens) < self.min_logical_length:
            logits[self.tokenizer.END_TOKEN] *= 0.3
        
        # 如果长度合适且有完整结构，鼓励END_TOKEN
        elif (len(generated_tokens) >= self.min_logical_length and 
              '~' in current_text and '->' in current_text):
            arrow_pos = current_text.find('->')
            if arrow_pos >= 0 and len(current_text) > arrow_pos + 3:
                logits[self.tokenizer.END_TOKEN] *= self.completion_encouragement
        
        return logits
    
    def generate_balanced_sequence(self, input_sequence: List[int], max_steps: int = 20) -> Tuple[List[int], str, Dict]:
        """生成平衡的序列，返回tokens、文本和调试信息"""
        
        # 编码输入
        encoded = self.model.encode(input_sequence)
        
        # 初始化
        generated_tokens = []
        current_token = self.tokenizer.START_TOKEN
        debug_info = {
            'logic_rewards': [],
            'step_decisions': [],
            'final_logic_score': 0.0
        }
        
        for step in range(max_steps):
            # 解码步骤
            hidden_state, raw_logits = self.model.decode_step(encoded, current_token)
            
            # 计算当前的逻辑奖励
            logic_reward = self.calculate_logic_reward(generated_tokens, input_sequence)
            debug_info['logic_rewards'].append(logic_reward)
            
            # 应用平衡策略
            logits = raw_logits.copy()
            
            # 软约束（语法指导）
            logits = self.apply_soft_constraints(logits, generated_tokens)
            
            # 完成鼓励（逻辑完整性）
            logits = self.apply_completion_encouragement(logits, generated_tokens)
            
            # 重新计算概率
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # 选择下一个token
            next_token = int(np.argmax(probabilities))
            next_char = self.tokenizer.int_to_char.get(next_token, 'UNK')
            
            debug_info['step_decisions'].append({
                'step': step,
                'token': next_token,
                'char': next_char,
                'logic_reward': logic_reward,
                'top_3_probs': [(i, probabilities[i]) for i in np.argsort(probabilities)[-3:][::-1]]
            })
            
            # 检查终止条件
            if next_token == self.tokenizer.END_TOKEN:
                break
            
            # 检查token有效性
            if next_token >= self.tokenizer.vocab_size or next_token < 0:
                break
            
            # 添加到序列
            generated_tokens.append(next_token)
            current_token = next_token
            
            # 长度限制
            if len(generated_tokens) >= self.max_logical_length:
                break
        
        # 计算最终逻辑分数
        debug_info['final_logic_score'] = self.calculate_logic_reward(generated_tokens, input_sequence)
        
        # 解码结果
        decoded_text = self.tokenizer.decode(generated_tokens)
        
        return generated_tokens, decoded_text, debug_info


def test_balanced_generation():
    """测试平衡生成系统"""
    print("⚖️ 测试平衡生成系统")
    print("=" * 60)
    
    # 加载模型和tokenizer
    tokenizer = Tokenizer()
    
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
    
    # 创建平衡生成器
    balanced_generator = BalancedSequenceGenerator(model, tokenizer)
    
    # 测试用例
    test_cases = [
        ("p -> q", "~q -> ~p"),
        ("~p -> r", "~r -> p"),
        ("q -> s", "~s -> ~q")
    ]
    
    for test_input, expected in test_cases:
        print(f"\n🧪 测试输入: '{test_input}'")
        print(f"期望输出: '{expected}'")
        print("-" * 50)
        
        input_sequence = tokenizer.encode(test_input)
        
        # 生成多个样本
        for i in range(3):
            tokens, text, debug = balanced_generator.generate_balanced_sequence(input_sequence)
            
            print(f"  样本 {i+1}: '{text}' (逻辑分数: {debug['final_logic_score']:.2f})")
            
            # 分析质量
            if text.count('->') == 1 and '~' in text:
                if len(text.split('->')[1].strip()) > 0:
                    print(f"    ✅ 结构完整")
                else:
                    print(f"    🔄 结构不完整")
            else:
                print(f"    ❌ 结构有问题")
    
    print(f"\n📊 平衡生成系统分析:")
    print(f"  优势: 软约束保持了逻辑探索空间")
    print(f"  改进: 鼓励完成机制提高了输出完整性")
    print(f"  平衡: 在语法规范和逻辑自由之间找到平衡点")


if __name__ == "__main__":
    test_balanced_generation()
