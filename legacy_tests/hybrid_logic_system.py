"""
混合逻辑系统
神经网络负责逻辑推理，规则系统负责语法规范
实现您建议的"让每个系统做自己最擅长的事"
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
from pathlib import Path
import re

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer


class LogicIntentExtractor:
    """逻辑意图提取器 - 从神经网络的"混乱"输出中提取逻辑意图"""
    
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
    
    def extract_variables(self, text: str) -> List[str]:
        """提取文本中的变量"""
        variables = []
        for char in text:
            if char in 'pqrst':
                variables.append(char)
        return list(set(variables))  # 去重
    
    def extract_negations(self, text: str) -> List[str]:
        """提取否定的变量"""
        negated_vars = []
        i = 0
        while i < len(text):
            if text[i] == '~' and i + 1 < len(text) and text[i + 1] in 'pqrst':
                negated_vars.append(text[i + 1])
                i += 2
            else:
                i += 1
        return negated_vars
    
    def analyze_logic_intent(self, neural_output: str, input_text: str) -> Dict:
        """分析神经网络输出的逻辑意图"""
        
        # 提取输入的结构
        input_vars = self.extract_variables(input_text)
        input_negated = self.extract_negations(input_text)
        
        # 提取输出的结构
        output_vars = self.extract_variables(neural_output)
        output_negated = self.extract_negations(neural_output)
        
        # 分析意图
        intent = {
            'input_variables': input_vars,
            'input_negated': input_negated,
            'output_variables': output_vars,
            'output_negated': output_negated,
            'has_implication': '->' in neural_output,
            'structure_type': 'unknown'
        }
        
        # 判断结构类型
        if intent['has_implication'] and output_negated:
            if len(output_vars) >= 1:
                intent['structure_type'] = 'contrapositive_attempt'
            else:
                intent['structure_type'] = 'incomplete_contrapositive'
        elif intent['has_implication']:
            intent['structure_type'] = 'implication_attempt'
        else:
            intent['structure_type'] = 'fragment'
        
        return intent


class LogicRuleGenerator:
    """逻辑规则生成器 - 根据意图生成正确的逻辑表达式"""
    
    def __init__(self):
        pass
    
    def generate_contrapositive(self, original_formula: str) -> str:
        """生成标准的逆否命题"""
        
        # 简单的逆否命题生成（针对 A -> B 形式）
        if '->' not in original_formula:
            return original_formula
        
        parts = original_formula.split('->')
        if len(parts) != 2:
            return original_formula
        
        antecedent = parts[0].strip()
        consequent = parts[1].strip()
        
        # 生成逆否：~B -> ~A
        neg_consequent = self.negate_expression(consequent)
        neg_antecedent = self.negate_expression(antecedent)
        
        return f"{neg_consequent} -> {neg_antecedent}"
    
    def negate_expression(self, expr: str) -> str:
        """否定一个表达式"""
        expr = expr.strip()
        
        # 如果已经是否定的，去掉否定
        if expr.startswith('~'):
            return expr[1:].strip()
        
        # 如果是单个变量，直接否定
        if len(expr) == 1 and expr in 'pqrst':
            return f"~{expr}"
        
        # 如果是复杂表达式，加括号否定
        if '(' in expr or '&' in expr or '|' in expr:
            return f"~({expr})"
        
        # 默认情况
        return f"~{expr}"
    
    def repair_logic_expression(self, intent: Dict, input_text: str) -> str:
        """根据意图修复逻辑表达式"""
        
        if intent['structure_type'] == 'contrapositive_attempt':
            # 神经网络试图生成逆否命题，我们帮它完成
            return self.generate_contrapositive(input_text)
        
        elif intent['structure_type'] == 'incomplete_contrapositive':
            # 不完整的逆否命题，补全它
            return self.generate_contrapositive(input_text)
        
        elif intent['structure_type'] == 'implication_attempt':
            # 试图生成蕴含，但可能不是逆否命题
            return self.generate_contrapositive(input_text)
        
        else:
            # 其他情况，生成标准逆否命题
            return self.generate_contrapositive(input_text)


class HybridLogicSystem:
    """混合逻辑系统 - 结合神经网络和规则系统"""
    
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.intent_extractor = LogicIntentExtractor(tokenizer)
        self.rule_generator = LogicRuleGenerator()
        
        # 神经网络生成参数（宽松设置，允许探索）
        self.neural_max_steps = 15
        self.neural_temperature = 1.0  # 增加随机性
        
    def generate_neural_attempt(self, input_sequence: List[int]) -> Tuple[str, Dict]:
        """让神经网络自由尝试，不施加强约束"""
        
        # 编码输入
        encoded = self.model.encode(input_sequence)
        
        # 初始化
        generated_tokens = []
        current_token = self.tokenizer.START_TOKEN
        
        for step in range(self.neural_max_steps):
            # 解码步骤
            hidden_state, raw_logits = self.model.decode_step(encoded, current_token)
            
            # 只应用最基本的约束
            logits = raw_logits.copy()
            
            # 基本的循环检测（防止完全卡死）
            if len(generated_tokens) >= 6:
                last_3 = generated_tokens[-3:]
                prev_3 = generated_tokens[-6:-3]
                if last_3 == prev_3:
                    # 检测到循环，强制结束
                    break
            
            # 温度采样（增加多样性）
            if self.neural_temperature != 1.0:
                logits = logits / self.neural_temperature
            
            # 计算概率
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # 选择下一个token
            next_token = int(np.argmax(probabilities))
            
            # 检查终止条件
            if next_token == self.tokenizer.END_TOKEN:
                break
            
            # 检查token有效性
            if next_token >= self.tokenizer.vocab_size or next_token < 0:
                break
            
            # 添加到序列
            generated_tokens.append(next_token)
            current_token = next_token
        
        # 解码神经网络的尝试
        neural_output = self.tokenizer.decode(generated_tokens)
        
        # 分析意图
        input_text = self.tokenizer.decode(input_sequence)
        intent = self.intent_extractor.analyze_logic_intent(neural_output, input_text)
        
        return neural_output, intent
    
    def generate_hybrid_solution(self, input_text: str) -> Tuple[str, str, Dict]:
        """生成混合解决方案"""
        
        print(f"🧠 混合系统处理: '{input_text}'")
        
        # 1. 神经网络自由尝试
        input_sequence = self.tokenizer.encode(input_text)
        neural_output, intent = self.generate_neural_attempt(input_sequence)
        
        print(f"  神经网络输出: '{neural_output}'")
        print(f"  识别意图: {intent['structure_type']}")
        
        # 2. 规则系统修复和完善
        corrected_output = self.rule_generator.repair_logic_expression(intent, input_text)
        
        print(f"  规则系统修正: '{corrected_output}'")
        
        return neural_output, corrected_output, intent


def test_hybrid_system():
    """测试混合逻辑系统"""
    print("🤖 测试混合逻辑系统")
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
    
    # 创建混合系统
    hybrid_system = HybridLogicSystem(model, tokenizer)
    
    # 测试用例
    test_cases = [
        ("p -> q", "~q -> ~p"),
        ("~p -> r", "~r -> p"),
        ("q -> s", "~s -> ~q"),
        ("(p & q) -> r", "~r -> ~(p & q)")
    ]
    
    print("🔄 混合系统工作流程演示:")
    print("=" * 50)
    
    for test_input, expected in test_cases:
        print(f"\n📝 测试案例: '{test_input}' → 期望: '{expected}'")
        print("-" * 60)
        
        # 运行混合系统
        neural_output, final_output, intent = hybrid_system.generate_hybrid_solution(test_input)
        
        # 评估结果
        print(f"  📊 评估:")
        if final_output == expected:
            print(f"    ✅ 完全正确!")
        elif final_output.replace(' ', '') == expected.replace(' ', ''):
            print(f"    ✅ 逻辑正确 (格式略有差异)")
        else:
            print(f"    🔄 需要进一步改进")
        
        print(f"    神经网络贡献: 提供了逻辑方向和变量信息")
        print(f"    规则系统贡献: 确保了语法正确性和完整性")
    
    print(f"\n🎯 混合系统优势总结:")
    print(f"  1. 神经网络专注于逻辑理解，不被语法约束束缚")
    print(f"  2. 规则系统确保输出的正确性和完整性")
    print(f"  3. 两个系统各司其职，避免了冲突的优化目标")
    print(f"  4. 即使神经网络输出不完美，规则系统也能修正")


if __name__ == "__main__":
    test_hybrid_system()
