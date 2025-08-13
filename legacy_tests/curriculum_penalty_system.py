"""
课程学习的惩罚机制
实现渐进式约束强度，平衡逻辑学习和语法规范
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


class CurriculumPenaltyGenerator:
    """课程学习的序列生成器，实现渐进式惩罚强度"""
    
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # 课程学习参数
        self.current_epoch = 0
        self.total_epochs = 60  # 总训练轮次
        
        # 三阶段课程设计
        self.stage1_epochs = 20  # 自由探索阶段
        self.stage2_epochs = 25  # 渐进约束阶段  
        self.stage3_epochs = 15  # 精细调优阶段
        
        # 基础惩罚参数（会根据阶段调整）
        self.base_repetition_penalty = 1.2
        self.base_cycle_penalty = 0.1
        self.base_end_token_boost = 2.0
        self.base_structural_guidance = 2.0
        
        # 当前阶段的实际惩罚强度
        self.current_penalties = self.calculate_current_penalties()
        
    def calculate_current_penalties(self) -> Dict[str, float]:
        """根据当前训练阶段计算惩罚强度"""
        
        if self.current_epoch <= self.stage1_epochs:
            # 阶段1：自由探索 - 最小约束
            stage = "free_exploration"
            progress = self.current_epoch / self.stage1_epochs
            
            penalties = {
                'repetition_penalty': 1.05,  # 极弱的重复惩罚
                'cycle_penalty': 0.8,        # 允许一些循环
                'end_token_boost': 1.2,      # 轻微的结束提升
                'structural_guidance': 1.1,  # 最小的结构指导
                'stage': stage,
                'progress': progress
            }
            
        elif self.current_epoch <= self.stage1_epochs + self.stage2_epochs:
            # 阶段2：渐进约束 - 逐步增强
            stage = "progressive_constraint"
            stage_epoch = self.current_epoch - self.stage1_epochs
            progress = stage_epoch / self.stage2_epochs
            
            # 线性插值增强惩罚强度
            penalties = {
                'repetition_penalty': 1.05 + (self.base_repetition_penalty - 1.05) * progress,
                'cycle_penalty': 0.8 - (0.8 - self.base_cycle_penalty) * progress,
                'end_token_boost': 1.2 + (self.base_end_token_boost - 1.2) * progress,
                'structural_guidance': 1.1 + (self.base_structural_guidance - 1.1) * progress,
                'stage': stage,
                'progress': progress
            }
            
        else:
            # 阶段3：精细调优 - 最强约束
            stage = "fine_tuning"
            stage_epoch = self.current_epoch - self.stage1_epochs - self.stage2_epochs
            progress = stage_epoch / self.stage3_epochs
            
            penalties = {
                'repetition_penalty': self.base_repetition_penalty,
                'cycle_penalty': self.base_cycle_penalty,
                'end_token_boost': self.base_end_token_boost,
                'structural_guidance': self.base_structural_guidance,
                'stage': stage,
                'progress': progress
            }
        
        return penalties
    
    def update_epoch(self, epoch: int):
        """更新当前训练轮次并重新计算惩罚强度"""
        self.current_epoch = epoch
        self.current_penalties = self.calculate_current_penalties()
        
        print(f"\n📚 课程学习状态更新:")
        print(f"  当前轮次: {epoch}/{self.total_epochs}")
        print(f"  当前阶段: {self.current_penalties['stage']}")
        print(f"  阶段进度: {self.current_penalties['progress']:.2%}")
        print(f"  惩罚强度: 重复={self.current_penalties['repetition_penalty']:.2f}, "
              f"循环={self.current_penalties['cycle_penalty']:.2f}")
    
    def apply_adaptive_repetition_penalty(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """自适应重复惩罚"""
        if not generated_tokens:
            return logits
        
        penalty_strength = self.current_penalties['repetition_penalty']
        
        # 统计token出现次数
        token_counts = {}
        for token in generated_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # 应用惩罚
        for token, count in token_counts.items():
            if token < len(logits):
                penalty = penalty_strength ** count
                logits[token] /= penalty
        
        return logits
    
    def apply_adaptive_cycle_penalty(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """自适应循环惩罚"""
        if len(generated_tokens) < 6:
            return logits
        
        penalty_strength = self.current_penalties['cycle_penalty']
        
        # 检测循环模式
        window_size = 3
        recent_tokens = generated_tokens[-window_size:]
        
        for i in range(len(generated_tokens) - window_size * 2, -1, -1):
            if i < 0:
                break
            
            prev_window = generated_tokens[i:i + window_size]
            if prev_window == recent_tokens:
                # 发现循环，应用惩罚
                for token in set(recent_tokens):
                    if token < len(logits):
                        logits[token] *= penalty_strength
                break
        
        return logits
    
    def apply_adaptive_structural_guidance(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """自适应结构指导"""
        if not generated_tokens:
            return logits
        
        guidance_strength = self.current_penalties['structural_guidance']
        last_token = generated_tokens[-1]
        
        # 只在阶段2和3应用强结构指导
        if self.current_penalties['stage'] == 'free_exploration':
            return logits
        
        # 结构化规则（强度可调）
        if last_token == self.tokenizer.char_to_int.get('-', -1):
            # 破折号后应该跟大于号
            gt_token = self.tokenizer.char_to_int.get('>', -1)
            if gt_token >= 0 and gt_token < len(logits):
                logits[gt_token] *= guidance_strength
            
            # 抑制再次生成破折号（强度可调）
            dash_token = self.tokenizer.char_to_int.get('-', -1)
            if dash_token >= 0 and dash_token < len(logits):
                logits[dash_token] *= (1.0 / guidance_strength)
        
        elif last_token == self.tokenizer.char_to_int.get('>', -1):
            # 大于号后应该跟空格
            space_token = self.tokenizer.char_to_int.get(' ', -1)
            if space_token >= 0 and space_token < len(logits):
                logits[space_token] *= guidance_strength
        
        return logits
    
    def apply_adaptive_end_token_boost(self, logits: np.ndarray, generated_tokens: List[int]) -> np.ndarray:
        """自适应END_TOKEN提升"""
        if len(generated_tokens) < 5:
            return logits
        
        boost_strength = self.current_penalties['end_token_boost']
        
        # 检查是否有基本结构
        has_negation = self.tokenizer.char_to_int.get('~', -1) in generated_tokens
        has_arrow = (self.tokenizer.char_to_int.get('-', -1) in generated_tokens and 
                    self.tokenizer.char_to_int.get('>', -1) in generated_tokens)
        
        if has_negation and has_arrow:
            logits[self.tokenizer.END_TOKEN] *= boost_strength
        
        return logits
    
    def generate_with_curriculum(self, input_sequence: List[int], max_steps: int = 20) -> Tuple[List[int], str]:
        """使用课程学习的序列生成"""
        
        # 编码输入
        encoded = self.model.encode(input_sequence)
        
        # 初始化
        generated_tokens = []
        current_token = self.tokenizer.START_TOKEN
        
        for step in range(max_steps):
            # 解码步骤
            hidden_state, raw_logits = self.model.decode_step(encoded, current_token)
            
            # 转换为概率
            logits = raw_logits.copy()
            
            # 应用自适应惩罚机制
            logits = self.apply_adaptive_repetition_penalty(logits, generated_tokens)
            logits = self.apply_adaptive_cycle_penalty(logits, generated_tokens)
            logits = self.apply_adaptive_structural_guidance(logits, generated_tokens)
            logits = self.apply_adaptive_end_token_boost(logits, generated_tokens)
            
            # 重新计算概率
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
            
            # 基本循环检测（始终保留）
            if len(generated_tokens) >= 6:
                last_3 = generated_tokens[-3:]
                prev_3 = generated_tokens[-6:-3]
                if last_3 == prev_3:
                    break
        
        # 解码结果
        decoded_text = self.tokenizer.decode(generated_tokens)
        
        return generated_tokens, decoded_text


def test_curriculum_penalty_system():
    """测试课程学习惩罚系统"""
    print("🎓 测试课程学习惩罚系统")
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
    
    # 创建课程学习生成器
    curriculum_generator = CurriculumPenaltyGenerator(model, tokenizer)
    
    # 测试不同阶段的生成效果
    test_input = "p -> q"
    input_sequence = tokenizer.encode(test_input)
    
    print(f"测试输入: '{test_input}'")
    print("=" * 40)
    
    # 模拟不同训练阶段
    test_epochs = [5, 15, 30, 45, 55]
    
    for epoch in test_epochs:
        curriculum_generator.update_epoch(epoch)
        
        # 生成多个样本观察差异
        print(f"\n🧪 轮次 {epoch} 的生成结果:")
        for i in range(3):
            tokens, text = curriculum_generator.generate_with_curriculum(input_sequence)
            print(f"  样本 {i+1}: '{text}'")
    
    print(f"\n📊 课程学习效果分析:")
    print(f"  阶段1 (自由探索): 应该看到更多样化但可能不完整的输出")
    print(f"  阶段2 (渐进约束): 应该看到逐步改善的结构")
    print(f"  阶段3 (精细调优): 应该看到最规范的输出")


if __name__ == "__main__":
    test_curriculum_penalty_system()
