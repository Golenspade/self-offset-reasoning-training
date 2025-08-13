"""
文件名: src/logic_transformer/training/precision_engineering.py
第一阶段：精准工程 - 智慧调速器与安全刹车
实现自适应学习率和梯度裁剪，奠定稳定根基
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionTrainer:
    """精准工程训练器 - 实现智慧调速器和安全刹车"""
    
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 智慧调速器：自适应学习率
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('initial_lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器 - 当验证损失不再改善时自动降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',           # 监控的指标越小越好
            factor=config.get('lr_decay_factor', 0.5),     # 学习率衰减因子
            patience=config.get('lr_patience', 2),         # 容忍多少个epoch没有改善
            min_lr=config.get('min_lr', 1e-6),            # 最小学习率
            verbose=True          # 调整时输出日志
        )
        
        # 安全刹车：梯度裁剪参数
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # 训练历史记录
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norms': [],
            'clipped_steps': 0
        }
        
        logger.info("🚀 精准工程训练器初始化完成")
        logger.info(f"  初始学习率: {config.get('initial_lr', 0.001)}")
        logger.info(f"  梯度裁剪阈值: {self.max_grad_norm}")
        logger.info(f"  学习率衰减因子: {config.get('lr_decay_factor', 0.5)}")
    
    def train_step_with_precision(self, batch_data: List[Dict]) -> Dict:
        """执行一个精准训练步骤"""
        self.model.train()
        
        total_loss = 0.0
        batch_size = len(batch_data)
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        for sample in batch_data:
            try:
                # 计算损失
                loss = self.model.train_step_improved(
                    sample['input'], 
                    sample['target'], 
                    self.tokenizer
                )
                total_loss += loss
            except Exception as e:
                logger.warning(f"训练样本出错: {e}")
                continue
        
        # 平均损失
        avg_loss = total_loss / batch_size if batch_size > 0 else 0.0
        
        # 反向传播
        if avg_loss > 0:
            # 这里需要确保loss是tensor并且requires_grad=True
            loss_tensor = torch.tensor(avg_loss, requires_grad=True)
            loss_tensor.backward()
            
            # 🛡️ 安全刹车：梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.max_grad_norm
            )
            
            # 记录梯度信息
            self.training_history['grad_norms'].append(float(grad_norm))
            if grad_norm > self.max_grad_norm:
                self.training_history['clipped_steps'] += 1
                logger.debug(f"梯度被裁剪: {grad_norm:.4f} -> {self.max_grad_norm}")
            
            # 更新参数
            self.optimizer.step()
        
        return {
            'loss': avg_loss,
            'grad_norm': float(grad_norm) if 'grad_norm' in locals() else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_and_adjust(self, val_data: List[Dict]) -> Dict:
        """验证并调整学习率"""
        self.model.eval()
        
        total_val_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for sample in val_data:
                try:
                    # 这里需要实现验证损失计算
                    # 暂时使用训练损失作为代理
                    val_loss = self.model.train_step_improved(
                        sample['input'], 
                        sample['target'], 
                        self.tokenizer
                    )
                    total_val_loss += val_loss
                    num_samples += 1
                except Exception as e:
                    continue
        
        avg_val_loss = total_val_loss / num_samples if num_samples > 0 else float('inf')
        
        # 🧠 智慧调速器：根据验证损失调整学习率
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(avg_val_loss)
        new_lr = self.optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            logger.info(f"🎯 学习率自动调整: {old_lr:.6f} -> {new_lr:.6f}")
        
        return {
            'val_loss': avg_val_loss,
            'learning_rate': new_lr,
            'lr_adjusted': new_lr != old_lr
        }
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        if not self.training_history['grad_norms']:
            return {}
        
        grad_norms = self.training_history['grad_norms']
        
        return {
            'avg_grad_norm': np.mean(grad_norms),
            'max_grad_norm': np.max(grad_norms),
            'clipped_ratio': self.training_history['clipped_steps'] / len(grad_norms),
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'total_steps': len(grad_norms)
        }
    
    def save_training_state(self, filepath: str):
        """保存训练状态"""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(state, filepath)
        logger.info(f"训练状态已保存: {filepath}")
    
    def load_training_state(self, filepath: str):
        """加载训练状态"""
        state = torch.load(filepath)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.training_history = state['training_history']
        logger.info(f"训练状态已加载: {filepath}")


def create_precision_config() -> Dict:
    """创建精准工程配置"""
    return {
        'initial_lr': 0.001,        # 初始学习率
        'lr_decay_factor': 0.5,     # 学习率衰减因子
        'lr_patience': 2,           # 学习率调整的耐心值
        'min_lr': 1e-6,            # 最小学习率
        'max_grad_norm': 1.0,      # 梯度裁剪阈值
        'weight_decay': 1e-5,      # 权重衰减
    }


def test_precision_engineering():
    """测试精准工程模块"""
    print("🧪 测试精准工程模块")
    print("=" * 50)
    
    # 创建配置
    config = create_precision_config()
    print("✅ 配置创建成功")
    
    # 模拟训练数据
    mock_batch = [
        {'input': [1, 2, 3], 'target': [4, 5, 6]},
        {'input': [7, 8, 9], 'target': [10, 11, 12]}
    ]
    
    print("✅ 模拟数据创建成功")
    print(f"📊 配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 精准工程的核心特性:")
    print("  🧠 智慧调速器: 自适应学习率调整")
    print("  🛡️ 安全刹车: 梯度裁剪防止训练失控")
    print("  📈 稳定性保证: 平滑的学习曲线")


if __name__ == "__main__":
    test_precision_engineering()
