"""
文件名: breakthrough_training_system_refactored.py
重构后的突破性训练系统
修复了原系统中的关键Bug和设计缺陷

主要改进：
1. 修复验证集训练Bug - 分离训练和评估逻辑
2. 统一训练循环 - 移除冗余的远程训练代码
3. 改进模型接口 - 使用安全的权重管理
4. 重构配置系统 - 统一使用嵌套配置
5. 改进数据流 - 重新设计经验回放和新数据生成
6. 增强异常处理 - 避免吞掉所有异常
"""

import sys
import os
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

# 添加src目录到Python路径（脚本已移动到 scripts/ 下，因此这里取上一级作为项目根目录）
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperienceReplayBuffer:
    """改进的经验回放缓冲区"""
    
    def __init__(self, capacity: int = 15000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, sample: Dict, priority: float = 1.0):
        """添加单个样本"""
        self.buffer.append(sample)
        self.priorities.append(priority)
    
    def push_batch(self, samples: List[Dict], priorities: Optional[List[float]] = None):
        """批量添加样本"""
        if priorities is None:
            priorities = [1.0] * len(samples)
        
        for sample, priority in zip(samples, priorities):
            self.push(sample, priority)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """采样批次数据"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # 基于优先级的采样
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities, replace=False)
        return [self.buffer[i] for i in indices]
    
    def update_priority(self, index: int, priority: float):
        """更新样本优先级"""
        if 0 <= index < len(self.priorities):
            self.priorities[index] = priority
    
    def __len__(self):
        return len(self.buffer)
    
    def utilization(self) -> float:
        """缓冲区利用率"""
        return len(self.buffer) / self.capacity


class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, initial_lr: float = 0.001, patience: int = 3, factor: float = 0.7, min_lr: float = 1e-6):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.best_loss = float('inf')
        self.wait_count = 0
        self.adjustments = 0
    
    def step(self, val_loss: float) -> bool:
        """更新学习率，返回是否进行了调整"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait_count = 0
            return False
        
        self.wait_count += 1
        
        if self.wait_count >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait_count = 0
            self.adjustments += 1
            
            if self.current_lr != old_lr:
                logger.info(f"学习率调整: {old_lr:.6f} -> {self.current_lr:.6f}")
                return True
        
        return False


class BreakthroughTrainingSystem:
    """重构后的突破性训练系统"""
    
    def __init__(self, config: Dict):
        """
        初始化训练系统
        
        Args:
            config: 嵌套配置字典，包含所有训练参数
        """
        self.config = config
        self.tokenizer = Tokenizer()
        
        # 从配置中提取参数
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        precision_config = config.get('precision', {})
        
        # 创建模型
        self.model = ImprovedSimpleModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=model_config.get('hidden_size', 64),
            max_length=model_config.get('max_length', 50),
            learning_rate=training_config.get('initial_lr', 0.001)
        )
        
        # 创建目标网络（用于稳定训练）
        self.target_model = ImprovedSimpleModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=model_config.get('hidden_size', 64),
            max_length=model_config.get('max_length', 50),
            learning_rate=training_config.get('initial_lr', 0.001)
        )
        self.target_model.copy_weights_from(self.model)
        
        # 学习率调度器
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=training_config.get('initial_lr', 0.001),
            patience=precision_config.get('lr_patience', 3),
            factor=precision_config.get('lr_decay_factor', 0.7)
        )
        
        # 经验回放缓冲区
        replay_config = config.get('replay', {})
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=replay_config.get('buffer_size', 15000)
        )
        
        # 训练状态
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'memory_utilization': [],
            'target_updates': [],
            'gradient_health': []
        }
        
        # 训练参数
        self.target_update_frequency = training_config.get('target_update_freq', 10)
        self.target_update_tau = training_config.get('target_update_tau', 0.01)
        self.gradient_clip_threshold = training_config.get('gradient_clip_threshold', 2.0)
        
        logger.info("✅ 突破性训练系统初始化完成")
        logger.info(f"模型参数: {self.model.get_model_info()}")
    
    def prepare_training_data(self, all_data: List[Dict], epoch: int) -> Tuple[List[Dict], List[Dict]]:
        """
        准备训练数据，实现课程学习策略
        
        Args:
            all_data: 所有训练数据
            epoch: 当前epoch
            
        Returns:
            (new_data, replay_data): 新数据和回放数据
        """
        # 课程学习：根据epoch逐步增加数据复杂度
        complexity_levels = ['simple', 'medium', 'complex']
        max_complexity_index = min(epoch // 10, len(complexity_levels) - 1)
        available_complexities = complexity_levels[:max_complexity_index + 1]
        
        # 筛选当前可用的数据
        available_data = [
            sample for sample in all_data 
            if sample.get('complexity', 'simple') in available_complexities
        ]
        
        # 新数据：从可用数据中随机采样
        new_data_size = min(len(available_data) // 4, 100)  # 每次使用25%的可用数据，最多100个
        new_data = random.sample(available_data, new_data_size) if available_data else []
        
        # 回放数据：从经验缓冲区采样
        replay_data_size = min(len(self.replay_buffer), new_data_size * 2)  # 回放数据是新数据的2倍
        replay_data = self.replay_buffer.sample(replay_data_size) if replay_data_size > 0 else []
        
        # 将新数据添加到经验缓冲区
        if new_data:
            self.replay_buffer.push_batch(new_data)
        
        logger.debug(f"Epoch {epoch}: 新数据 {len(new_data)}, 回放数据 {len(replay_data)}, 缓冲区利用率 {self.replay_buffer.utilization():.2f}")
        
        return new_data, replay_data
    
    def train_step(self, sample: Dict) -> Tuple[float, bool]:
        """
        单步训练
        
        Args:
            sample: 训练样本
            
        Returns:
            (loss, gradient_clipped): 损失值和是否进行了梯度裁剪
        """
        try:
            # 准备输入和目标
            input_tokens = self.tokenizer.encode(sample['noisy_prop'])
            target_tokens = self.tokenizer.encode(sample['target_contrapositive'])
            
            if not input_tokens or not target_tokens:
                logger.warning(f"空序列跳过: {sample}")
                return 0.0, False
            
            # 执行训练步骤
            loss = self.model.train_step_improved(input_tokens, target_tokens, self.tokenizer)
            
            # 梯度健康检查（基于损失值的简单检查）
            gradient_clipped = False
            if loss > self.gradient_clip_threshold:
                gradient_clipped = True
                loss = min(loss, self.gradient_clip_threshold)  # 损失裁剪
                logger.debug(f"梯度异常检测，损失被裁剪: {loss}")
            
            return loss, gradient_clipped
            
        except Exception as e:
            logger.error(f"训练步骤失败: {e}, 样本: {sample}")
            return float('inf'), False
    
    def evaluate_step(self, sample: Dict) -> float:
        """
        单步评估（不更新权重）
        
        Args:
            sample: 评估样本
            
        Returns:
            loss: 损失值
        """
        try:
            # 准备输入和目标
            input_tokens = self.tokenizer.encode(sample['noisy_prop'])
            target_tokens = self.tokenizer.encode(sample['target_contrapositive'])
            
            if not input_tokens or not target_tokens:
                return 0.0
            
            # 使用评估方法（不更新权重）
            loss = self.model.evaluate_step(input_tokens, target_tokens, self.tokenizer)
            return loss
            
        except Exception as e:
            logger.error(f"评估步骤失败: {e}, 样本: {sample}")
            return float('inf')

    def train_epoch(self, train_data: List[Dict], val_data: List[Dict], epoch: int) -> Dict:
        """
        训练一个epoch

        Args:
            train_data: 训练数据
            val_data: 验证数据
            epoch: 当前epoch

        Returns:
            epoch_metrics: epoch指标字典
        """
        epoch_start_time = time.time()

        # 准备训练数据（课程学习 + 经验回放）
        new_data, replay_data = self.prepare_training_data(train_data, epoch)
        combined_data = new_data + replay_data

        if not combined_data:
            logger.warning(f"Epoch {epoch}: 没有可用的训练数据")
            return self._create_empty_metrics(epoch)

        # 打乱数据
        random.shuffle(combined_data)

        # 训练阶段
        total_loss = 0.0
        gradient_clips = 0
        successful_steps = 0

        for sample in combined_data:
            loss, clipped = self.train_step(sample)

            if loss != float('inf'):
                total_loss += loss
                successful_steps += 1
                if clipped:
                    gradient_clips += 1

        # 计算训练指标
        avg_train_loss = total_loss / max(successful_steps, 1)
        gradient_health = 1.0 - (gradient_clips / max(successful_steps, 1))

        # 验证阶段
        val_loss = self.evaluate_validation(val_data)

        # 学习率调整
        lr_adjusted = self.lr_scheduler.step(val_loss)
        if lr_adjusted:
            self.model.learning_rate = self.lr_scheduler.current_lr

        # 目标网络更新
        target_updated = False
        if epoch % self.target_update_frequency == 0:
            self.target_model.soft_update_from(self.model, self.target_update_tau)
            target_updated = True
            logger.debug(f"目标网络软更新完成 (tau={self.target_update_tau})")

        # 记录训练历史
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': self.lr_scheduler.current_lr,
            'memory_utilization': self.replay_buffer.utilization(),
            'gradient_health': gradient_health,
            'target_updated': target_updated,
            'new_samples': len(new_data),
            'replay_samples': len(replay_data),
            'successful_steps': successful_steps,
            'epoch_time': time.time() - epoch_start_time
        }

        # 更新历史记录
        for key in ['train_loss', 'val_loss', 'learning_rate', 'memory_utilization', 'gradient_health']:
            if key in epoch_metrics:
                self.training_history[key].append(epoch_metrics[key])

        self.training_history['target_updates'].append(target_updated)

        return epoch_metrics

    def evaluate_validation(self, val_data: List[Dict]) -> float:
        """
        验证集评估（修复后的版本，不会训练模型）

        Args:
            val_data: 验证数据

        Returns:
            avg_val_loss: 平均验证损失
        """
        if not val_data:
            return 0.0

        total_loss = 0.0
        successful_evals = 0

        for sample in val_data:
            loss = self.evaluate_step(sample)  # 使用评估方法，不更新权重

            if loss != float('inf'):
                total_loss += loss
                successful_evals += 1

        avg_val_loss = total_loss / max(successful_evals, 1)

        logger.debug(f"验证完成: {successful_evals}/{len(val_data)} 样本成功, 平均损失: {avg_val_loss:.4f}")

        return avg_val_loss

    def _create_empty_metrics(self, epoch: int) -> Dict:
        """创建空的epoch指标"""
        return {
            'epoch': epoch,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'learning_rate': self.lr_scheduler.current_lr,
            'memory_utilization': self.replay_buffer.utilization(),
            'gradient_health': 1.0,
            'target_updated': False,
            'new_samples': 0,
            'replay_samples': 0,
            'successful_steps': 0,
            'epoch_time': 0.0
        }

    def run_training(self, train_data: List[Dict], val_data: List[Dict],
                    epochs: int = 50, save_frequency: int = 10,
                    output_dir: str = "outputs") -> Dict:
        """
        运行完整的训练流程

        Args:
            train_data: 训练数据
            val_data: 验证数据
            epochs: 训练轮次
            save_frequency: 保存频率
            output_dir: 输出目录

        Returns:
            training_results: 训练结果字典
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"🚀 开始突破性训练: {epochs} epochs")
        logger.info(f"📊 训练数据: {len(train_data)} 样本")
        logger.info(f"📊 验证数据: {len(val_data)} 样本")

        # 初始化经验回放缓冲区
        if train_data:
            # 随机采样初始数据，避免偏差
            initial_samples = random.sample(train_data, min(500, len(train_data)))
            self.replay_buffer.push_batch(initial_samples)
            logger.info(f"经验缓冲区初始化: {len(initial_samples)} 样本")

        training_start_time = time.time()
        best_val_loss = float('inf')

        try:
            for epoch in range(epochs):
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 Epoch {epoch + 1}/{epochs}")
                logger.info(f"{'='*60}")

                # 训练一个epoch
                epoch_metrics = self.train_epoch(train_data, val_data, epoch)

                # 打印epoch结果
                self._log_epoch_results(epoch_metrics)

                # 保存最佳模型
                if epoch_metrics['val_loss'] < best_val_loss:
                    best_val_loss = epoch_metrics['val_loss']
                    best_model_path = os.path.join(output_dir, "best_model.npz")
                    self.model.save_model(best_model_path)
                    logger.info(f"💾 保存最佳模型: {best_model_path}")

                # 定期保存检查点
                if (epoch + 1) % save_frequency == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.npz")
                    self.model.save_model(checkpoint_path)
                    logger.info(f"💾 保存检查点: {checkpoint_path}")

        except KeyboardInterrupt:
            logger.info("⚠️ 训练被用户中断")

        except Exception as e:
            logger.error(f"❌ 训练过程中发生错误: {e}")
            raise

        finally:
            # 保存最终模型和训练历史
            final_model_path = os.path.join(output_dir, "final_model.npz")
            self.model.save_model(final_model_path)

            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)

            total_time = time.time() - training_start_time
            logger.info(f"🎉 训练完成! 总耗时: {total_time:.2f}s")

            return {
                'best_val_loss': best_val_loss,
                'total_epochs': len(self.training_history['train_loss']),
                'training_history': self.training_history,
                'final_model_path': final_model_path,
                'history_path': history_path,
                'total_time': total_time
            }

    def _log_epoch_results(self, metrics: Dict):
        """记录epoch结果"""
        logger.info(f"📊 Epoch {metrics['epoch'] + 1} 结果:")
        logger.info(f"  训练损失: {metrics['train_loss']:.4f}")
        logger.info(f"  验证损失: {metrics['val_loss']:.4f}")
        logger.info(f"  学习率: {metrics['learning_rate']:.6f}")
        logger.info(f"  内存利用率: {metrics['memory_utilization']:.2f}")
        logger.info(f"  梯度健康度: {metrics['gradient_health']:.3f}")
        logger.info(f"  新样本/回放样本: {metrics['new_samples']}/{metrics['replay_samples']}")
        logger.info(f"  成功步骤: {metrics['successful_steps']}")
        logger.info(f"  耗时: {metrics['epoch_time']:.2f}s")
        if metrics['target_updated']:
            logger.info(f"  🎯 目标网络已更新")


def create_breakthrough_config() -> Dict:
    """
    创建突破性训练配置
    统一的嵌套配置结构
    """
    return {
        'model': {
            'hidden_size': 128,
            'max_length': 100
        },
        'training': {
            'initial_lr': 0.001,
            'target_update_freq': 10,
            'target_update_tau': 0.01,
            'gradient_clip_threshold': 2.0
        },
        'precision': {
            'lr_patience': 3,
            'lr_decay_factor': 0.7,
            'min_lr': 1e-6
        },
        'replay': {
            'buffer_size': 15000,
            'initial_fill_ratio': 0.1
        },
        'curriculum': {
            'complexity_progression_epochs': 10,
            'new_data_ratio': 0.25,
            'replay_data_multiplier': 2
        }
    }


def load_training_data(data_dir: str = "data") -> Tuple[List[Dict], List[Dict]]:
    """
    加载训练数据的改进版本
    更安全的文件加载和错误处理
    """
    # 尝试加载不同级别的数据文件
    data_files = [
        ("train_level_3_鲁棒版.json", "val_level_3_鲁棒版.json"),
        ("train_level_2_鲁棒版.json", "val_level_2_鲁棒版.json"),
        ("train_level_1_鲁棒版.json", "val_level_1_鲁棒版.json")
    ]

    for train_file, val_file in data_files:
        train_path = os.path.join(data_dir, train_file)
        val_path = os.path.join(data_dir, val_file)

        if os.path.exists(train_path) and os.path.exists(val_path):
            try:
                logger.info(f"📊 加载数据文件: {train_file}, {val_file}")

                # 尝试加载JSONL格式（每行一个JSON对象）
                train_data = []
                with open(train_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                train_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                val_data = []
                with open(val_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                val_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                if train_data and val_data:
                    logger.info(f"✅ 数据加载成功: 训练 {len(train_data)}, 验证 {len(val_data)}")
                    return train_data, val_data
                else:
                    logger.warning(f"⚠️ 数据文件为空或格式错误: {train_file}, {val_file}")
                    continue

            except Exception as e:
                logger.error(f"❌ 加载数据文件失败 {train_file}: {e}")
                continue

    # 如果所有文件都加载失败，抛出异常
    raise FileNotFoundError(f"在 {data_dir} 目录中未找到可用的训练数据文件")


def main():
    """主函数"""
    print("🚀 启动重构后的突破性训练系统")
    print("=" * 60)

    try:
        # 创建配置
        config = create_breakthrough_config()
        logger.info("✅ 配置创建完成")

        # 加载数据
        train_data, val_data = load_training_data()

        # 创建训练系统
        trainer = BreakthroughTrainingSystem(config)

        # 开始训练
        results = trainer.run_training(
            train_data=train_data,
            val_data=val_data,
            epochs=50,
            save_frequency=10,
            output_dir="outputs/breakthrough_refactored"
        )

        print("\n🎉 训练完成!")
        print(f"📊 最佳验证损失: {results['best_val_loss']:.4f}")
        print(f"📊 总训练轮次: {results['total_epochs']}")
        print(f"📊 总耗时: {results['total_time']:.2f}s")
        print(f"📁 最终模型: {results['final_model_path']}")

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
