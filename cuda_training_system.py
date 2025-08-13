"""
文件名: cuda_training_system.py
CUDA加速训练系统
基于breakthrough_training_system.py的GPU优化版本
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from cuda_utils import CUDAManager
from logic_transformer.data_utils import Tokenizer
from model import create_cuda_model, get_model_memory_usage

logger = logging.getLogger(__name__)


class CUDABreakthroughTraining:
    """CUDA加速的突破性训练系统"""
    
    def __init__(self, config: Dict):
        """
        初始化CUDA训练系统
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.tokenizer = Tokenizer()
        
        # CUDA管理器
        self.cuda_manager = CUDAManager(
            memory_fraction=config.get('gpu_memory_fraction', 0.8),
            auto_optimize=True
        )
        self.device = self.cuda_manager.device
        
        # 创建CUDA优化模型
        self.model, _ = create_cuda_model(
            vocab_size=self.tokenizer.vocab_size,
            device=self.device,
            use_mixed_precision=config.get('use_mixed_precision', True),
            d_model=config.get('hidden_size', 128),
            nhead=config.get('num_heads', 8),
            num_encoder_layers=config.get('num_encoder_layers', 3),
            num_decoder_layers=config.get('num_decoder_layers', 3),
            dim_feedforward=config.get('dim_feedforward', 512),
            max_len=config.get('max_length', 100)
        )
        
        # 优化器配置
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_decay_factor', 0.5),
            patience=config.get('lr_patience', 3),
            verbose=True,
            min_lr=1e-7
        )
        
        # 混合精度训练
        self.use_amp = (config.get('use_mixed_precision', True) and 
                       self.device.type == 'cuda' and 
                       self.cuda_manager.supports_mixed_precision())
        
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("✅ 启用自动混合精度训练")
        else:
            self.scaler = None
            logger.info("ℹ️ 使用标准精度训练")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.PAD_TOKEN,
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gpu_memory': [],
            'epoch_time': []
        }
        
        # 梯度累积
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # 早停
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.patience_counter = 0
        
        logger.info(f"🚀 CUDA训练系统初始化完成")
        logger.info(f"📍 设备: {self.device}")
        logger.info(f"🔥 混合精度: {self.use_amp}")
        logger.info(f"📊 梯度累积步数: {self.gradient_accumulation_steps}")
    
    def prepare_batch_cuda(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        准备CUDA批次数据
        
        Args:
            batch_data: 批次数据列表
            
        Returns:
            (src_batch, tgt_input, tgt_output): 源序列、目标输入、目标输出
        """
        if not batch_data:
            return None, None, None
        
        batch_size = len(batch_data)
        
        # 计算最大长度
        max_src_len = max(len(self.tokenizer.encode(item['noisy_prop'])) for item in batch_data)
        max_tgt_len = max(len(self.tokenizer.encode(item['target_contrapositive'])) for item in batch_data)
        
        # 创建张量
        src_batch = torch.full(
            (max_src_len, batch_size), 
            self.tokenizer.PAD_TOKEN, 
            dtype=torch.long, 
            device=self.device
        )
        
        tgt_batch = torch.full(
            (max_tgt_len + 1, batch_size),  # +1 for START_TOKEN
            self.tokenizer.PAD_TOKEN,
            dtype=torch.long,
            device=self.device
        )
        
        # 填充数据
        for i, item in enumerate(batch_data):
            # 编码源序列
            src_tokens = self.tokenizer.encode(item['noisy_prop'])
            src_len = len(src_tokens)
            src_batch[:src_len, i] = torch.tensor(src_tokens, device=self.device)
            
            # 编码目标序列（添加START_TOKEN）
            tgt_tokens = [self.tokenizer.START_TOKEN] + self.tokenizer.encode(item['target_contrapositive'])
            tgt_len = len(tgt_tokens)
            tgt_batch[:tgt_len, i] = torch.tensor(tgt_tokens, device=self.device)
        
        # 分离输入和输出
        tgt_input = tgt_batch[:-1]  # 去掉最后一个token作为输入
        tgt_output = tgt_batch[1:]  # 去掉第一个token作为目标
        
        return src_batch, tgt_input, tgt_output
    
    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建注意力掩码
        
        Args:
            src: 源序列 [seq_len, batch_size]
            tgt: 目标序列 [seq_len, batch_size]
            
        Returns:
            (src_mask, tgt_mask): 源掩码和目标掩码
        """
        src_seq_len = src.size(0)
        tgt_seq_len = tgt.size(0)
        
        # 源序列掩码（padding掩码）
        src_mask = (src == self.tokenizer.PAD_TOKEN).transpose(0, 1)  # [batch_size, seq_len]
        
        # 目标序列掩码（因果掩码 + padding掩码）
        tgt_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len, device=self.device), 
            diagonal=1
        ).bool()
        
        tgt_padding_mask = (tgt == self.tokenizer.PAD_TOKEN).transpose(0, 1)
        
        return src_mask, tgt_mask, tgt_padding_mask
    
    def train_step_cuda(self, batch_data: List[Dict], accumulate_gradients: bool = False) -> Dict:
        """
        CUDA加速的训练步骤
        
        Args:
            batch_data: 批次数据
            accumulate_gradients: 是否累积梯度
            
        Returns:
            训练指标字典
        """
        if not accumulate_gradients:
            self.optimizer.zero_grad()
        
        # 准备批次数据
        src_batch, tgt_input, tgt_output = self.prepare_batch_cuda(batch_data)
        
        if src_batch is None:
            return {'loss': 0.0, 'grad_norm': 0.0}
        
        # 创建掩码
        src_mask, tgt_mask, tgt_padding_mask = self.create_masks(src_batch, tgt_input)
        
        try:
            if self.use_amp:
                # 混合精度训练
                with autocast():
                    output = self.model(
                        src_batch,
                        tgt_input,
                        src_key_padding_mask=src_mask,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_padding_mask
                    )
                    
                    # 计算损失
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt_output.reshape(-1)
                    )
                    
                    # 梯度累积
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                if not accumulate_gradients:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get('max_grad_norm', 1.0)
                    )
                    
                    # 优化器步骤
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = 0.0
                
            else:
                # 标准精度训练
                output = self.model(
                    src_batch,
                    tgt_input,
                    src_key_padding_mask=src_mask,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                )
                
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if not accumulate_gradients:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                else:
                    grad_norm = 0.0
            
            return {
                'loss': loss.item() * self.gradient_accumulation_steps,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("⚠️ GPU内存不足，清理缓存...")
                torch.cuda.empty_cache()
                return {'loss': float('inf'), 'grad_norm': 0.0, 'learning_rate': 0.0}
            else:
                raise e
    
    def validate_cuda(self, val_data: List[Dict]) -> Dict:
        """
        CUDA验证
        
        Args:
            val_data: 验证数据
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        batch_size = self.config.get('batch_size', 16)
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                
                src_batch, tgt_input, tgt_output = self.prepare_batch_cuda(batch)
                
                if src_batch is None:
                    continue
                
                src_mask, tgt_mask, tgt_padding_mask = self.create_masks(src_batch, tgt_input)
                
                try:
                    if self.use_amp:
                        with autocast():
                            output = self.model(
                                src_batch,
                                tgt_input,
                                src_key_padding_mask=src_mask,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_padding_mask
                            )
                    else:
                        output = self.model(
                            src_batch,
                            tgt_input,
                            src_key_padding_mask=src_mask,
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_padding_mask
                        )
                    
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt_output.reshape(-1)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, filepath: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
            'tokenizer_vocab_size': self.tokenizer.vocab_size
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {})
        
        epoch = checkpoint['epoch']
        logger.info(f"📂 检查点已加载: {filepath} (epoch {epoch})")

        return epoch

    def train_epoch_cuda(self, train_data: List[Dict], val_data: List[Dict], epoch: int) -> Dict:
        """
        CUDA加速的epoch训练

        Args:
            train_data: 训练数据
            val_data: 验证数据
            epoch: 当前epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        self.current_epoch = epoch

        batch_size = self.config.get('batch_size', 16)
        total_loss = 0.0
        num_batches = 0

        # 随机打乱训练数据
        random.shuffle(train_data)

        epoch_start_time = time.time()

        # 训练循环
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            # 梯度累积
            accumulate = (i + batch_size) % (batch_size * self.gradient_accumulation_steps) != 0

            with self.cuda_manager.memory_monitor(f"训练批次 {i//batch_size + 1}"):
                metrics = self.train_step_cuda(batch, accumulate_gradients=accumulate)

            if metrics['loss'] != float('inf'):
                total_loss += metrics['loss']
                num_batches += 1

            # 定期打印进度和GPU状态
            if (i // batch_size + 1) % self.config.get('log_frequency', 50) == 0:
                memory_info = self.cuda_manager.get_memory_info()
                if memory_info and 'error' not in memory_info:
                    logger.info(
                        f"Batch {i//batch_size + 1}/{len(train_data)//batch_size}: "
                        f"Loss={metrics['loss']:.4f}, "
                        f"LR={metrics['learning_rate']:.6f}, "
                        f"GPU={memory_info['allocated_memory']:.1f}GB/"
                        f"{memory_info['total_memory']:.1f}GB"
                    )

        # 计算平均训练损失
        avg_train_loss = total_loss / max(num_batches, 1)

        # 验证阶段
        logger.info("🔍 开始验证...")
        val_metrics = self.validate_cuda(val_data)
        val_loss = val_metrics['val_loss']

        # 更新学习率
        self.scheduler.step(val_loss)
        current_lr = self.optimizer.param_groups[0]['lr']

        # 早停检查
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # 记录训练历史
        epoch_time = time.time() - epoch_start_time

        self.training_history['train_loss'].append(avg_train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['learning_rate'].append(current_lr)
        self.training_history['epoch_time'].append(epoch_time)

        # 记录GPU内存使用
        if self.device.type == 'cuda':
            memory_info = self.cuda_manager.get_memory_info()
            if memory_info and 'error' not in memory_info:
                self.training_history['gpu_memory'].append(memory_info['allocated_memory'])

        return {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'epoch_time': epoch_time,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'early_stop': self.patience_counter >= self.early_stopping_patience
        }

    def run_cuda_training(self, train_data: List[Dict], val_data: List[Dict],
                         output_dir: str = "outputs") -> Dict:
        """
        运行完整的CUDA训练流程

        Args:
            train_data: 训练数据
            val_data: 验证数据
            output_dir: 输出目录

        Returns:
            训练结果字典
        """
        os.makedirs(output_dir, exist_ok=True)

        epochs = self.config.get('epochs', 50)
        save_frequency = self.config.get('save_frequency', 10)

        logger.info(f"🚀 开始CUDA训练: {epochs} epochs")
        logger.info(f"📊 训练数据: {len(train_data)} 样本")
        logger.info(f"📊 验证数据: {len(val_data)} 样本")

        training_start_time = time.time()

        try:
            for epoch in range(epochs):
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 Epoch {epoch + 1}/{epochs}")
                logger.info(f"{'='*60}")

                # 训练一个epoch
                epoch_metrics = self.train_epoch_cuda(train_data, val_data, epoch)

                # 打印epoch结果
                logger.info(f"📊 Epoch {epoch + 1} 结果:")
                logger.info(f"  训练损失: {epoch_metrics['train_loss']:.4f}")
                logger.info(f"  验证损失: {epoch_metrics['val_loss']:.4f}")
                logger.info(f"  最佳验证损失: {epoch_metrics['best_val_loss']:.4f}")
                logger.info(f"  学习率: {epoch_metrics['learning_rate']:.6f}")
                logger.info(f"  耗时: {epoch_metrics['epoch_time']:.2f}s")
                logger.info(f"  早停计数: {epoch_metrics['patience_counter']}/{self.early_stopping_patience}")

                # 保存检查点
                if (epoch + 1) % save_frequency == 0 or epoch_metrics['val_loss'] == self.best_val_loss:
                    checkpoint_path = os.path.join(
                        output_dir,
                        f"cuda_checkpoint_epoch_{epoch + 1}.pth"
                    )
                    self.save_checkpoint(epoch, epoch_metrics['val_loss'], checkpoint_path)

                # 早停检查
                if epoch_metrics['early_stop']:
                    logger.info(f"🛑 早停触发 (patience: {self.early_stopping_patience})")
                    break

        except KeyboardInterrupt:
            logger.info("⚠️ 训练被用户中断")

        except Exception as e:
            logger.error(f"❌ 训练过程中发生错误: {e}")
            raise

        finally:
            # 保存最终模型
            final_model_path = os.path.join(output_dir, "final_cuda_model.pth")
            self.save_checkpoint(self.current_epoch, self.best_val_loss, final_model_path)

            # 保存训练历史
            history_path = os.path.join(output_dir, "cuda_training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)

            total_time = time.time() - training_start_time
            logger.info(f"🎉 训练完成! 总耗时: {total_time:.2f}s")

            # 清理GPU缓存
            self.cuda_manager.clear_cache()

        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'training_history': self.training_history,
            'final_model_path': final_model_path,
            'history_path': history_path
        }
