"""
文件名: formal_training_50_epochs.py
正式的50轮训练
使用混合系统进行完整的自偏移推理训练
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel
from hybrid_logic_system import HybridLogicSystem


class FormalTrainingSystem:
    """正式训练系统"""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = None
        self.hybrid_system = None
        
        # 训练参数
        self.total_epochs = 50
        self.batch_size = 8
        self.learning_rate = 0.002  # 稍微降低学习率
        
        # 记录训练历史
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_exact_acc': [],
            'val_logical_acc': [],
            'hybrid_acc': [],
            'training_time': []
        }
        
        # 创建输出目录
        os.makedirs('outputs/formal_training', exist_ok=True)
        os.makedirs('outputs/formal_training/models', exist_ok=True)
        os.makedirs('outputs/formal_training/figures', exist_ok=True)
        
    def initialize_model(self):
        """初始化模型"""
        print("🚀 初始化模型...")
        
        self.model = ImprovedSimpleModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=128,
            max_length=50,
            learning_rate=self.learning_rate
        )
        
        # 尝试加载已有的最佳模型作为起点
        existing_model_path = 'outputs/trained_models/robust_model_Level_1_鲁棒版.npz'
        if os.path.exists(existing_model_path):
            if self.model.load_model(existing_model_path):
                print(f"✅ 加载已有模型: {existing_model_path}")
            else:
                print(f"⚠️ 无法加载已有模型，从头开始训练")
        else:
            print(f"📝 从头开始训练新模型")
        
        # 创建混合系统
        self.hybrid_system = HybridLogicSystem(self.model, self.tokenizer)
        
    def load_training_data(self):
        """加载训练数据"""
        print("📚 加载训练数据...")
        
        # 加载鲁棒数据集
        train_files = [
            'data/train_level_1_鲁棒版.json',
            'data/train_level_2_鲁棒版.json',
            'data/train_level_3_鲁棒版.json'
        ]
        
        val_files = [
            'data/val_level_1_鲁棒版.json',
            'data/val_level_2_鲁棒版.json',
            'data/val_level_3_鲁棒版.json'
        ]
        
        # 合并所有训练数据
        all_train_data = []
        all_val_data = []
        
        for train_file in train_files:
            if os.path.exists(train_file):
                data = load_dataset(train_file, self.tokenizer, 1000)  # 每个级别1000样本
                if data:
                    all_train_data.extend(data)
                    print(f"  ✅ 加载训练数据: {train_file} ({len(data)} 样本)")
        
        for val_file in val_files:
            if os.path.exists(val_file):
                data = load_dataset(val_file, self.tokenizer, 100)  # 每个级别100样本
                if data:
                    all_val_data.extend(data)
                    print(f"  ✅ 加载验证数据: {val_file} ({len(data)} 样本)")
        
        print(f"📊 总训练样本: {len(all_train_data)}")
        print(f"📊 总验证样本: {len(all_val_data)}")
        
        return all_train_data, all_val_data
    
    def evaluate_model(self, val_data: List[Dict], epoch: int) -> Tuple[float, float, float]:
        """评估模型性能"""
        if not val_data:
            return 0.0, 0.0, 0.0
        
        correct_exact = 0
        correct_logical = 0
        correct_hybrid = 0
        total = min(len(val_data), 50)  # 评估50个样本
        
        for i, sample in enumerate(val_data[:total]):
            try:
                # 1. 原始模型预测
                predicted_tokens = self.model.predict(sample['input'], self.tokenizer)
                predicted_text = self.tokenizer.decode(predicted_tokens).strip()
                target_text = sample['target_text'].strip()
                
                # 精确匹配
                if predicted_text == target_text:
                    correct_exact += 1
                    correct_logical += 1
                    correct_hybrid += 1
                else:
                    # 逻辑匹配（基本结构正确）
                    if (len(predicted_text) > 5 and 
                        '->' in predicted_text and 
                        '~' in predicted_text and
                        not predicted_text.startswith('-> -> ->')):
                        correct_logical += 1
                
                # 2. 混合系统预测
                try:
                    neural_output, hybrid_output, intent = self.hybrid_system.generate_hybrid_solution(sample['input_text'])
                    if hybrid_output.strip() == target_text:
                        correct_hybrid += 1
                    elif hybrid_output.replace(' ', '') == target_text.replace(' ', ''):
                        correct_hybrid += 1  # 允许空格差异
                except:
                    pass
                
            except Exception as e:
                continue
        
        exact_acc = correct_exact / total if total > 0 else 0
        logical_acc = correct_logical / total if total > 0 else 0
        hybrid_acc = correct_hybrid / total if total > 0 else 0
        
        return exact_acc, logical_acc, hybrid_acc
    
    def train_epoch(self, train_data: List[Dict], epoch: int) -> float:
        """训练一个epoch"""
        total_loss = 0
        num_batches = 0
        
        # 打乱数据
        np.random.shuffle(train_data)
        
        # 批次训练
        for i in range(0, len(train_data), self.batch_size):
            batch = train_data[i:i+self.batch_size]
            batch_loss = 0
            
            for sample in batch:
                loss = self.model.train_step_improved(sample['input'], sample['target'], self.tokenizer)
                batch_loss += loss
            
            total_loss += batch_loss / len(batch)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        # 保存模型
        if is_best:
            model_path = f'outputs/formal_training/models/best_model_epoch_{epoch}.npz'
        else:
            model_path = f'outputs/formal_training/models/model_epoch_{epoch}.npz'
        
        self.model.save_model(model_path)
        
        # 保存训练历史
        history_path = 'outputs/formal_training/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def plot_training_progress(self):
        """绘制训练进度"""
        if not self.training_history['epochs']:
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('正式训练50轮 - 自偏移推理训练进度', fontsize=16, fontweight='bold')
        
        epochs = self.training_history['epochs']
        
        # 1. 训练损失
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_title('训练损失', fontsize=14, fontweight='bold')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值')
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率对比
        ax2.plot(epochs, [acc * 100 for acc in self.training_history['val_exact_acc']], 
                'r-', linewidth=2, marker='s', markersize=3, label='精确匹配')
        ax2.plot(epochs, [acc * 100 for acc in self.training_history['val_logical_acc']], 
                'g-', linewidth=2, marker='^', markersize=3, label='逻辑正确')
        ax2.plot(epochs, [acc * 100 for acc in self.training_history['hybrid_acc']], 
                'purple', linewidth=2, marker='D', markersize=3, label='混合系统')
        ax2.set_title('验证准确率对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练时间
        if self.training_history['training_time']:
            ax3.plot(epochs, self.training_history['training_time'], 'orange', linewidth=2, marker='o', markersize=3)
            ax3.set_title('每轮训练时间', fontsize=14, fontweight='bold')
            ax3.set_xlabel('训练轮次')
            ax3.set_ylabel('时间 (秒)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 学习效率（准确率提升速度）
        if len(self.training_history['hybrid_acc']) > 1:
            acc_diff = np.diff(self.training_history['hybrid_acc'])
            ax4.plot(epochs[1:], acc_diff, 'teal', linewidth=2, marker='v', markersize=3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('混合系统准确率变化率', fontsize=14, fontweight='bold')
            ax4.set_xlabel('训练轮次')
            ax4.set_ylabel('准确率变化')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('outputs/formal_training/figures/training_progress.png', dpi=300, bbox_inches='tight')
        plt.savefig('outputs/formal_training/figures/training_progress.pdf', bbox_inches='tight')
        plt.show()
    
    def run_formal_training(self):
        """运行正式训练"""
        print("🎯 开始正式的50轮训练")
        print("=" * 80)
        
        # 初始化
        self.initialize_model()
        train_data, val_data = self.load_training_data()
        
        if not train_data:
            print("❌ 没有可用的训练数据")
            return
        
        best_hybrid_acc = 0.0
        
        print(f"\n🚀 开始训练循环...")
        print(f"总轮次: {self.total_epochs}")
        print(f"训练样本: {len(train_data)}")
        print(f"验证样本: {len(val_data)}")
        print("=" * 80)
        
        for epoch in range(1, self.total_epochs + 1):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_data, epoch)
            
            # 评估模型
            exact_acc, logical_acc, hybrid_acc = self.evaluate_model(val_data, epoch)
            
            # 记录时间
            epoch_time = time.time() - start_time
            
            # 更新历史记录
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_exact_acc'].append(exact_acc)
            self.training_history['val_logical_acc'].append(logical_acc)
            self.training_history['hybrid_acc'].append(hybrid_acc)
            self.training_history['training_time'].append(epoch_time)
            
            # 检查是否是最佳模型
            is_best = hybrid_acc > best_hybrid_acc
            if is_best:
                best_hybrid_acc = hybrid_acc
            
            # 打印进度
            print(f"Epoch {epoch:2d}/{self.total_epochs}: "
                  f"Loss={train_loss:.4f}, "
                  f"精确={exact_acc:.1%}, "
                  f"逻辑={logical_acc:.1%}, "
                  f"混合={hybrid_acc:.1%}, "
                  f"时间={epoch_time:.1f}s"
                  f"{' 🏆' if is_best else ''}")
            
            # 定期保存检查点
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 定期绘制进度图
            if epoch % 10 == 0:
                self.plot_training_progress()
        
        # 最终保存和总结
        self.save_checkpoint(self.total_epochs, False)
        self.plot_training_progress()
        
        print(f"\n🎉 训练完成！")
        print(f"最佳混合系统准确率: {best_hybrid_acc:.2%}")
        print(f"最终混合系统准确率: {hybrid_acc:.2%}")
        print(f"训练历史已保存到: outputs/formal_training/")


def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    
    # 创建训练系统
    training_system = FormalTrainingSystem()
    
    # 运行正式训练
    training_system.run_formal_training()


if __name__ == "__main__":
    main()
