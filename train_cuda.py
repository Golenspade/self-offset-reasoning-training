"""
文件名: train_cuda.py
CUDA加速训练主脚本
支持GPU加速的自偏移推理训练
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from cuda_training_system import CUDABreakthroughTraining
from cuda_utils import CUDAManager, print_cuda_summary


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_training_data(data_dir: str = "data") -> tuple:
    """
    加载训练数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        (train_data, val_data): 训练和验证数据
    """
    # 尝试加载不同级别的数据
    data_files = [
        ("train_level_3_鲁棒版.json", "val_level_3_鲁棒版.json"),
        ("train_level_2_鲁棒版.json", "val_level_2_鲁棒版.json"),
        ("train_level_1_鲁棒版.json", "val_level_1_鲁棒版.json"),
        ("train_data.json", "val_data.json")
    ]
    
    for train_file, val_file in data_files:
        train_path = os.path.join(data_dir, train_file)
        val_path = os.path.join(data_dir, val_file)
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"📊 加载数据文件: {train_file}, {val_file}")
            
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            with open(val_path, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
            
            return train_data, val_data
    
    raise FileNotFoundError(f"在 {data_dir} 目录中未找到训练数据文件")


def create_training_config(args) -> Dict:
    """
    创建训练配置
    
    Args:
        args: 命令行参数
        
    Returns:
        训练配置字典
    """
    # 基础配置
    config = {
        # 模型参数
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'max_length': args.max_length,
        
        # 训练参数
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_grad_norm': args.max_grad_norm,
        
        # CUDA参数
        'use_mixed_precision': args.use_mixed_precision,
        'gpu_memory_fraction': args.gpu_memory_fraction,
        
        # 调度器参数
        'lr_decay_factor': args.lr_decay_factor,
        'lr_patience': args.lr_patience,
        
        # 早停参数
        'early_stopping_patience': args.early_stopping_patience,
        
        # 保存参数
        'save_frequency': args.save_frequency,
        'log_frequency': args.log_frequency,
        
        # 正则化
        'label_smoothing': args.label_smoothing
    }
    
    return config


def optimize_batch_size(cuda_manager: CUDAManager, initial_batch_size: int, 
                       model_size_estimate: float = 0.1) -> int:
    """
    根据GPU内存自动优化批次大小
    
    Args:
        cuda_manager: CUDA管理器
        initial_batch_size: 初始批次大小
        model_size_estimate: 模型大小估计(GB)
        
    Returns:
        优化后的批次大小
    """
    if cuda_manager.device.type != 'cuda':
        return initial_batch_size
    
    optimal_batch_size = cuda_manager.get_optimal_batch_size(
        model_memory_gb=model_size_estimate,
        max_batch_size=initial_batch_size * 2
    )
    
    # 确保批次大小是合理的
    optimal_batch_size = max(4, min(optimal_batch_size, 128))
    
    if optimal_batch_size != initial_batch_size:
        print(f"🎯 批次大小优化: {initial_batch_size} -> {optimal_batch_size}")
    
    return optimal_batch_size


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CUDA加速的自偏移推理训练")
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--output-dir', type=str, default='outputs/cuda_training', help='输出目录')
    
    # 模型参数
    parser.add_argument('--hidden-size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-encoder-layers', type=int, default=4, help='编码器层数')
    parser.add_argument('--num-decoder-layers', type=int, default=4, help='解码器层数')
    parser.add_argument('--dim-feedforward', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--max-length', type=int, default=128, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪阈值')
    
    # CUDA参数
    parser.add_argument('--use-mixed-precision', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8, help='GPU内存使用比例')
    parser.add_argument('--auto-batch-size', action='store_true', help='自动优化批次大小')
    
    # 调度器参数
    parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--lr-patience', type=int, default=3, help='学习率调度器耐心值')
    
    # 早停参数
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='早停耐心值')
    
    # 保存和日志参数
    parser.add_argument('--save-frequency', type=int, default=10, help='保存频率')
    parser.add_argument('--log-frequency', type=int, default=50, help='日志频率')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    
    # 正则化参数
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = os.path.join(args.output_dir, 'training.log')
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    
    print("🚀 启动CUDA加速训练系统")
    print("=" * 60)
    
    # 检查CUDA环境
    print_cuda_summary()
    
    # 创建CUDA管理器
    cuda_manager = CUDAManager(
        memory_fraction=args.gpu_memory_fraction,
        auto_optimize=True
    )
    
    print(f"\n📍 使用设备: {cuda_manager.device}")
    
    # 加载数据
    print(f"\n📊 加载训练数据...")
    try:
        train_data, val_data = load_training_data(args.data_dir)
        print(f"✅ 数据加载成功:")
        print(f"  训练样本: {len(train_data):,}")
        print(f"  验证样本: {len(val_data):,}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # 创建训练配置
    config = create_training_config(args)
    
    # 自动优化批次大小
    if args.auto_batch_size:
        config['batch_size'] = optimize_batch_size(
            cuda_manager, 
            config['batch_size'],
            model_size_estimate=config['hidden_size'] * config['num_encoder_layers'] / 1000
        )
    
    print(f"\n🔧 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 创建训练系统
    print(f"\n🏗️ 初始化CUDA训练系统...")
    trainer = CUDABreakthroughTraining(config)
    
    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"📂 恢复训练: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print(f"\n🎯 开始CUDA加速训练...")
    print(f"起始epoch: {start_epoch}")
    
    try:
        # 记录开始时间
        training_start_time = time.time()
        
        # 运行训练
        results = trainer.run_cuda_training(
            train_data=train_data,
            val_data=val_data,
            output_dir=args.output_dir
        )
        
        # 计算总时间
        total_time = time.time() - training_start_time
        
        print(f"\n🎉 训练完成!")
        print(f"📊 最终结果:")
        print(f"  最佳验证损失: {results['best_val_loss']:.4f}")
        print(f"  总训练轮次: {results['total_epochs']}")
        print(f"  总耗时: {total_time:.2f}s ({total_time/3600:.2f}h)")
        print(f"  最终模型: {results['final_model_path']}")
        print(f"  训练历史: {results['history_path']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return 1
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
