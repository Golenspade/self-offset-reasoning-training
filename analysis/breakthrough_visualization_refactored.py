"""
文件名: breakthrough_visualization_refactored.py
重构后的突破性训练可视化
基于真实数据的诚实可视化，移除虚构内容和误导性指标

主要改进：
1. 移除虚构的"三阶段改进对比"图表
2. 修复雷达图中的代理指标问题
3. 增强数据健壮性检查
4. 优化数据预处理
5. 移除无效的report加载
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_breakthrough_data():
    """
    加载突破性训练数据
    只加载实际使用的训练历史数据
    """
    history_path = 'outputs/breakthrough_training/training_history.json'
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        logger.info(f"✅ 成功加载训练历史数据: {history_path}")
        return history
    except FileNotFoundError:
        logger.error(f"❌ 训练历史文件未找到: {history_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析错误: {e}")
        raise


def preprocess_metrics_data(history):
    """
    预处理指标数据，将复杂的嵌套结构转换为易于访问的格式
    
    Args:
        history: 训练历史数据
        
    Returns:
        dict: 预处理后的指标数据
    """
    # 检查数据完整性
    if 'breakthrough_metrics' not in history:
        logger.warning("⚠️ 训练历史中缺少breakthrough_metrics数据")
        return {}
    
    breakthrough_metrics = history['breakthrough_metrics']
    if not breakthrough_metrics:
        logger.warning("⚠️ breakthrough_metrics为空")
        return {}
    
    # 提取所有可能的指标名称
    all_metric_names = set()
    for metrics in breakthrough_metrics:
        if isinstance(metrics, dict):
            all_metric_names.update(metrics.keys())
    
    # 构建时间序列数据
    metrics_over_time = {}
    for metric_name in all_metric_names:
        metrics_over_time[metric_name] = []
        for metrics in breakthrough_metrics:
            if isinstance(metrics, dict):
                metrics_over_time[metric_name].append(metrics.get(metric_name, 0.0))
            else:
                metrics_over_time[metric_name].append(0.0)
    
    logger.info(f"✅ 预处理完成，提取到 {len(all_metric_names)} 个指标")
    return metrics_over_time


def validate_data_completeness(history):
    """
    验证数据完整性，确保有足够的数据进行可视化
    
    Args:
        history: 训练历史数据
        
    Returns:
        dict: 验证结果和统计信息
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'stats': {}
    }
    
    # 检查基本训练数据
    required_fields = ['train_loss', 'val_loss', 'epochs']
    for field in required_fields:
        if field not in history:
            validation_result['is_valid'] = False
            validation_result['warnings'].append(f"缺少必需字段: {field}")
        elif not history[field]:
            validation_result['warnings'].append(f"字段为空: {field}")
        else:
            validation_result['stats'][field] = len(history[field])
    
    # 检查数据长度一致性
    if all(field in history for field in required_fields):
        lengths = [len(history[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            validation_result['warnings'].append(f"数据长度不一致: {dict(zip(required_fields, lengths))}")
    
    # 检查是否有足够的数据点进行分析
    if 'train_loss' in history and len(history['train_loss']) < 2:
        validation_result['warnings'].append("训练数据点过少，无法计算改进趋势")
    
    return validation_result


def create_breakthrough_visualization():
    """创建基于真实数据的突破性训练可视化"""
    
    # 加载和验证数据
    history = load_breakthrough_data()
    validation = validate_data_completeness(history)
    
    if not validation['is_valid']:
        logger.error("❌ 数据验证失败，无法生成可视化")
        for warning in validation['warnings']:
            logger.error(f"  - {warning}")
        return
    
    # 输出警告信息
    for warning in validation['warnings']:
        logger.warning(f"⚠️ {warning}")
    
    # 预处理指标数据
    metrics_over_time = preprocess_metrics_data(history)
    
    # 创建图表 - 调整为2x3布局，移除虚构的三阶段对比图
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('📊 突破性训练系统 - 真实数据可视化报告', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 安全获取数据
    epochs = history.get('epochs', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    # 1. 训练损失趋势
    ax1 = plt.subplot(2, 3, 1)
    if len(epochs) > 0 and len(train_loss) > 0:
        ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='训练损失', alpha=0.8)
        if len(val_loss) > 0:
            ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='验证损失', alpha=0.8)
        
        ax1.set_title('📉 损失曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加改进统计
        if len(train_loss) > 1:
            improvement = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
            ax1.text(0.02, 0.98, f'训练损失改进: {improvement:.1f}%', 
                    transform=ax1.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    verticalalignment='top')
    else:
        ax1.text(0.5, 0.5, '❌ 损失数据不足', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('📉 损失曲线 (数据不足)', fontsize=14)
    
    # 2. 学习率变化
    ax2 = plt.subplot(2, 3, 2)
    learning_rates = history.get('learning_rate', [])
    if len(epochs) > 0 and len(learning_rates) > 0:
        ax2.plot(epochs, learning_rates, 'g-', linewidth=2, alpha=0.8)
        ax2.set_title('📈 学习率调度', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('学习率')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 使用对数刻度更好地显示学习率变化
    else:
        ax2.text(0.5, 0.5, '❌ 学习率数据不足', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('📈 学习率调度 (数据不足)', fontsize=14)
    
    # 3. 梯度健康度
    ax3 = plt.subplot(2, 3, 3)
    gradient_health = metrics_over_time.get('gradient_health', [])
    if len(epochs) > 0 and len(gradient_health) > 0:
        ax3.plot(epochs, gradient_health, 'purple', linewidth=2, alpha=0.8)
        ax3.set_title('🧠 梯度健康度', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('健康度')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # 添加健康度统计
        if gradient_health:
            avg_health = np.mean(gradient_health)
            ax3.axhline(y=avg_health, color='red', linestyle='--', alpha=0.7, 
                       label=f'平均值: {avg_health:.3f}')
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, '❌ 梯度健康度数据不足', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('🧠 梯度健康度 (数据不足)', fontsize=14)
    
    # 4. 内存利用率
    ax4 = plt.subplot(2, 3, 4)
    memory_utilization = history.get('memory_utilization', [])
    if len(epochs) > 0 and len(memory_utilization) > 0:
        ax4.plot(epochs, memory_utilization, 'orange', linewidth=2, alpha=0.8)
        ax4.set_title('💾 内存利用率', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('利用率')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # 添加利用率统计
        if memory_utilization:
            final_util = memory_utilization[-1]
            ax4.text(0.02, 0.98, f'最终利用率: {final_util:.2f}', 
                    transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
                    verticalalignment='top')
    else:
        ax4.text(0.5, 0.5, '❌ 内存利用率数据不足', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('💾 内存利用率 (数据不足)', fontsize=14)
    
    # 5. 真实性能雷达图 (修复后的版本)
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    
    # 获取最终指标 (如果有数据的话)
    if metrics_over_time and any(len(values) > 0 for values in metrics_over_time.values()):
        # 只使用真实的、有意义的指标
        categories = []
        values = []
        
        # 定义真实指标及其获取方式
        real_metrics = {
            '稳定性': metrics_over_time.get('stability_score', []),
            '记忆利用率': history.get('memory_utilization', []),
            '梯度健康': metrics_over_time.get('gradient_health', [])
        }
        
        for category, metric_values in real_metrics.items():
            if metric_values:  # 只添加有数据的指标
                categories.append(category)
                values.append(metric_values[-1])  # 使用最终值
        
        if categories and values:
            # 闭合雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合
            angles += angles[:1]  # 闭合
            
            ax5.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.8)
            ax5.fill(angles, values, alpha=0.25, color='red')
            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels(categories)
            ax5.set_ylim(0, 1)
            ax5.set_title('🎯 真实性能指标\n(基于最终训练结果)', fontsize=14, fontweight='bold', pad=20)
            ax5.grid(True)
        else:
            ax5.text(0.5, 0.5, '❌ 性能指标数据不足', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('🎯 真实性能指标 (数据不足)', fontsize=14)
    else:
        ax5.text(0.5, 0.5, '❌ 性能指标数据不足', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('🎯 真实性能指标 (数据不足)', fontsize=14)
    
    # 6. 训练统计摘要
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')  # 隐藏坐标轴
    
    # 计算真实的训练统计
    stats_text = "📊 训练统计摘要\n\n"
    
    if validation['stats']:
        stats_text += f"数据点数量: {validation['stats'].get('epochs', 0)}\n"
    
    if len(train_loss) > 1:
        initial_loss = train_loss[0]
        final_loss = train_loss[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        stats_text += f"训练损失改进: {improvement:.1f}%\n"
        stats_text += f"初始损失: {initial_loss:.4f}\n"
        stats_text += f"最终损失: {final_loss:.4f}\n"
    
    if len(val_loss) > 1:
        best_val_loss = min(val_loss)
        stats_text += f"最佳验证损失: {best_val_loss:.4f}\n"
    
    if gradient_health:
        avg_gradient_health = np.mean(gradient_health)
        stats_text += f"平均梯度健康度: {avg_gradient_health:.3f}\n"
    
    if memory_utilization:
        final_memory_util = memory_utilization[-1]
        stats_text += f"最终内存利用率: {final_memory_util:.3f}\n"
    
    # 添加数据质量信息
    if validation['warnings']:
        stats_text += f"\n⚠️ 数据质量警告:\n"
        for warning in validation['warnings'][:3]:  # 只显示前3个警告
            stats_text += f"• {warning}\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'outputs/breakthrough_training/breakthrough_visualization_honest.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ 可视化图表已保存: {output_path}")
    
    plt.show()
    
    return fig


def main():
    """主函数"""
    print("📊 启动重构后的突破性训练可视化")
    print("=" * 60)
    print("🎯 特点:")
    print("  ✅ 基于真实训练数据")
    print("  ✅ 移除虚构的改进对比")
    print("  ✅ 修复误导性指标")
    print("  ✅ 增强数据健壮性")
    print("  ✅ 诚实的性能展示")
    print("=" * 60)
    
    try:
        fig = create_breakthrough_visualization()
        print("\n🎉 可视化生成完成!")
        print("📁 输出文件: outputs/breakthrough_training/breakthrough_visualization_honest.png")
        
    except Exception as e:
        logger.error(f"❌ 可视化生成失败: {e}")
        raise


if __name__ == "__main__":
    main()
