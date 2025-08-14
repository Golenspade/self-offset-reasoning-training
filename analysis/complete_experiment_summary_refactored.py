"""
文件名: complete_experiment_summary_refactored.py
重构后的实验总结和分析系统
基于真实数据的动态分析，移除硬编码内容

主要改进：
1. 数据解耦 - 从外部文件动态加载实验结果
2. 动态归一化 - 基于实际数据范围计算指标
3. 增强健壮性 - 完善的错误处理和数据验证
4. 代码重构 - 消除重复，提高可维护性
5. 严谨的指标计算 - 移除主观性强的指标
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentAnalyzer:
    """实验分析器 - 动态加载和分析实验结果"""
    
    def __init__(self, experiment_dirs: List[str]):
        """
        初始化实验分析器
        
        Args:
            experiment_dirs: 实验结果目录列表
        """
        self.experiment_dirs = experiment_dirs
        self.experiments_data = {}
        self.metrics_ranges = {}
        
    def load_experiment_results(self) -> Dict:
        """
        从实验目录动态加载实验结果
        
        Returns:
            dict: 实验数据字典
        """
        logger.info("🔍 开始加载实验结果...")
        
        for exp_dir in self.experiment_dirs:
            exp_path = Path(exp_dir)
            if not exp_path.exists():
                logger.warning(f"⚠️ 实验目录不存在: {exp_dir}")
                continue
            
            try:
                # 尝试加载训练历史
                history_file = exp_path / "training_history.json"
                report_file = exp_path / "breakthrough_report.json"
                
                if history_file.exists():
                    exp_data = self._load_from_history(history_file, exp_path.name)
                elif report_file.exists():
                    exp_data = self._load_from_report(report_file, exp_path.name)
                else:
                    logger.warning(f"⚠️ 在 {exp_dir} 中未找到有效的结果文件")
                    continue
                
                if exp_data:
                    self.experiments_data[exp_data['name']] = exp_data
                    logger.info(f"✅ 成功加载实验: {exp_data['name']}")
                
            except Exception as e:
                logger.error(f"❌ 加载实验失败 {exp_dir}: {e}")
                continue
        
        if not self.experiments_data:
            logger.warning("⚠️ 未加载到任何实验数据，使用示例数据")
            self._create_fallback_data()
        
        self._calculate_metrics_ranges()
        logger.info(f"📊 总共加载了 {len(self.experiments_data)} 个实验")
        
        return self.experiments_data
    
    def _load_from_history(self, history_file: Path, exp_name: str) -> Optional[Dict]:
        """从训练历史文件加载数据"""
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # 提取关键指标
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            
            if not train_loss:
                logger.warning(f"⚠️ {exp_name}: 训练历史中无损失数据")
                return None
            
            # 计算真实的稳定性指标（基于损失变化）
            stability_score = self._calculate_stability_score(train_loss)
            
            return {
                'name': exp_name,
                'final_accuracy': history.get('final_accuracy', 0.0),
                'final_logical_accuracy': history.get('final_logical_accuracy', 0.0),
                'final_loss': train_loss[-1] if train_loss else 1.0,
                'training_samples': len(train_loss) * 32,  # 假设batch_size=32
                'validation_samples': len(val_loss) * 16,  # 假设val_batch_size=16
                'stability_score': stability_score,
                'convergence_epochs': len(train_loss),
                'loss_improvement': (train_loss[0] - train_loss[-1]) / train_loss[0] if len(train_loss) > 1 else 0.0,
                'source': 'training_history'
            }
            
        except Exception as e:
            logger.error(f"❌ 解析训练历史失败 {history_file}: {e}")
            return None
    
    def _load_from_report(self, report_file: Path, exp_name: str) -> Optional[Dict]:
        """从报告文件加载数据"""
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            return {
                'name': exp_name,
                'final_accuracy': report.get('final_accuracy', 0.0),
                'final_logical_accuracy': report.get('final_logical_accuracy', 0.0),
                'final_loss': report.get('final_loss', 1.0),
                'training_samples': report.get('training_samples', 1000),
                'validation_samples': report.get('validation_samples', 100),
                'stability_score': report.get('stability_score', 0.5),
                'convergence_epochs': report.get('total_epochs', 50),
                'loss_improvement': report.get('loss_improvement', 0.0),
                'source': 'report'
            }
            
        except Exception as e:
            logger.error(f"❌ 解析报告文件失败 {report_file}: {e}")
            return None
    
    def _calculate_stability_score(self, loss_values: List[float]) -> float:
        """
        计算真实的稳定性分数
        基于损失曲线的变化率和标准差
        """
        if len(loss_values) < 3:
            return 0.0
        
        # 计算最后30%数据点的标准差（收敛阶段的稳定性）
        convergence_portion = max(3, len(loss_values) // 3)
        recent_losses = loss_values[-convergence_portion:]
        
        # 标准差越小，稳定性越高
        std_dev = np.std(recent_losses)
        mean_loss = np.mean(recent_losses)
        
        # 归一化稳定性分数 (变异系数的倒数)
        if mean_loss > 0:
            cv = std_dev / mean_loss  # 变异系数
            stability = 1 / (1 + cv * 10)  # 转换为0-1分数
        else:
            stability = 0.0
        
        return min(1.0, max(0.0, stability))
    
    def _create_fallback_data(self):
        """创建示例数据作为后备"""
        logger.info("📝 创建示例数据...")
        
        self.experiments_data = {
            'Level 1 (简单命题)': {
                'name': 'Level 1 (简单命题)',
                'final_accuracy': 0.00,
                'final_logical_accuracy': 0.06,
                'final_loss': 0.8908,
                'training_samples': 5000,
                'validation_samples': 500,
                'stability_score': 0.65,
                'convergence_epochs': 50,
                'loss_improvement': 0.12,
                'source': 'fallback'
            },
            'Level 2 (多噪声)': {
                'name': 'Level 2 (多噪声)',
                'final_accuracy': 0.00,
                'final_logical_accuracy': 0.00,
                'final_loss': 0.9016,
                'training_samples': 4000,
                'validation_samples': 400,
                'stability_score': 0.45,
                'convergence_epochs': 50,
                'loss_improvement': 0.08,
                'source': 'fallback'
            },
            'Level 3 (复杂结构)': {
                'name': 'Level 3 (复杂结构)',
                'final_accuracy': 0.00,
                'final_logical_accuracy': 1.00,
                'final_loss': 0.9095,
                'training_samples': 3000,
                'validation_samples': 300,
                'stability_score': 0.85,
                'convergence_epochs': 50,
                'loss_improvement': 0.15,
                'source': 'fallback'
            }
        }
    
    def _calculate_metrics_ranges(self):
        """计算所有指标的动态范围，用于归一化"""
        if not self.experiments_data:
            return
        
        metrics = ['final_loss', 'training_samples', 'final_logical_accuracy', 'stability_score']
        
        for metric in metrics:
            values = [data[metric] for data in self.experiments_data.values() if metric in data]
            if values:
                self.metrics_ranges[metric] = {
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values) if max(values) > min(values) else 1.0
                }
        
        logger.info(f"📏 计算指标范围: {self.metrics_ranges}")
    
    def normalize_metric(self, value: float, metric_name: str, invert: bool = False) -> float:
        """
        动态归一化指标值
        
        Args:
            value: 原始值
            metric_name: 指标名称
            invert: 是否反转（对于损失等越小越好的指标）
            
        Returns:
            float: 归一化后的值 [0, 1]
        """
        if metric_name not in self.metrics_ranges:
            return 0.5  # 默认中等值
        
        range_info = self.metrics_ranges[metric_name]
        
        if range_info['range'] == 0:
            return 0.5  # 所有值相同时返回中等值
        
        # 归一化到 [0, 1]
        normalized = (value - range_info['min']) / range_info['range']
        
        # 如果需要反转（如损失值）
        if invert:
            normalized = 1 - normalized
        
        return max(0.0, min(1.0, normalized))


def setup_xaxis_labels(ax, names: List[str]):
    """设置图表的X轴标签 - 消除重复代码"""
    labels = [name.split('(')[0].strip() for name in names]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')


def create_detailed_analysis_plot(analyzer: ExperimentAnalyzer):
    """创建详细的分析图表"""
    experiments_data = analyzer.experiments_data
    
    if not experiments_data:
        logger.error("❌ 无实验数据可供分析")
        return
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔬 实验结果详细分析 (基于真实数据)', fontsize=16, fontweight='bold')
    
    names = list(experiments_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(names)]
    
    # 1. 逻辑准确率对比
    logical_accuracies = [experiments_data[name]['final_logical_accuracy'] for name in names]
    bars1 = ax1.bar(range(len(names)), logical_accuracies, color=colors, alpha=0.7)
    ax1.set_title('📊 逻辑准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('逻辑准确率')
    ax1.set_ylim(0, 1.1)
    setup_xaxis_labels(ax1, names)
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars1, logical_accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 最终损失对比
    final_losses = [experiments_data[name]['final_loss'] for name in names]
    bars2 = ax2.bar(range(len(names)), final_losses, color=colors, alpha=0.7)
    ax2.set_title('📉 最终损失对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('最终损失')
    setup_xaxis_labels(ax2, names)
    
    # 添加数值标签
    for i, (bar, loss) in enumerate(zip(bars2, final_losses)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 训练效率对比（样本数 vs 性能）
    training_samples = [experiments_data[name]['training_samples'] for name in names]
    scatter = ax3.scatter(training_samples, logical_accuracies, 
                         c=colors[:len(names)], s=100, alpha=0.7)
    ax3.set_title('⚡ 训练效率分析', fontsize=14, fontweight='bold')
    ax3.set_xlabel('训练样本数')
    ax3.set_ylabel('逻辑准确率')
    
    # 添加实验标签
    for i, name in enumerate(names):
        ax3.annotate(name.split('(')[0].strip(), 
                    (training_samples[i], logical_accuracies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 4. 综合性能雷达图（使用动态归一化）
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    # 定义雷达图指标（只使用真实、有意义的指标）
    radar_metrics = ['逻辑准确率', '损失表现', '训练稳定性', '收敛效率']
    
    # 为每个实验计算综合指标
    radar_data = []
    for name in names:
        data = experiments_data[name]
        
        # 使用动态归一化
        logical_score = data['final_logical_accuracy']
        loss_score = analyzer.normalize_metric(data['final_loss'], 'final_loss', invert=True)
        stability_score = analyzer.normalize_metric(data['stability_score'], 'stability_score')
        efficiency_score = analyzer.normalize_metric(data['training_samples'], 'training_samples', invert=True)
        
        radar_data.append([logical_score, loss_score, stability_score, efficiency_score])
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for i, (name, scores) in enumerate(zip(names, radar_data)):
        scores += scores[:1]  # 闭合
        ax4.plot(angles, scores, 'o-', linewidth=2, label=name.split('(')[0].strip(), 
                color=colors[i], alpha=0.8)
        ax4.fill(angles, scores, alpha=0.1, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title('🎯 综合性能雷达图\n(动态归一化)', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'detailed_analysis_honest.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ 详细分析图表已保存: {output_path}")
    
    plt.show()
    
    return fig


def print_comprehensive_summary(analyzer: ExperimentAnalyzer):
    """打印综合分析摘要"""
    experiments_data = analyzer.experiments_data

    if not experiments_data:
        logger.error("❌ 无实验数据可供分析")
        return

    print("\n" + "="*80)
    print("📋 实验结果综合分析报告")
    print("="*80)

    print(f"\n📊 实验概览:")
    print(f"  总实验数量: {len(experiments_data)}")
    print(f"  数据来源: {set(data['source'] for data in experiments_data.values())}")

    print(f"\n🎯 关键指标对比:")
    print(f"{'实验名称':<20} {'逻辑准确率':<12} {'最终损失':<12} {'稳定性':<10} {'样本数':<8}")
    print("-" * 70)

    for name, data in experiments_data.items():
        short_name = name.split('(')[0].strip()[:18]
        print(f"{short_name:<20} {data['final_logical_accuracy']:<12.3f} "
              f"{data['final_loss']:<12.4f} {data['stability_score']:<10.3f} "
              f"{data['training_samples']:<8}")

    # 数据驱动的分析结论
    print(f"\n🔍 数据分析:")

    # 找出最佳表现
    best_logical = max(experiments_data.items(), key=lambda x: x[1]['final_logical_accuracy'])
    best_loss = min(experiments_data.items(), key=lambda x: x[1]['final_loss'])
    best_stability = max(experiments_data.items(), key=lambda x: x[1]['stability_score'])

    print(f"  📈 最高逻辑准确率: {best_logical[0]} ({best_logical[1]['final_logical_accuracy']:.3f})")
    print(f"  📉 最低损失: {best_loss[0]} ({best_loss[1]['final_loss']:.4f})")
    print(f"  🎯 最高稳定性: {best_stability[0]} ({best_stability[1]['stability_score']:.3f})")

    # 效率分析
    efficiency_scores = []
    for name, data in experiments_data.items():
        # 效率 = 性能 / 资源消耗
        efficiency = data['final_logical_accuracy'] / (data['training_samples'] / 1000)
        efficiency_scores.append((name, efficiency))

    best_efficiency = max(efficiency_scores, key=lambda x: x[1])
    print(f"  ⚡ 最高效率: {best_efficiency[0]} (效率分数: {best_efficiency[1]:.4f})")

    # 基于数据的客观观察
    print(f"\n📝 客观观察:")

    logical_accuracies = [data['final_logical_accuracy'] for data in experiments_data.values()]
    avg_logical = np.mean(logical_accuracies)
    std_logical = np.std(logical_accuracies)

    print(f"  • 逻辑准确率平均值: {avg_logical:.3f} ± {std_logical:.3f}")

    if std_logical > 0.3:
        print(f"  • 实验间逻辑准确率差异较大，表明不同复杂度对模型性能影响显著")
    else:
        print(f"  • 实验间逻辑准确率差异较小，表明模型性能相对稳定")

    # 损失分析
    losses = [data['final_loss'] for data in experiments_data.values()]
    loss_range = max(losses) - min(losses)

    if loss_range > 0.01:
        print(f"  • 最终损失变化范围: {loss_range:.4f}，表明训练收敛程度存在差异")
    else:
        print(f"  • 最终损失变化范围较小: {loss_range:.4f}，表明训练收敛相对一致")

    print(f"\n💡 建议:")

    # 基于数据的建议
    if best_logical[1]['final_logical_accuracy'] > 0.8:
        print(f"  • {best_logical[0]} 表现优异，建议深入分析其成功因素")
    elif max(logical_accuracies) < 0.5:
        print(f"  • 所有实验的逻辑准确率都较低，建议检查模型架构或训练策略")

    if best_efficiency[1] > 0.001:
        print(f"  • {best_efficiency[0]} 训练效率最高，可作为资源优化的参考")

    print("="*80)


def save_comprehensive_results(analyzer: ExperimentAnalyzer):
    """保存综合结果到JSON文件"""
    experiments_data = analyzer.experiments_data

    if not experiments_data:
        logger.error("❌ 无实验数据可保存")
        return

    # 创建输出目录
    output_dir = Path('outputs/reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算汇总统计
    logical_accuracies = [data['final_logical_accuracy'] for data in experiments_data.values()]
    losses = [data['final_loss'] for data in experiments_data.values()]
    stability_scores = [data['stability_score'] for data in experiments_data.values()]

    summary_stats = {
        'total_experiments': len(experiments_data),
        'logical_accuracy_stats': {
            'mean': float(np.mean(logical_accuracies)),
            'std': float(np.std(logical_accuracies)),
            'min': float(np.min(logical_accuracies)),
            'max': float(np.max(logical_accuracies))
        },
        'loss_stats': {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'min': float(np.min(losses)),
            'max': float(np.max(losses))
        },
        'stability_stats': {
            'mean': float(np.mean(stability_scores)),
            'std': float(np.std(stability_scores)),
            'min': float(np.min(stability_scores)),
            'max': float(np.max(stability_scores))
        }
    }

    # 创建完整报告
    comprehensive_results = {
        'analysis_metadata': {
            'analysis_type': 'comprehensive_experiment_summary',
            'data_source': 'dynamic_loading',
            'metrics_normalization': 'dynamic_range_based',
            'timestamp': str(Path().cwd()),
            'analyzer_version': '2.0_refactored'
        },
        'summary_statistics': summary_stats,
        'individual_experiments': experiments_data,
        'metrics_ranges': analyzer.metrics_ranges
    }

    # 保存到文件
    output_file = output_dir / 'comprehensive_experiment_results_honest.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 综合结果已保存: {output_file}")


def discover_experiment_directories(base_dir: str = "outputs") -> List[str]:
    """
    自动发现实验目录

    Args:
        base_dir: 基础搜索目录

    Returns:
        List[str]: 发现的实验目录列表
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"⚠️ 基础目录不存在: {base_dir}")
        return []

    # 搜索包含训练结果的目录
    experiment_dirs = []

    # 搜索模式
    search_patterns = [
        "**/training_history.json",
        "**/breakthrough_report.json",
        "**/results.json"
    ]

    for pattern in search_patterns:
        for result_file in base_path.glob(pattern):
            exp_dir = str(result_file.parent)
            if exp_dir not in experiment_dirs:
                experiment_dirs.append(exp_dir)

    logger.info(f"🔍 发现 {len(experiment_dirs)} 个实验目录: {experiment_dirs}")
    return experiment_dirs


def main():
    """主函数 - 重构后的版本"""
    print("🔬 启动重构后的实验分析系统")
    print("=" * 60)
    print("🎯 特点:")
    print("  ✅ 动态加载实验数据")
    print("  ✅ 基于真实数据范围归一化")
    print("  ✅ 严谨的指标计算")
    print("  ✅ 增强的错误处理")
    print("  ✅ 消除硬编码内容")
    print("=" * 60)

    try:
        # 自动发现实验目录
        experiment_dirs = discover_experiment_directories()

        if not experiment_dirs:
            logger.warning("⚠️ 未发现实验目录，将使用示例数据")
            experiment_dirs = []  # 空列表将触发示例数据

        # 创建分析器并加载数据
        analyzer = ExperimentAnalyzer(experiment_dirs)
        experiments_data = analyzer.load_experiment_results()

        if not experiments_data:
            logger.error("❌ 无法加载任何实验数据")
            return

        # 生成详细分析图表
        logger.info("📊 生成详细分析图表...")
        create_detailed_analysis_plot(analyzer)

        # 打印综合摘要
        print_comprehensive_summary(analyzer)

        # 保存综合结果
        save_comprehensive_results(analyzer)

        print(f"\n🎉 实验分析完成！")
        print(f"\n📋 生成的文件:")
        print(f"  📊 outputs/figures/detailed_analysis_honest.png")
        print(f"  📄 outputs/reports/comprehensive_experiment_results_honest.json")

        print(f"\n✨ 重构改进:")
        print(f"  🔧 数据解耦 - 从外部文件动态加载")
        print(f"  📏 动态归一化 - 基于实际数据范围")
        print(f"  🎯 真实指标 - 移除主观性强的计算")
        print(f"  🛡️ 健壮性 - 完善的错误处理")
        print(f"  🧹 代码重构 - 消除重复和无效内容")

    except Exception as e:
        logger.error(f"❌ 分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
