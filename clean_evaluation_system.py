"""
文件名: clean_evaluation_system.py
清理后的评估系统 - 去除所有"装模作样"的代码
真正有用、直接有效的评估功能
"""

import json
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel

# 直接导入逻辑工具函数
def verify_equivalence(pred, target):
    """简化的逻辑等价性检查"""
    # 标准化处理
    pred = pred.strip().replace(' ', '')
    target = target.strip().replace(' ', '')

    # 直接比较
    if pred == target:
        return True

    # 简单的等价性检查
    equivalences = [
        ('p->q', '~q->~p'),
        ('~p->q', '~q->p'),
        ('p->~q', 'q->~p'),
        ('~p->~q', 'q->p')
    ]

    for eq1, eq2 in equivalences:
        if (pred == eq1 and target == eq2) or (pred == eq2 and target == eq1):
            return True

    return False

def to_contrapositive(prop):
    """简化的逆否命题转换"""
    prop = prop.strip().replace(' ', '')

    # 基本的逆否命题转换规则
    if prop == 'p->q':
        return '~q->~p'
    elif prop == '~p->q':
        return '~q->p'
    elif prop == 'p->~q':
        return 'q->~p'
    elif prop == '~p->~q':
        return 'q->p'
    else:
        return prop  # 无法转换时返回原命题


class CleanEvaluationSystem:
    """清理后的评估系统 - 没有装模作样的代码"""
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = None
        
    def load_model(self, model_path: str) -> bool:
        """加载模型"""
        try:
            self.model = ImprovedSimpleModel(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=128,
                max_length=50
            )
            self.model.load_model(model_path)
            print(f"✅ 模型加载成功: {model_path}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def evaluate_model_performance(self, test_data: List[Dict], max_samples: int = 100) -> Dict:
        """评估模型性能 - 直接、有效的评估"""
        if not self.model:
            print("❌ 请先加载模型")
            return {}
        
        print(f"\n🎯 评估模型性能 (样本数: {min(len(test_data), max_samples)})")
        
        exact_correct = 0
        logical_correct = 0
        total_samples = min(len(test_data), max_samples)
        
        detailed_results = []
        
        for i, sample in enumerate(test_data[:total_samples]):
            try:
                # 模型预测
                prediction = self.model.predict(sample['input'], self.tokenizer)
                pred_text = self.tokenizer.decode(prediction).strip()
                target_text = sample['target_text'].strip()
                
                # 精确匹配
                exact_match = pred_text == target_text
                if exact_match:
                    exact_correct += 1
                
                # 逻辑等价性检查
                logical_match = verify_equivalence(pred_text, target_text)
                if logical_match:
                    logical_correct += 1
                
                # 记录详细结果
                detailed_results.append({
                    'input': sample.get('input_text', ''),
                    'target': target_text,
                    'prediction': pred_text,
                    'exact_match': exact_match,
                    'logical_match': logical_match
                })
                
                # 进度显示
                if (i + 1) % 20 == 0:
                    print(f"  进度: {i + 1}/{total_samples}")
                    
            except Exception as e:
                print(f"  样本 {i} 评估失败: {e}")
                continue
        
        # 计算准确率
        exact_accuracy = exact_correct / total_samples
        logical_accuracy = logical_correct / total_samples
        
        results = {
            'exact_accuracy': exact_accuracy,
            'logical_accuracy': logical_accuracy,
            'exact_correct': exact_correct,
            'logical_correct': logical_correct,
            'total_samples': total_samples,
            'detailed_results': detailed_results
        }
        
        print(f"\n📊 评估结果:")
        print(f"  精确匹配准确率: {exact_accuracy:.2%} ({exact_correct}/{total_samples})")
        print(f"  逻辑等价准确率: {logical_accuracy:.2%} ({logical_correct}/{total_samples})")
        
        return results
    
    def compare_with_real_baseline(self, test_data: List[Dict], max_samples: int = 100) -> Dict:
        """与真正的基线方法比较 - 不是装模作样的版本"""
        print(f"\n🔍 真实基线比较 (样本数: {min(len(test_data), max_samples)})")
        
        total_samples = min(len(test_data), max_samples)
        
        # 基线1: 随机预测
        random_correct = 0
        
        # 基线2: 规则方法 (使用真实的输入数据)
        rule_correct = 0
        rule_attempted = 0
        
        for sample in test_data[:total_samples]:
            target_text = sample['target_text'].strip()
            input_text = sample.get('input_text', '').strip()
            
            # 随机基线 - 从常见的逆否命题模式中随机选择
            common_patterns = ['~p -> ~q', 'q -> p', '~q -> ~p', 'p -> q']
            random_prediction = np.random.choice(common_patterns)
            if random_prediction == target_text:
                random_correct += 1
            
            # 规则基线 - 使用真实的输入数据
            if input_text:
                try:
                    rule_prediction = to_contrapositive(input_text)
                    rule_attempted += 1
                    if rule_prediction.strip() == target_text:
                        rule_correct += 1
                except Exception:
                    # 规则方法失败时不计入
                    pass
        
        # 计算基线准确率
        random_accuracy = random_correct / total_samples
        rule_accuracy = rule_correct / rule_attempted if rule_attempted > 0 else 0.0
        
        baseline_results = {
            'random_accuracy': random_accuracy,
            'rule_accuracy': rule_accuracy,
            'random_correct': random_correct,
            'rule_correct': rule_correct,
            'rule_attempted': rule_attempted,
            'total_samples': total_samples
        }
        
        print(f"📊 基线结果:")
        print(f"  随机预测准确率: {random_accuracy:.2%} ({random_correct}/{total_samples})")
        print(f"  规则方法准确率: {rule_accuracy:.2%} ({rule_correct}/{rule_attempted})")
        
        return baseline_results
    
    def analyze_errors(self, evaluation_results: Dict, max_errors: int = 10) -> Dict:
        """错误分析 - 真正有用的分析"""
        print(f"\n🔍 错误分析 (显示前{max_errors}个错误)")
        
        detailed_results = evaluation_results.get('detailed_results', [])
        errors = [r for r in detailed_results if not r['logical_match']]
        
        error_patterns = {
            'format_error': 0,      # 格式错误
            'logic_error': 0,       # 逻辑错误
            'symbol_error': 0,      # 符号错误
            'complete_wrong': 0     # 完全错误
        }
        
        print(f"错误样本分析:")
        for i, error in enumerate(errors[:max_errors]):
            print(f"\n  错误 {i+1}:")
            print(f"    输入: {error['input']}")
            print(f"    目标: {error['target']}")
            print(f"    预测: {error['prediction']}")
            
            # 简单的错误分类
            if not error['prediction']:
                error_patterns['format_error'] += 1
                print(f"    类型: 格式错误 (空预测)")
            elif error['exact_match']:
                error_patterns['logic_error'] += 1
                print(f"    类型: 逻辑错误 (格式正确但逻辑不等价)")
            elif any(sym in error['prediction'] for sym in ['p', 'q', '~', '->']):
                error_patterns['symbol_error'] += 1
                print(f"    类型: 符号错误 (包含逻辑符号但不正确)")
            else:
                error_patterns['complete_wrong'] += 1
                print(f"    类型: 完全错误")
        
        # 统计所有错误的模式
        for error in errors:
            if not error['prediction']:
                error_patterns['format_error'] += 1
            elif error['exact_match']:
                error_patterns['logic_error'] += 1
            elif any(sym in error['prediction'] for sym in ['p', 'q', '~', '->']):
                error_patterns['symbol_error'] += 1
            else:
                error_patterns['complete_wrong'] += 1
        
        print(f"\n📊 错误模式统计:")
        total_errors = len(errors)
        for pattern, count in error_patterns.items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"  {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        return {
            'error_patterns': error_patterns,
            'total_errors': total_errors,
            'error_examples': errors[:max_errors]
        }
    
    def comprehensive_evaluation(self, model_path: str, test_data_path: str) -> Dict:
        """综合评估 - 一次性完成所有有用的评估"""
        print("🎯 开始综合评估")
        print("=" * 60)
        
        # 1. 加载模型
        if not self.load_model(model_path):
            return {}
        
        # 2. 加载测试数据
        try:
            test_data = load_dataset(test_data_path, self.tokenizer, max_samples=200)
            if not test_data:
                print("❌ 无法加载测试数据")
                return {}
            print(f"✅ 加载了 {len(test_data)} 个测试样本")
        except Exception as e:
            print(f"❌ 测试数据加载失败: {e}")
            return {}
        
        # 3. 模型性能评估
        model_results = self.evaluate_model_performance(test_data, max_samples=100)
        
        # 4. 基线比较
        baseline_results = self.compare_with_real_baseline(test_data, max_samples=100)
        
        # 5. 错误分析
        error_analysis = self.analyze_errors(model_results, max_errors=5)
        
        # 6. 综合报告
        comprehensive_results = {
            'model_performance': model_results,
            'baseline_comparison': baseline_results,
            'error_analysis': error_analysis,
            'summary': {
                'model_vs_random': model_results['logical_accuracy'] / baseline_results['random_accuracy'] if baseline_results['random_accuracy'] > 0 else float('inf'),
                'model_vs_rule': model_results['logical_accuracy'] / baseline_results['rule_accuracy'] if baseline_results['rule_accuracy'] > 0 else float('inf'),
                'improvement_over_random': (model_results['logical_accuracy'] - baseline_results['random_accuracy']) * 100,
                'improvement_over_rule': (model_results['logical_accuracy'] - baseline_results['rule_accuracy']) * 100
            }
        }
        
        print(f"\n🏆 综合评估总结:")
        print(f"  模型逻辑准确率: {model_results['logical_accuracy']:.2%}")
        print(f"  相比随机提升: {comprehensive_results['summary']['improvement_over_random']:.1f}个百分点")
        print(f"  相比规则提升: {comprehensive_results['summary']['improvement_over_rule']:.1f}个百分点")
        
        return comprehensive_results


def main():
    """主函数 - 直接执行有用的评估，不装模作样"""
    print("🧹 清理后的评估系统")
    print("去除所有装模作样的代码，只保留真正有用的功能")
    print("=" * 60)
    
    # 创建评估系统
    evaluator = CleanEvaluationSystem()
    
    # 寻找最佳模型
    model_candidates = [
        'outputs/breakthrough_training/models/best_breakthrough_model_epoch_23.npz',
        'outputs/trained_models/best_model.npz',
        'outputs/formal_training/models/formal_model_epoch_10.npz'
    ]
    
    best_model = None
    for model_path in model_candidates:
        if os.path.exists(model_path):
            best_model = model_path
            break
    
    if not best_model:
        print("❌ 找不到可用的模型文件")
        return
    
    # 寻找测试数据
    test_data_candidates = [
        'data/val_level_3_鲁棒版.json',
        'data/val.json',
        'data/val_level_1_鲁棒版.json'
    ]
    
    test_data_path = None
    for data_path in test_data_candidates:
        if os.path.exists(data_path):
            test_data_path = data_path
            break
    
    if not test_data_path:
        print("❌ 找不到可用的测试数据")
        return
    
    # 执行综合评估
    results = evaluator.comprehensive_evaluation(best_model, test_data_path)
    
    # 保存结果
    if results:
        output_path = 'outputs/clean_evaluation_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 评估结果已保存: {output_path}")
        print("\n🎉 清理后的评估完成！没有任何装模作样的代码。")


if __name__ == "__main__":
    main()
