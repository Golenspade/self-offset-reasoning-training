"""
重构后的评估脚本
使用配置文件和新的包结构
"""

import json
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.hybrid_model import HybridModel


def load_config(config_path="configs/default_config.json"):
    """加载配置文件"""
    config_file = project_root / config_path

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✅ 配置文件加载成功: {config_file}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None


def comprehensive_evaluation(config):
    """综合评估函数"""
    print("=== 自偏移推理训练 - 综合评估 (重构版) ===\n")

    # 初始化tokenizer
    tokenizer = Tokenizer()

    # 创建混合模型
    model_path = project_root / config["paths"]["model_save_path"]
    hybrid_model = HybridModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["model"]["hidden_size"],
        model_path=str(model_path),
    )

    # 加载评估数据
    print("加载评估数据...")
    val_path = project_root / config["data"]["val_path"]
    val_data = load_dataset(
        str(val_path), tokenizer, config["evaluation"]["max_samples"]
    )

    if not val_data:
        print("❌ 评估数据加载失败")
        return None

    print(f"评估数据: {len(val_data)} 样本")

    # 进行评估
    print("\n=== 开始评估 ===")
    results = hybrid_model.evaluate_on_dataset(val_data)

    # 显示结果
    print(f"\n=== 评估结果 ===")
    print(
        f"精确匹配准确率: {results['exact_accuracy']:.2%} ({results['correct_samples']}/{results['total_samples']})"
    )

    stats = results["prediction_statistics"]
    print(f"\n=== 方法使用统计 ===")
    print(f"规则方法成功: {stats['rule_success']} ({stats['rule_success_rate']:.1%})")
    print(
        f"神经网络回退: {stats['neural_fallback']} ({stats['neural_fallback_rate']:.1%})"
    )
    print(f"总预测次数: {stats['total']}")

    print(f"\n=== 方法分布 ===")
    for method, count in results["method_distribution"].items():
        percentage = count / results["total_samples"] * 100
        print(f"{method}: {count} ({percentage:.1f}%)")

    # 保存评估报告
    save_evaluation_report(results, config)

    return results


def save_evaluation_report(results, config):
    """保存评估报告"""
    report_path = project_root / config["paths"]["evaluation_report_path"]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # 准备报告数据
    report = {
        "evaluation_summary": {
            "exact_accuracy": results["exact_accuracy"],
            "total_samples": results["total_samples"],
            "correct_samples": results["correct_samples"],
        },
        "method_statistics": results["prediction_statistics"],
        "method_distribution": results["method_distribution"],
        "config_used": config,
    }

    # 保存报告
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ 评估报告已保存到: {report_path}")


def show_sample_predictions(hybrid_model, val_data, num_samples=10):
    """显示样本预测结果"""
    print(f"\n=== 样本预测展示 (前{num_samples}个) ===")

    for i, sample in enumerate(val_data[:num_samples]):
        input_text = sample["input_text"]
        target_text = sample["target_text"]

        predicted_text, method = hybrid_model.predict(input_text)
        is_correct = predicted_text.strip() == target_text.strip()

        print(f"\n样本 {i+1}:")
        print(f"  输入: {input_text}")
        print(f"  目标: {target_text}")
        print(f"  预测: {predicted_text}")
        print(f"  方法: {method}")
        print(f"  正确: {'✓' if is_correct else '✗'}")


def main():
    """主评估函数"""
    # 加载配置
    config = load_config()
    if config is None:
        print("❌ 无法加载配置文件，退出评估")
        return

    # 进行综合评估
    results = comprehensive_evaluation(config)

    if results is None:
        print("❌ 评估失败")
        return

    # 创建混合模型用于样本展示
    tokenizer = Tokenizer()
    model_path = project_root / config["paths"]["model_save_path"]
    hybrid_model = HybridModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["model"]["hidden_size"],
        model_path=str(model_path),
    )

    # 加载数据用于样本展示
    val_path = project_root / config["data"]["val_path"]
    val_data = load_dataset(str(val_path), tokenizer, 20)  # 只加载20个样本用于展示

    # 显示样本预测
    show_sample_predictions(hybrid_model, val_data, 10)

    # 最终总结
    print(f"\n=== 评估总结 ===")
    if results["exact_accuracy"] >= 0.95:
        print(f"🎉 评估成功！达到了 {results['exact_accuracy']:.1%} 的精确准确率！")
        print(f"这证明了'自偏移推理训练'概念的可行性。")
    else:
        print(f"📊 当前准确率: {results['exact_accuracy']:.1%}")
        print(f"还有改进空间，可以考虑优化模型或增加训练数据。")


if __name__ == "__main__":
    main()
