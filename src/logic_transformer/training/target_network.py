"""
文件名: src/logic_transformer/training/target_network.py
第三阶段：架构革命 - 目标网络系统
实现稳定的"北极星"，彻底解决追逐移动目标的问题
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TargetNetworkSystem:
    """目标网络系统 - 稳定的学习指导者"""

    def __init__(self, learning_model, config: Dict):
        self.learning_model = learning_model
        self.config = config

        # 创建目标网络 - 学习网络的完全副本
        self.target_model = copy.deepcopy(learning_model)

        # 冻结目标网络的参数，它不通过梯度下降更新
        for param in self.target_model.parameters():
            param.requires_grad = False

        # 软更新参数
        self.tau = config.get("tau", 1e-3)  # 软更新系数
        self.update_frequency = config.get("update_frequency", 1)  # 更新频率
        self.update_counter = 0

        # 稳定性监控
        self.stability_metrics = {
            "parameter_divergence": [],
            "output_consistency": [],
            "learning_stability": [],
        }

        logger.info("🌟 目标网络系统初始化完成")
        logger.info(f"  软更新系数 τ: {self.tau}")
        logger.info(f"  更新频率: {self.update_frequency}")

    def soft_update(self):
        """软更新目标网络 - 缓慢吸收学习网络的知识"""
        param_divergence = 0.0
        param_count = 0

        for target_param, learning_param in zip(
            self.target_model.parameters(), self.learning_model.parameters()
        ):
            # 计算参数差异（用于监控稳定性）
            param_diff = torch.norm(target_param.data - learning_param.data).item()
            param_divergence += param_diff
            param_count += 1

            # 软更新：target = τ * learning + (1-τ) * target
            target_param.data.copy_(
                self.tau * learning_param.data + (1.0 - self.tau) * target_param.data
            )

        # 记录参数差异
        avg_divergence = param_divergence / param_count if param_count > 0 else 0.0
        self.stability_metrics["parameter_divergence"].append(avg_divergence)

        self.update_counter += 1

        if self.update_counter % 100 == 0:
            logger.debug(
                f"目标网络软更新 #{self.update_counter}, 平均参数差异: {avg_divergence:.6f}"
            )

    def hard_update(self):
        """硬更新目标网络 - 完全复制学习网络"""
        self.target_model.load_state_dict(self.learning_model.state_dict())
        logger.info("执行了目标网络硬更新")

    def should_update(self) -> bool:
        """判断是否应该更新目标网络"""
        return self.update_counter % self.update_frequency == 0

    def compute_stable_targets(self, inputs: List, tokenizer) -> List:
        """使用目标网络计算稳定的目标值"""
        self.target_model.eval()

        stable_targets = []

        with torch.no_grad():
            for input_seq in inputs:
                try:
                    # 使用目标网络生成稳定的预测
                    target_prediction = self.target_model.predict(input_seq, tokenizer)
                    stable_targets.append(target_prediction)
                except Exception as e:
                    logger.warning(f"目标网络预测失败: {e}")
                    stable_targets.append([])

        return stable_targets

    def evaluate_consistency(self, test_inputs: List, tokenizer) -> float:
        """评估学习网络和目标网络的一致性"""
        if not test_inputs:
            return 1.0

        consistency_scores = []

        for input_seq in test_inputs[:10]:  # 只测试前10个样本
            try:
                # 学习网络预测
                learning_pred = self.learning_model.predict(input_seq, tokenizer)

                # 目标网络预测
                with torch.no_grad():
                    target_pred = self.target_model.predict(input_seq, tokenizer)

                # 计算一致性（简单的序列相似度）
                if len(learning_pred) == len(target_pred):
                    matches = sum(
                        1 for a, b in zip(learning_pred, target_pred) if a == b
                    )
                    consistency = matches / len(learning_pred) if learning_pred else 0.0
                else:
                    consistency = 0.0

                consistency_scores.append(consistency)

            except Exception as e:
                logger.warning(f"一致性评估失败: {e}")
                continue

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        self.stability_metrics["output_consistency"].append(avg_consistency)

        return avg_consistency

    def get_stability_report(self) -> Dict:
        """获取稳定性报告"""
        if not self.stability_metrics["parameter_divergence"]:
            return {"status": "no_data"}

        param_div = self.stability_metrics["parameter_divergence"]
        output_cons = self.stability_metrics["output_consistency"]

        return {
            "update_count": self.update_counter,
            "parameter_divergence": {
                "mean": np.mean(param_div),
                "std": np.std(param_div),
                "trend": (
                    "stable"
                    if np.std(param_div[-10:]) < np.std(param_div)
                    else "unstable"
                ),
            },
            "output_consistency": {
                "mean": np.mean(output_cons) if output_cons else 0.0,
                "latest": output_cons[-1] if output_cons else 0.0,
            },
            "stability_score": self._compute_stability_score(),
        }

    def _compute_stability_score(self) -> float:
        """计算综合稳定性分数"""
        if not self.stability_metrics["parameter_divergence"]:
            return 0.0

        # 参数稳定性：差异越小越稳定
        param_stability = 1.0 / (
            1.0 + np.mean(self.stability_metrics["parameter_divergence"][-10:])
        )

        # 输出一致性：一致性越高越稳定
        output_stability = (
            np.mean(self.stability_metrics["output_consistency"][-5:])
            if self.stability_metrics["output_consistency"]
            else 0.0
        )

        # 综合分数
        stability_score = 0.6 * param_stability + 0.4 * output_stability

        return min(stability_score, 1.0)

    def save_target_network(self, filepath: str):
        """保存目标网络"""
        torch.save(
            {
                "target_model_state_dict": self.target_model.state_dict(),
                "learning_model_state_dict": self.learning_model.state_dict(),
                "config": self.config,
                "update_counter": self.update_counter,
                "stability_metrics": self.stability_metrics,
            },
            filepath,
        )
        logger.info(f"目标网络已保存: {filepath}")

    def load_target_network(self, filepath: str):
        """加载目标网络"""
        checkpoint = torch.load(filepath)
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.learning_model.load_state_dict(checkpoint["learning_model_state_dict"])
        self.update_counter = checkpoint["update_counter"]
        self.stability_metrics = checkpoint["stability_metrics"]
        logger.info(f"目标网络已加载: {filepath}")


class StabilizedTrainingLoop:
    """稳定化训练循环 - 整合目标网络的训练系统"""

    def __init__(self, target_system: TargetNetworkSystem, config: Dict):
        self.target_system = target_system
        self.config = config

        # 训练参数
        self.stability_check_frequency = config.get("stability_check_frequency", 10)
        self.min_stability_threshold = config.get("min_stability_threshold", 0.7)

        logger.info("🎯 稳定化训练循环初始化完成")

    def train_step_with_stability(self, batch_data: List[Dict], tokenizer) -> Dict:
        """执行稳定化训练步骤"""

        # 1. 正常的学习网络训练
        learning_results = self._train_learning_network(batch_data, tokenizer)

        # 2. 检查是否需要更新目标网络
        if self.target_system.should_update():
            self.target_system.soft_update()

        # 3. 定期进行稳定性检查
        stability_info = {}
        if self.target_system.update_counter % self.stability_check_frequency == 0:
            test_inputs = [sample["input"] for sample in batch_data[:5]]
            consistency = self.target_system.evaluate_consistency(
                test_inputs, tokenizer
            )
            stability_report = self.target_system.get_stability_report()

            stability_info = {
                "consistency": consistency,
                "stability_score": stability_report.get("stability_score", 0.0),
            }

            # 如果稳定性过低，考虑硬更新
            if (
                stability_report.get("stability_score", 0.0)
                < self.min_stability_threshold
            ):
                logger.warning(
                    f"稳定性过低 ({stability_report.get('stability_score', 0.0):.3f})，考虑调整训练策略"
                )

        return {
            **learning_results,
            **stability_info,
            "target_updates": self.target_system.update_counter,
        }

    def _train_learning_network(self, batch_data: List[Dict], tokenizer) -> Dict:
        """训练学习网络"""
        # 这里应该调用实际的训练逻辑
        # 暂时返回模拟结果
        return {"learning_loss": 0.5, "learning_accuracy": 0.8}


def test_target_network():
    """测试目标网络系统"""
    print("🧪 测试目标网络系统")
    print("=" * 50)

    # 模拟一个简单的模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

        def predict(self, input_seq, tokenizer):
            return [1, 2, 3]  # 模拟预测

    # 创建配置
    config = {"tau": 1e-3, "update_frequency": 1, "stability_check_frequency": 5}

    # 创建目标网络系统
    learning_model = MockModel()
    target_system = TargetNetworkSystem(learning_model, config)

    print("✅ 目标网络系统创建成功")

    # 测试软更新
    for i in range(10):
        target_system.soft_update()

    print("✅ 软更新测试完成")

    # 测试稳定性报告
    stability_report = target_system.get_stability_report()
    print(f"📊 稳定性报告: {stability_report}")

    print("\n🎯 目标网络的核心优势:")
    print("  🌟 稳定指导: 提供稳定的学习目标")
    print("  🔄 软更新: 缓慢吸收新知识")
    print("  📈 持续稳定: 避免训练震荡")


if __name__ == "__main__":
    test_target_network()
