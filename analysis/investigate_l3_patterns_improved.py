"""
文件名: investigate_l3_patterns_improved.py
改进版Level 3模式调查脚本
解决原版本中的"偷懒"问题，提供更高效、健壮的分析
"""

import json
import random
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional


class L3PatternAnalyzer:
    """Level 3 数据模式分析器 - 改进版"""

    def __init__(self, data_file: str = "data/val_L3_complex.json"):
        self.data_file = data_file
        self.samples = []
        self.analysis_results = {}

    def load_data(self) -> bool:
        """加载数据"""
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                self.samples = [json.loads(line) for line in f if line.strip()]
            print(f"✅ 成功加载 {len(self.samples)} 个样本")
            return True
        except FileNotFoundError:
            print(f"❌ 文件未找到: {self.data_file}")
            return False
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False

    def find_common_substrings_efficient(
        self, str1: str, str2: str, min_length: int = 5
    ) -> List[str]:
        """
        高效的共同子串查找算法
        使用集合操作替代暴力循环，时间复杂度从O(N³)降到O(N²)
        """
        if not str1 or not str2:
            return []

        # 生成所有可能的子串（使用集合推导式）
        substrings1 = {
            str1[i : i + j]
            for i in range(len(str1))
            for j in range(min_length, len(str1) - i + 1)
        }
        substrings2 = {
            str2[i : i + j]
            for i in range(len(str2))
            for j in range(min_length, len(str2) - i + 1)
        }

        # 使用集合交集操作，效率远高于嵌套循环
        common = list(substrings1.intersection(substrings2))

        # 按长度降序排序，返回最长的几个
        common.sort(key=len, reverse=True)
        return common[:3]

    def check_simple_transformations_robust(
        self, noisy: str, target: str
    ) -> Dict[str, any]:
        """
        健壮的变换模式检查
        使用正则表达式替代脆弱的字符串分割
        """
        results = {
            "cheating_detected": False,
            "pattern_type": None,
            "confidence": 0.0,
            "details": {},
        }

        # 清理输入字符串
        noisy = noisy.strip()
        target = target.strip()

        # 模式1: 检查 A|B -> ~B -> ~A 的直接映射
        target_match = re.match(r"~\s*(.+?)\s*->\s*~\s*(.+)", target)
        if target_match:
            target_b = target_match.group(1).strip()
            target_a = target_match.group(2).strip()

            # 更健壮的噪声解析，处理括号
            noise_patterns = [
                r"\(\s*(.+?)\s*\)\s*\|\s*(.+)",  # (A) | B
                r"(.+?)\s*\|\s*\(\s*(.+?)\s*\)",  # A | (B)
                r"(.+?)\s*\|\s*(.+)",  # A | B
            ]

            for pattern in noise_patterns:
                noise_match = re.match(pattern, noisy)
                if noise_match:
                    noise_a = noise_match.group(1).strip()
                    noise_b = noise_match.group(2).strip()

                    # 检查直接映射关系
                    if self._normalize_expression(
                        noise_a
                    ) == self._normalize_expression(
                        target_a
                    ) and self._normalize_expression(
                        noise_b
                    ) == self._normalize_expression(
                        target_b
                    ):
                        results.update(
                            {
                                "cheating_detected": True,
                                "pattern_type": "direct_disjunction_mapping",
                                "confidence": 0.95,
                                "details": {
                                    "noise_a": noise_a,
                                    "noise_b": noise_b,
                                    "target_a": target_a,
                                    "target_b": target_b,
                                },
                            }
                        )
                        return results

        # 模式2: 检查变量直接替换模式
        if self._check_variable_substitution(noisy, target):
            results.update(
                {
                    "cheating_detected": True,
                    "pattern_type": "variable_substitution",
                    "confidence": 0.8,
                    "details": self._get_variable_mapping(noisy, target),
                }
            )

        return results

    def _normalize_expression(self, expr: str) -> str:
        """标准化表达式，去除多余的空格和括号"""
        expr = re.sub(r"\s+", "", expr)  # 去除所有空格
        expr = re.sub(r"^\((.+)\)$", r"\1", expr)  # 去除外层括号
        return expr

    def _check_variable_substitution(self, noisy: str, target: str) -> bool:
        """检查是否存在简单的变量替换模式"""
        noisy_vars = set(re.findall(r"[pqrst]", noisy))
        target_vars = set(re.findall(r"[pqrst]", target))

        # 如果变量集合完全相同，可能存在简单映射
        return len(noisy_vars) == len(target_vars) and len(noisy_vars) <= 3

    def _get_variable_mapping(self, noisy: str, target: str) -> Dict[str, str]:
        """获取变量映射关系"""
        noisy_vars = sorted(set(re.findall(r"[pqrst]", noisy)))
        target_vars = sorted(set(re.findall(r"[pqrst]", target)))

        if len(noisy_vars) == len(target_vars):
            return dict(zip(noisy_vars, target_vars))
        return {}

    def analyze_noise_effectiveness_comprehensive(
        self, samples: List[Dict]
    ) -> Dict[str, any]:
        """
        全面的噪声有效性分析
        使用独立的if判断，允许多种噪声类型同时识别
        """
        noise_analysis = {
            "total_samples": len(samples),
            "noise_types": defaultdict(int),
            "multi_noise_samples": 0,
            "noise_combinations": defaultdict(int),
            "effectiveness_score": 0.0,
        }

        for i, sample in enumerate(samples):
            original = sample.get("original_prop", "")
            noisy = sample.get("noisy_prop", "")

            applied_noises = []

            # 独立检查各种噪声类型
            if "->" in original and "|" in noisy and "->" not in noisy:
                noise_analysis["noise_types"]["implication_to_disjunction"] += 1
                applied_noises.append("impl_to_disj")

            if "~~" in noisy:
                noise_analysis["noise_types"]["double_negation"] += 1
                applied_noises.append("double_neg")

            # 修正的括号计数
            original_parens = original.count("(")
            noisy_parens = noisy.count("(")
            double_neg_parens = noisy.count("~~") * 2  # 每个~~通常增加2个括号

            if noisy_parens > original_parens + double_neg_parens:
                noise_analysis["noise_types"]["redundant_parentheses"] += 1
                applied_noises.append("redundant_parens")

            if re.search(r"\(\s*\(\s*[^)]+\s*\)\s*\)", noisy):
                noise_analysis["noise_types"]["nested_parentheses"] += 1
                applied_noises.append("nested_parens")

            if len(re.findall(r"[&|]", noisy)) > len(re.findall(r"[&|]", original)):
                noise_analysis["noise_types"]["extra_operators"] += 1
                applied_noises.append("extra_ops")

            # 记录噪声组合
            if len(applied_noises) > 1:
                noise_analysis["multi_noise_samples"] += 1
                combination = "+".join(sorted(applied_noises))
                noise_analysis["noise_combinations"][combination] += 1
            elif len(applied_noises) == 0:
                if noisy == original:
                    noise_analysis["noise_types"]["no_change"] += 1
                else:
                    noise_analysis["noise_types"]["unknown_change"] += 1

        # 计算噪声有效性分数
        total_noise_applications = sum(noise_analysis["noise_types"].values())
        if total_noise_applications > 0:
            noise_analysis["effectiveness_score"] = (
                total_noise_applications - noise_analysis["noise_types"]["no_change"]
            ) / len(samples)

        return noise_analysis

    def check_variable_patterns_precise(self, samples: List[Dict]) -> Dict[str, any]:
        """
        精确的变量模式检查
        使用更直接的字符匹配，避免单词边界问题
        """
        variable_analysis = {
            "variable_distribution": defaultdict(int),
            "variable_consistency": 0.0,
            "suspicious_patterns": [],
        }

        for sample in samples:
            noisy = sample.get("noisy_prop", "")
            target = sample.get("target_contrapositive", "")

            # 直接字符匹配，更可靠
            noisy_vars = sorted(set(re.findall(r"[pqrst]", noisy)))
            target_vars = sorted(set(re.findall(r"[pqrst]", target)))

            # 记录变量分布
            for var in noisy_vars:
                variable_analysis["variable_distribution"][var] += 1

            # 检查可疑模式
            if len(noisy_vars) == len(target_vars) and len(noisy_vars) <= 2:
                if noisy_vars == target_vars:
                    variable_analysis["suspicious_patterns"].append(
                        {
                            "type": "identical_variables",
                            "variables": noisy_vars,
                            "sample_index": samples.index(sample),
                        }
                    )

        # 计算变量一致性
        total_vars = sum(variable_analysis["variable_distribution"].values())
        if total_vars > 0:
            max_var_count = max(variable_analysis["variable_distribution"].values())
            variable_analysis["variable_consistency"] = max_var_count / total_vars

        return variable_analysis

    def run_comprehensive_analysis(self, sample_size: int = 50) -> Dict[str, any]:
        """运行全面分析"""
        if not self.load_data():
            return {}

        # 随机采样
        analysis_samples = random.sample(
            self.samples, min(sample_size, len(self.samples))
        )

        print(f"\n🔍 开始全面分析 {len(analysis_samples)} 个样本...")

        # 1. 作弊模式检测
        cheating_patterns = []
        for i, sample in enumerate(analysis_samples):
            result = self.check_simple_transformations_robust(
                sample.get("noisy_prop", ""), sample.get("target_contrapositive", "")
            )
            if result["cheating_detected"]:
                result["sample_index"] = i
                cheating_patterns.append(result)

        # 2. 噪声有效性分析
        noise_analysis = self.analyze_noise_effectiveness_comprehensive(
            analysis_samples
        )

        # 3. 变量模式分析
        variable_analysis = self.check_variable_patterns_precise(analysis_samples)

        # 4. 共同子串分析（改进版）
        substring_analysis = self._analyze_common_substrings(analysis_samples[:10])

        # 汇总结果
        self.analysis_results = {
            "sample_size": len(analysis_samples),
            "cheating_patterns": cheating_patterns,
            "noise_analysis": noise_analysis,
            "variable_analysis": variable_analysis,
            "substring_analysis": substring_analysis,
            "overall_risk_score": self._calculate_risk_score(
                cheating_patterns, noise_analysis
            ),
        }

        return self.analysis_results

    def _analyze_common_substrings(self, samples: List[Dict]) -> Dict[str, any]:
        """分析共同子串（使用改进的算法）"""
        substring_results = []

        for sample in samples:
            noisy = sample.get("noisy_prop", "")
            target = sample.get("target_contrapositive", "")

            common = self.find_common_substrings_efficient(noisy, target, min_length=3)
            if common:
                substring_results.append(
                    {"noisy": noisy, "target": target, "common_substrings": common}
                )

        return {
            "samples_with_common_substrings": len(substring_results),
            "examples": substring_results[:3],
        }

    def _calculate_risk_score(
        self, cheating_patterns: List[Dict], noise_analysis: Dict
    ) -> float:
        """计算整体风险分数"""
        risk_score = 0.0

        # 作弊模式风险
        if cheating_patterns:
            high_confidence_patterns = [
                p for p in cheating_patterns if p["confidence"] > 0.9
            ]
            risk_score += len(high_confidence_patterns) / len(cheating_patterns) * 0.5

        # 噪声有效性风险
        if noise_analysis["effectiveness_score"] < 0.5:
            risk_score += 0.3

        # 多噪声样本比例
        if noise_analysis["total_samples"] > 0:
            multi_noise_ratio = (
                noise_analysis["multi_noise_samples"] / noise_analysis["total_samples"]
            )
            if multi_noise_ratio < 0.3:
                risk_score += 0.2

        return min(risk_score, 1.0)

    def generate_detailed_report(self) -> str:
        """生成详细的分析报告"""
        if not self.analysis_results:
            return "❌ 请先运行分析"

        results = self.analysis_results

        report = f"""
🔍 Level 3 数据模式分析报告（改进版）
{'='*60}

📊 基本统计:
  分析样本数: {results['sample_size']}
  整体风险分数: {results['overall_risk_score']:.2f} (0-1, 越高越危险)

🚨 作弊模式检测:
  发现可疑模式: {len(results['cheating_patterns'])} 个
"""

        if results["cheating_patterns"]:
            report += "  详细信息:\n"
            for pattern in results["cheating_patterns"][:3]:
                report += f"    - 类型: {pattern['pattern_type']}\n"
                report += f"      置信度: {pattern['confidence']:.2f}\n"
                report += f"      详情: {pattern['details']}\n"

        noise = results["noise_analysis"]
        report += f"""
🎭 噪声有效性分析:
  噪声有效性分数: {noise['effectiveness_score']:.2f}
  多重噪声样本: {noise['multi_noise_samples']} ({noise['multi_noise_samples']/noise['total_samples']*100:.1f}%)
  
  噪声类型分布:
"""
        for noise_type, count in noise["noise_types"].items():
            percentage = count / noise["total_samples"] * 100
            report += f"    {noise_type}: {count} ({percentage:.1f}%)\n"

        var = results["variable_analysis"]
        report += f"""
🔤 变量模式分析:
  变量一致性: {var['variable_consistency']:.2f}
  可疑模式数: {len(var['suspicious_patterns'])}
  
  变量分布: {dict(var['variable_distribution'])}
"""

        substr = results["substring_analysis"]
        report += f"""
🧩 共同子串分析:
  有共同子串的样本: {substr['samples_with_common_substrings']}
  
💡 改进建议:
"""

        if results["overall_risk_score"] > 0.6:
            report += "  ⚠️  高风险：数据集存在明显的作弊捷径，建议重新生成\n"
        elif results["overall_risk_score"] > 0.3:
            report += "  ⚠️  中等风险：存在一些可疑模式，建议增加噪声复杂度\n"
        else:
            report += "  ✅ 低风险：数据集质量良好，作弊风险较低\n"

        return report


def main():
    """主函数"""
    print("🔍 Level 3 模式分析器（改进版）")
    print("解决原版本中的效率和健壮性问题")
    print("=" * 60)

    analyzer = L3PatternAnalyzer()
    results = analyzer.run_comprehensive_analysis(sample_size=50)

    if results:
        print(analyzer.generate_detailed_report())

        # 保存结果
        with open(
            "outputs/l3_pattern_analysis_improved.json", "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n✅ 分析结果已保存到 outputs/l3_pattern_analysis_improved.json")
    else:
        print("❌ 分析失败")


if __name__ == "__main__":
    main()
