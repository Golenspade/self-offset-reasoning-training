#!/usr/bin/env python3
"""
赛博裁判长 - LLM 数据蒸馏模块
功能: 使用 LLM 对清洗后的数据进行进一步提炼和格式化
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import os


@dataclass
class DistilledExample:
    """蒸馏后的示例"""

    instruction: str
    output: str
    reasoning: str  # LLM 提取的推理过程
    style_tags: List[str]  # 风格标签


class LLMDistiller:
    """LLM 蒸馏器"""

    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b"):
        """
        初始化蒸馏器

        Args:
            api_key: Cerebras API Key (如果为空则从环境变量读取)
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self.model = model
        self.base_url = "https://api.cerebras.ai/v1"

        if not self.api_key:
            print("⚠️  警告: 未设置 CEREBRAS_API_KEY，蒸馏功能将不可用")

    def create_distill_prompt(self, case: str, verdict: str) -> str:
        """创建蒸馏提示词"""
        return f"""你是一个专业的数据标注专家。请分析以下对话，提取其中的精华部分。

原始案情: {case}
原始判决: {verdict}

请完成以下任务:
1. 去除脏话和不文明用语，但保留幽默感和讽刺意味
2. 将判决改写成"法官判词"的格式，使其更加正式但不失趣味
3. 提取判决的推理逻辑
4. 标注判决的风格特征（如：讽刺、直接、阴阳怪气等）

请以 JSON 格式返回:
{{
  "instruction": "改写后的案情",
  "output": "改写后的判决",
  "reasoning": "推理过程",
  "style_tags": ["风格标签1", "风格标签2"]
}}
"""

    async def distill_single(self, case: str, verdict: str) -> DistilledExample:
        """蒸馏单条数据"""
        # TODO: 实现实际的 API 调用
        # 这里是示例实现

        # 如果没有 API Key，返回简单处理的结果
        if not self.api_key:
            return DistilledExample(
                instruction=case,
                output=verdict,
                reasoning="未进行 LLM 蒸馏",
                style_tags=["原始"],
            )

        # 实际实现应该调用 Cerebras API
        # response = await self.call_cerebras_api(prompt)
        # return parse_response(response)

        return DistilledExample(
            instruction=case,
            output=verdict,
            reasoning="示例推理过程",
            style_tags=["讽刺", "直接"],
        )

    async def distill_batch(
        self, data: List[Dict], max_concurrent: int = 5
    ) -> List[DistilledExample]:
        """批量蒸馏数据"""
        print(f"🧪 开始蒸馏数据，共 {len(data)} 条")

        # 创建任务队列
        semaphore = asyncio.Semaphore(max_concurrent)

        async def distill_with_limit(item):
            async with semaphore:
                return await self.distill_single(item["instruction"], item["output"])

        # 并发执行
        tasks = [distill_with_limit(item) for item in data]
        results = await asyncio.gather(*tasks)

        print(f"✅ 蒸馏完成")
        return results

    def save_for_production(self, examples: List[DistilledExample], output_file: Path):
        """保存为生产环境的 Few-Shot 示例"""
        # 选择高质量示例（这里简单选择前10条）
        top_examples = examples[:10]

        # 格式化为 Few-Shot 格式
        few_shot_text = "# 赛博裁判长 - 判例参考\n\n"
        for i, ex in enumerate(top_examples, 1):
            few_shot_text += f"## 判例 {i}\n"
            few_shot_text += f"用户: {ex.instruction}\n"
            few_shot_text += f"裁判: {ex.output}\n\n"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(few_shot_text)

        print(f"💾 Few-Shot 示例已保存: {output_file}")

    def save_for_training(self, examples: List[DistilledExample], output_file: Path):
        """保存为训练格式 (JSONL)"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for ex in examples:
                # 转换为训练格式
                train_item = {
                    "messages": [
                        {"role": "user", "content": ex.instruction},
                        {"role": "assistant", "content": ex.output},
                    ],
                    "metadata": {
                        "reasoning": ex.reasoning,
                        "style_tags": ex.style_tags,
                    },
                }
                f.write(json.dumps(train_item, ensure_ascii=False) + "\n")

        print(f"💾 训练数据已保存: {output_file}")


async def main():
    """主函数"""
    distiller = LLMDistiller()

    # 读取清洗后的数据
    input_file = (
        Path(__file__).parent.parent / "data" / "processed" / "cleaned_judgments.json"
    )

    if not input_file.exists():
        print(f"❌ 找不到输入文件: {input_file}")
        print("请先运行 cleaner.py 清洗数据")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 蒸馏数据
    distilled = await distiller.distill_batch(data)

    # 保存为不同格式
    base_path = Path(__file__).parent.parent / "data" / "examples"
    distiller.save_for_production(distilled, base_path / "production_examples.txt")
    distiller.save_for_training(distilled, base_path / "train.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
