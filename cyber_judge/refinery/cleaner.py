#!/usr/bin/env python3
"""
赛博裁判长 - 数据清洗模块 (The Refinery)
功能: 清洗爬虫抓取的原始数据，去除噪声，格式化输出
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class CleanedJudgment:
    """清洗后的判例数据"""
    instruction: str  # 案情描述
    output: str       # 判决结果
    source: str       # 来源
    quality_score: float  # 质量分数 0-1
    

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        # HTML标签正则
        self.html_pattern = re.compile(r'<[^>]+>')
        # 表情包代码正则
        self.emoji_pattern = re.compile(r'\[.*?\]')
        # 废话过滤（太短的回复）
        self.min_length = 5
        # 无意义回复
        self.useless_replies = {'插眼', '顶', '沙发', '前排', '留名', '路过'}
        
    def clean_text(self, text: str) -> str:
        """清洗单条文本"""
        if not text:
            return ""
        
        # 去除HTML标签
        text = self.html_pattern.sub('', text)
        # 去除表情包代码
        text = self.emoji_pattern.sub('', text)
        # 去除多余空白
        text = ' '.join(text.split())
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def is_valid_verdict(self, verdict: str) -> bool:
        """判断是否是有效的判决"""
        if not verdict or len(verdict) < self.min_length:
            return False
        
        # 过滤无意义回复
        if verdict in self.useless_replies:
            return False
        
        # 必须包含至少一个关键词
        keywords = ['鉴定为', '纯纯的', '有一说一', '属于是', '驳回', '建议', '赛博']
        if not any(kw in verdict for kw in keywords):
            return False
        
        return True
    
    def calculate_quality_score(self, judgment: Dict) -> float:
        """计算判例质量分数"""
        score = 0.0
        
        # 点赞数权重
        upvotes = judgment.get('upvotes', 0)
        score += min(upvotes / 100, 0.4)  # 最多0.4分
        
        # 关键词数量权重
        keywords = judgment.get('keywords', [])
        score += min(len(keywords) * 0.1, 0.3)  # 最多0.3分
        
        # 文本长度权重（适中最好）
        verdict_len = len(judgment.get('verdict', ''))
        if 10 <= verdict_len <= 100:
            score += 0.3
        elif verdict_len > 100:
            score += 0.15
        
        return min(score, 1.0)
    
    def clean_judgment(self, raw_judgment: Dict) -> Optional[CleanedJudgment]:
        """清洗单条判例"""
        case = self.clean_text(raw_judgment.get('case', ''))
        verdict = self.clean_text(raw_judgment.get('verdict', ''))
        
        # 验证有效性
        if not case or not self.is_valid_verdict(verdict):
            return None
        
        quality_score = self.calculate_quality_score(raw_judgment)
        
        # 质量分数太低则丢弃
        if quality_score < 0.3:
            return None
        
        return CleanedJudgment(
            instruction=case,
            output=verdict,
            source=raw_judgment.get('source', ''),
            quality_score=quality_score
        )
    
    def clean_dataset(self, input_file: Path, output_file: Path, 
                     min_quality: float = 0.5) -> List[CleanedJudgment]:
        """清洗整个数据集"""
        print(f"🧹 开始清洗数据: {input_file}")
        
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"📊 原始数据量: {len(raw_data)}")
        
        # 清洗数据
        cleaned_data = []
        for raw_judgment in raw_data:
            cleaned = self.clean_judgment(raw_judgment)
            if cleaned and cleaned.quality_score >= min_quality:
                cleaned_data.append(cleaned)
        
        print(f"✅ 清洗后数据量: {len(cleaned_data)}")
        print(f"📉 过滤率: {(1 - len(cleaned_data)/len(raw_data))*100:.1f}%")
        
        # 保存清洗后的数据
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(j) for j in cleaned_data], f, 
                     ensure_ascii=False, indent=2)
        
        print(f"💾 已保存至: {output_file}")
        
        # 生成统计报告
        self.generate_report(cleaned_data, output_file.parent / 'cleaning_report.txt')
        
        return cleaned_data
    
    def generate_report(self, data: List[CleanedJudgment], report_file: Path):
        """生成清洗报告"""
        df = pd.DataFrame([asdict(j) for j in data])
        
        report = f"""
=== 数据清洗报告 ===

总数据量: {len(data)}

质量分数分布:
{df['quality_score'].describe()}

平均指令长度: {df['instruction'].str.len().mean():.1f}
平均输出长度: {df['output'].str.len().mean():.1f}

质量分数 >= 0.8 的数据: {len(df[df['quality_score'] >= 0.8])}
质量分数 >= 0.6 的数据: {len(df[df['quality_score'] >= 0.6])}
质量分数 >= 0.5 的数据: {len(df[df['quality_score'] >= 0.5])}
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 报告已生成: {report_file}")


def main():
    """主函数"""
    cleaner = DataCleaner()
    
    input_file = Path(__file__).parent.parent / 'data' / 'raw' / 'raw_judgments.json'
    output_file = Path(__file__).parent.parent / 'data' / 'processed' / 'cleaned_judgments.json'
    
    cleaner.clean_dataset(input_file, output_file, min_quality=0.5)


if __name__ == '__main__':
    main()

