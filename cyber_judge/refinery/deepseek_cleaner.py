#!/usr/bin/env python3
"""
使用 DeepSeek API 清洗数据
目标：提取群体潜意识，删除技术噪音
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from typing import List, Dict
import time

# DeepSeek API 配置
API_KEY = "sk-d7061f4e11fa4a60905f9a9791cf83bc"
BASE_URL = "https://api.deepseek.com"

# 清洗 Prompt
CLEANING_PROMPT = """你是一个数据清洗助手。我们正在提取网络论坛的"群体潜意识"——即一群人真实的说话方式和表达范式。

请判断以下内容是否应该保留：

【保留 KEEP】真实的人类表达：
- 评论、吐槽、判断、讨论
- 叙事、故事、创作
- 提问、回答、建议
- 任何有语义内容的文字（无论长短）

【删除 DELETE】技术噪音：
- 图片占位符（如"点击展开，查看完整图片"）
- 楼层占位（如"2l自留"、"占楼"）
- 纯符号/空内容（如只有"..."、"。。。"且无其他内容）
- 纯引用标记（如只有"@用户名"且无其他内容）

内容："{content}"

请只回答：KEEP 或 DELETE"""


def init_client():
    """初始化 DeepSeek 客户端"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def judge_content(client: OpenAI, content: str) -> str:
    """
    使用 DeepSeek 判断内容是否保留
    
    Returns:
        "KEEP" 或 "DELETE"
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个数据清洗助手，只回答 KEEP 或 DELETE。"},
                {"role": "user", "content": CLEANING_PROMPT.format(content=content[:500])}  # 限制长度
            ],
            temperature=0.1,  # 低温度，更确定性
            max_tokens=10,
            stream=False
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # 确保返回值有效
        if "KEEP" in result:
            return "KEEP"
        elif "DELETE" in result:
            return "DELETE"
        else:
            print(f"⚠️  未知响应: {result}, 默认保留")
            return "KEEP"
            
    except Exception as e:
        print(f"⚠️  API 调用失败: {e}, 默认保留")
        return "KEEP"


def clean_judgments(input_file: str, output_file: str, report_file: str):
    """
    清洗判断数据
    """
    print("🧹 开始数据清洗...")
    print(f"📂 输入: {input_file}")
    print(f"📂 输出: {output_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 原始数据: {len(data)} 个帖子")
    
    # 初始化客户端
    client = init_client()
    
    # 统计
    stats = {
        'total_threads': len(data),
        'total_replies_before': 0,
        'total_replies_after': 0,
        'deleted_replies': 0,
        'deleted_examples': []
    }
    
    cleaned_data = []
    
    # 遍历每个帖子
    for i, thread in enumerate(data):
        print(f"\n📄 [{i+1}/{len(data)}] {thread['title'][:50]}...")
        
        # 帖子内容通常保留（除非是纯占位）
        thread_content = thread.get('content', '').strip()
        
        cleaned_thread = thread.copy()
        cleaned_replies = []
        
        original_reply_count = len(thread.get('replies', []))
        stats['total_replies_before'] += original_reply_count
        
        # 清洗回复
        for j, reply in enumerate(thread.get('replies', [])):
            content = reply.get('content', '').strip()
            
            if not content:
                # 空内容直接删除
                stats['deleted_replies'] += 1
                continue
            
            # 调用 API 判断
            print(f"  [{j+1}/{original_reply_count}] 判断中...", end=' ')
            decision = judge_content(client, content)
            
            if decision == "KEEP":
                print("✅ KEEP")
                cleaned_replies.append(reply)
                stats['total_replies_after'] += 1
            else:
                print(f"❌ DELETE: {content[:50]}...")
                stats['deleted_replies'] += 1
                if len(stats['deleted_examples']) < 20:
                    stats['deleted_examples'].append(content[:100])
            
            # 避免 API 限流
            time.sleep(0.1)
        
        cleaned_thread['replies'] = cleaned_replies
        cleaned_data.append(cleaned_thread)
        
        print(f"  ✅ 保留 {len(cleaned_replies)}/{original_reply_count} 条回复")
    
    # 保存清洗后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    report = f"""
{'='*60}
数据清洗报告
{'='*60}

📊 统计信息:
  - 帖子数: {stats['total_threads']}
  - 原始回复数: {stats['total_replies_before']}
  - 清洗后回复数: {stats['total_replies_after']}
  - 删除回复数: {stats['deleted_replies']}
  - 保留率: {stats['total_replies_after']/stats['total_replies_before']*100:.1f}%

🗑️ 删除示例（前20条）:
"""
    for i, example in enumerate(stats['deleted_examples'], 1):
        report += f"  {i}. {example}...\n"
    
    report += f"\n{'='*60}\n"
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"✅ 清洗完成！")
    print(f"📂 输出文件: {output_file}")
    print(f"📂 报告文件: {report_file}")


if __name__ == "__main__":
    # 文件路径（数据在项目根目录的 data 文件夹）
    base_dir = Path(__file__).parent.parent.parent  # 回到项目根目录
    input_file = base_dir / "data" / "raw" / "bandori_judgments.json"
    output_file = base_dir / "data" / "processed" / "cleaned_judgments.json"
    report_file = base_dir / "data" / "processed" / "cleaning_report.txt"

    clean_judgments(str(input_file), str(output_file), str(report_file))

