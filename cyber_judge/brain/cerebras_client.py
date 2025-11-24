#!/usr/bin/env python3
"""
赛博裁判长 - Cerebras API 客户端
功能: 封装 Cerebras API 调用，提供极速推理能力
"""

import os
import asyncio
from typing import List, Dict, Optional
import aiohttp
from dataclasses import dataclass


@dataclass
class Message:
    """消息对象"""
    role: str  # system, user, assistant
    content: str


class CerebrasClient:
    """Cerebras API 客户端"""
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b"):
        """
        初始化客户端
        
        Args:
            api_key: Cerebras API Key
            model: 模型名称 (llama-3.1-8b / llama-3.3-70b)
        """
        self.api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        self.model = model
        self.base_url = "https://api.cerebras.ai/v1"
        
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY 未设置")
    
    async def chat(self, 
                   messages: List[Message],
                   temperature: float = 0.7,
                   max_tokens: int = 512,
                   stream: bool = False) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成长度
            stream: 是否流式输出
            
        Returns:
            模型回复
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API 请求失败: {response.status} - {error_text}")
                
                result = await response.json()
                return result['choices'][0]['message']['content']
    
    async def chat_with_system_prompt(self,
                                     user_message: str,
                                     system_prompt: str,
                                     few_shot_examples: List[Dict] = None,
                                     **kwargs) -> str:
        """
        使用系统提示词进行对话
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            few_shot_examples: Few-Shot 示例 [{"user": "...", "assistant": "..."}]
            **kwargs: 其他参数传递给 chat()
            
        Returns:
            模型回复
        """
        messages = [Message(role="system", content=system_prompt)]
        
        # 添加 Few-Shot 示例
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append(Message(role="user", content=example["user"]))
                messages.append(Message(role="assistant", content=example["assistant"]))
        
        # 添加用户消息
        messages.append(Message(role="user", content=user_message))
        
        return await self.chat(messages, **kwargs)
    
    async def parallel_chat(self,
                           messages_list: List[List[Message]],
                           **kwargs) -> List[str]:
        """
        并发请求多个对话（用于多重人格）
        
        Args:
            messages_list: 多组消息列表
            **kwargs: 其他参数
            
        Returns:
            多个回复
        """
        tasks = [self.chat(messages, **kwargs) for messages in messages_list]
        return await asyncio.gather(*tasks)


class PersonalityManager:
    """人格管理器 - 实现"并发多重人格"特性"""
    
    def __init__(self, client: CerebrasClient):
        self.client = client
        
        # 定义三种人格
        self.personalities = {
            "暴躁老哥": """你是一个暴躁的老哥，说话直接、不留情面。
你喜欢用"纯纯的"、"属于是"等词汇。
你的判决简短有力，一针见血。""",
            
            "理中客": """你是一个理性中立的客观评论者。
你会从多个角度分析问题，但最后总能给出犀利的结论。
你喜欢用"有一说一"、"客观来讲"等词汇。""",
            
            "阴阳人": """你是一个阴阳怪气的高手。
你表面上很客气，但话里有话，讽刺意味十足。
你喜欢用"建议"、"不妨"等词汇，但实际上是在嘲讽。"""
        }
    
    async def get_best_response(self, user_message: str, 
                               few_shot_examples: List[Dict] = None) -> Dict:
        """
        获取最佳回复（从三个人格中选择）
        
        Args:
            user_message: 用户消息
            few_shot_examples: Few-Shot 示例
            
        Returns:
            {"personality": "人格名", "response": "回复内容", "score": 分数}
        """
        # 并发请求三个人格
        tasks = []
        for name, system_prompt in self.personalities.items():
            task = self.client.chat_with_system_prompt(
                user_message=user_message,
                system_prompt=system_prompt,
                few_shot_examples=few_shot_examples,
                temperature=0.8
            )
            tasks.append((name, task))
        
        # 等待所有响应
        responses = []
        for name, task in tasks:
            response = await task
            responses.append({
                "personality": name,
                "response": response,
                "score": self._score_response(response)
            })
        
        # 选择最佳回复（这里简单选择字数最适中的）
        best = max(responses, key=lambda x: x["score"])
        return best
    
    def _score_response(self, response: str) -> float:
        """
        评分回复质量
        
        评分标准:
        - 长度适中 (20-100字): +0.5
        - 包含关键词: +0.3
        - 有标点符号: +0.2
        """
        score = 0.0
        length = len(response)
        
        # 长度评分
        if 20 <= length <= 100:
            score += 0.5
        elif 10 <= length < 20 or 100 < length <= 150:
            score += 0.3
        
        # 关键词评分
        keywords = ['鉴定为', '纯纯的', '有一说一', '属于是', '建议', '驳回']
        if any(kw in response for kw in keywords):
            score += 0.3
        
        # 标点符号评分
        if any(p in response for p in '。！？，、'):
            score += 0.2
        
        return score


# 示例使用
async def example_usage():
    """示例用法"""
    client = CerebrasClient()
    manager = PersonalityManager(client)
    
    # 单次对话
    response = await client.chat_with_system_prompt(
        user_message="4060 能跑 AI 吗？",
        system_prompt="你是赛博裁判长，说话刻薄但有趣。"
    )
    print(f"单人格回复: {response}")
    
    # 多重人格
    best = await manager.get_best_response("M3 Air 能训练大模型吗？")
    print(f"最佳人格: {best['personality']}")
    print(f"回复: {best['response']}")


if __name__ == '__main__':
    asyncio.run(example_usage())

