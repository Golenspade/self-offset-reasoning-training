#!/usr/bin/env python3
"""
赛博裁判长 - ReAct 循环实现
功能: 实现 Reasoning + Acting 循环，让 Bot 具备思考和行动能力
"""

import re
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """行动类型"""
    DIRECT_REPLY = "direct_reply"  # 直接回复
    SEARCH_HISTORY = "search_history"  # 查询历史
    SEARCH_WEB = "search_web"  # 搜索网络
    ANALYZE_PATTERN = "analyze_pattern"  # 分析模式


@dataclass
class Thought:
    """思考过程"""
    content: str  # 思考内容
    action: ActionType  # 决定的行动
    action_input: str  # 行动输入


@dataclass
class Observation:
    """观察结果"""
    content: str  # 观察到的内容
    source: str  # 来源


class ReActAgent:
    """ReAct 智能体"""
    
    def __init__(self, llm_client, memory_store, max_iterations: int = 3):
        """
        初始化 ReAct 智能体
        
        Args:
            llm_client: LLM 客户端（Cerebras）
            memory_store: 记忆存储
            max_iterations: 最大迭代次数
        """
        self.llm = llm_client
        self.memory = memory_store
        self.max_iterations = max_iterations
        
        # 注册可用的工具
        self.tools: Dict[str, Callable] = {
            "search_history": self._search_history,
            "search_web": self._search_web,
            "analyze_pattern": self._analyze_pattern,
        }
    
    def _parse_thought(self, llm_output: str) -> Thought:
        """解析 LLM 输出的思考过程"""
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', llm_output, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # 提取 Action
        action_match = re.search(r'Action:\s*(\w+)', llm_output)
        action_str = action_match.group(1) if action_match else "direct_reply"
        
        # 提取 Action Input
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', llm_output)
        action_input = input_match.group(1).strip() if input_match else ""
        
        try:
            action = ActionType(action_str.lower())
        except ValueError:
            action = ActionType.DIRECT_REPLY
        
        return Thought(
            content=thought,
            action=action,
            action_input=action_input
        )
    
    async def _search_history(self, user_id: str, query: str) -> Observation:
        """搜索用户历史"""
        history = await self.memory.get_user_history(user_id, limit=10)
        
        # 简单的关键词匹配
        relevant = [h for h in history if query.lower() in h.lower()]
        
        if relevant:
            content = f"找到 {len(relevant)} 条相关历史:\n" + "\n".join(relevant[:3])
        else:
            content = "未找到相关历史记录"
        
        return Observation(content=content, source="memory")
    
    async def _search_web(self, query: str) -> Observation:
        """搜索网络（占位符）"""
        # TODO: 实现实际的网络搜索
        return Observation(
            content=f"网络搜索结果: {query}（功能开发中）",
            source="web"
        )
    
    async def _analyze_pattern(self, user_id: str) -> Observation:
        """分析用户发言模式"""
        history = await self.memory.get_user_history(user_id, limit=50)
        
        # 简单的模式分析
        total = len(history)
        if total == 0:
            return Observation(content="该用户无历史记录", source="analysis")
        
        # 统计关键词
        keywords = {}
        for msg in history:
            for word in ['AI', '模型', '训练', '显卡', 'GPU']:
                if word in msg:
                    keywords[word] = keywords.get(word, 0) + 1
        
        if keywords:
            top_keyword = max(keywords.items(), key=lambda x: x[1])
            content = f"该用户共 {total} 条发言，最常提及: {top_keyword[0]} ({top_keyword[1]}次)"
        else:
            content = f"该用户共 {total} 条发言，暂无明显特征"
        
        return Observation(content=content, source="analysis")
    
    async def think_and_act(self, user_message: str, user_id: str, 
                           context: str = "") -> str:
        """
        思考并行动
        
        Args:
            user_message: 用户消息
            user_id: 用户ID
            context: 上下文信息
            
        Returns:
            最终回复
        """
        conversation_history = []
        
        for iteration in range(self.max_iterations):
            # 构建提示词
            prompt = self._build_react_prompt(
                user_message, 
                user_id,
                context,
                conversation_history
            )
            
            # 调用 LLM 思考
            llm_output = await self.llm.chat_with_system_prompt(
                user_message=prompt,
                system_prompt=self._get_react_system_prompt()
            )
            
            # 解析思考结果
            thought = self._parse_thought(llm_output)
            conversation_history.append(f"Thought: {thought.content}")
            
            # 如果决定直接回复，则结束循环
            if thought.action == ActionType.DIRECT_REPLY:
                # 提取最终回复
                final_match = re.search(r'Final Answer:\s*(.+)', llm_output, re.DOTALL)
                if final_match:
                    return final_match.group(1).strip()
                else:
                    return thought.content
            
            # 执行行动
            observation = await self._execute_action(
                thought.action, 
                user_id, 
                thought.action_input
            )
            conversation_history.append(f"Observation: {observation.content}")
        
        # 达到最大迭代次数，强制生成回复
        final_prompt = f"""基于以上思考和观察，请给出最终判决。

用户消息: {user_message}
思考历史: {chr(10).join(conversation_history)}

请直接给出判决，不要再思考。"""
        
        return await self.llm.chat_with_system_prompt(
            user_message=final_prompt,
            system_prompt="你是赛博裁判长，现在请给出最终判决。"
        )
    
    async def _execute_action(self, action: ActionType, user_id: str, 
                             action_input: str) -> Observation:
        """执行行动"""
        if action == ActionType.SEARCH_HISTORY:
            return await self._search_history(user_id, action_input)
        elif action == ActionType.SEARCH_WEB:
            return await self._search_web(action_input)
        elif action == ActionType.ANALYZE_PATTERN:
            return await self._analyze_pattern(user_id)
        else:
            return Observation(content="未知行动", source="error")
    
    def _build_react_prompt(self, user_message: str, user_id: str,
                           context: str, history: List[str]) -> str:
        """构建 ReAct 提示词"""
        prompt = f"""用户消息: {user_message}
用户ID: {user_id}
上下文: {context}

"""
        if history:
            prompt += "之前的思考:\n" + "\n".join(history) + "\n\n"
        
        prompt += """请思考如何回应这个用户。你可以:
1. 直接回复 (direct_reply)
2. 查询用户历史 (search_history)
3. 搜索网络 (search_web)
4. 分析用户模式 (analyze_pattern)

请按以下格式输出:
Thought: [你的思考过程]
Action: [选择的行动]
Action Input: [行动的输入参数]

如果决定直接回复，请输出:
Thought: [思考过程]
Action: direct_reply
Final Answer: [你的判决]
"""
        return prompt
    
    def _get_react_system_prompt(self) -> str:
        """获取 ReAct 系统提示词"""
        return """你是赛博裁判长，一个刻薄但有趣的 AI 法官。

你的能力:
1. 查成分: 分析用户历史发言
2. 下判决: 对用户的言论进行定性
3. 引经据典: 引用历史打脸

你说话的特点:
- 刻薄但不失幽默
- 逻辑自洽
- 喜欢用"鉴定为"、"纯纯的"、"驳回上诉"等词汇

请认真思考后再回复。"""


# 示例使用
async def example_usage():
    """示例用法"""
    from cyber_judge.brain.cerebras_client import CerebrasClient
    from cyber_judge.memory.sqlite_store import MemoryStore
    
    llm = CerebrasClient()
    memory = MemoryStore()
    agent = ReActAgent(llm, memory)
    
    response = await agent.think_and_act(
        user_message="4060 能跑 AI 吗？",
        user_id="user_123",
        context="群聊讨论 AI 训练"
    )
    
    print(f"最终回复: {response}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(example_usage())

