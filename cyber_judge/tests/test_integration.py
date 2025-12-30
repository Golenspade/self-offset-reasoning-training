#!/usr/bin/env python3
"""
赛博裁判长 - 集成测试
"""

import pytest
import asyncio
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMemoryStore:
    """测试记忆存储"""

    def test_save_and_retrieve_message(self):
        """测试保存和检索消息（同步封装 async 接口，避免依赖 pytest-asyncio）"""
        from memory.sqlite_store import MemoryStore

        store = MemoryStore(db_path=":memory:")  # 使用内存数据库

        async def _run():
            # 保存消息
            await store.save_message("user_123", "测试消息", group_id="group_456")

            # 检索消息
            history = await store.get_user_history("user_123")

            assert len(history) == 1
            assert history[0] == "测试消息"

        try:
            asyncio.run(_run())
        finally:
            store.close()

    def test_save_judgment(self):
        """测试保存判决"""
        from memory.sqlite_store import MemoryStore

        store = MemoryStore(db_path=":memory:")

        async def _run():
            await store.save_judgment(
                user_id="user_123",
                case="4060能跑AI吗？",
                verdict="鉴定为赛博乞丐",
                personality="暴躁老哥",
            )

            judgments = await store.get_user_judgments("user_123")

            assert len(judgments) == 1
            assert judgments[0]["verdict"] == "鉴定为赛博乞丐"

        try:
            asyncio.run(_run())
        finally:
            store.close()

    def test_user_stats(self):
        """测试用户统计"""
        from memory.sqlite_store import MemoryStore

        store = MemoryStore(db_path=":memory:")

        async def _run():
            # 保存多条消息
            for i in range(5):
                await store.save_message("user_123", f"消息{i}")

            stats = await store.get_user_stats("user_123")

            assert stats is not None
            assert stats["total_messages"] == 5

        try:
            asyncio.run(_run())
        finally:
            store.close()


class TestDataCleaner:
    """测试数据清洗"""

    def test_clean_text(self):
        """测试文本清洗"""
        from refinery.cleaner import DataCleaner

        cleaner = DataCleaner()

        # 测试去除HTML标签
        text = "<p>测试文本</p>"
        cleaned = cleaner.clean_text(text)
        assert cleaned == "测试文本"

        # 测试去除表情包
        text = "测试[表情]文本"
        cleaned = cleaner.clean_text(text)
        assert cleaned == "测试文本"

    def test_is_valid_verdict(self):
        """测试判决有效性检查"""
        from refinery.cleaner import DataCleaner

        cleaner = DataCleaner()

        # 有效判决
        assert cleaner.is_valid_verdict("鉴定为纯纯的赛博乞丐")
        assert cleaner.is_valid_verdict("有一说一，这个不行")

        # 无效判决
        assert not cleaner.is_valid_verdict("插眼")
        assert not cleaner.is_valid_verdict("顶")
        assert not cleaner.is_valid_verdict("abc")  # 太短且无关键词


class TestPersonalityManager:
    """测试人格管理器"""

    def test_score_response(self):
        """测试回复评分"""
        from brain.cerebras_client import PersonalityManager, CerebrasClient

        # 注意：这里不会实际调用 API
        try:
            client = CerebrasClient()
            manager = PersonalityManager(client)
        except ValueError:
            # 如果没有 API Key，跳过测试
            pytest.skip("CEREBRAS_API_KEY not set")
            return

        # 测试评分
        score1 = manager._score_response("鉴定为纯纯的赛博乞丐。")
        score2 = manager._score_response("好")

        # 长度适中且有关键词的回复应该得分更高
        assert score1 > score2


class TestReActAgent:
    """测试 ReAct 智能体"""

    def test_parse_thought(self):
        """测试思考解析"""
        from brain.react_loop import ReActAgent, ActionType
        from memory.sqlite_store import MemoryStore

        store = MemoryStore(db_path=":memory:")

        # 创建一个模拟的 LLM 客户端
        class MockLLM:
            async def chat_with_system_prompt(self, **kwargs):
                return "Thought: 需要查询历史\nAction: search_history\nAction Input: AI"

        agent = ReActAgent(MockLLM(), store)

        llm_output = "Thought: 需要查询历史\nAction: search_history\nAction Input: AI"
        thought = agent._parse_thought(llm_output)

        assert thought.content == "需要查询历史"
        assert thought.action == ActionType.SEARCH_HISTORY
        assert thought.action_input == "AI"

        store.close()


class TestEndToEnd:
    """端到端测试"""

    def test_full_pipeline(self):
        """测试完整流程（不调用实际 API）"""
        from memory.sqlite_store import MemoryStore

        store = MemoryStore(db_path=":memory:")

        async def _run():
            # 1. 保存用户消息
            await store.save_message("user_123", "4060能跑AI吗？")

            # 2. 模拟判决
            verdict = "鉴定为赛博乞丐。能跑是能跑，但建议自备灭火器。"

            # 3. 保存判决
            await store.save_judgment(
                user_id="user_123",
                case="4060能跑AI吗？",
                verdict=verdict,
                personality="暴躁老哥",
            )

            # 4. 验证数据
            history = await store.get_user_history("user_123")
            judgments = await store.get_user_judgments("user_123")
            stats = await store.get_user_stats("user_123")

            assert len(history) == 1
            assert len(judgments) == 1
            assert stats["total_messages"] == 1

        try:
            asyncio.run(_run())
        finally:
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
