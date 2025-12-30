#!/usr/bin/env python3
"""
赛博裁判长 - SQLite 记忆存储
功能: 使用 SQLite 存储用户历史消息和判决记录
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class UserMessage:
    """用户消息记录"""

    user_id: str
    message: str
    timestamp: datetime
    group_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class JudgmentRecord:
    """判决记录"""

    user_id: str
    case: str  # 案情
    verdict: str  # 判决
    personality: str  # 使用的人格
    timestamp: datetime
    group_id: Optional[str] = None


class MemoryStore:
    """记忆存储"""

    def __init__(self, db_path: str = None):
        """
        初始化存储

        Args:
            db_path: 数据库路径，默认为 data/memory.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "memory.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()

        # 用户消息表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                group_id TEXT,
                metadata TEXT
            )
        """
        )
        # 为常用查询字段创建索引（SQLite 需使用独立的 CREATE INDEX 语句）
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_messages_user_id
            ON user_messages(user_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_messages_timestamp
            ON user_messages(timestamp)
        """
        )

        # 判决记录表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS judgment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                case_text TEXT NOT NULL,
                verdict TEXT NOT NULL,
                personality TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                group_id TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judgment_records_user_id
            ON judgment_records(user_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judgment_records_timestamp
            ON judgment_records(timestamp)
        """
        )

        # 用户统计表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_stats (
                user_id TEXT PRIMARY KEY,
                total_messages INTEGER DEFAULT 0,
                total_judgments INTEGER DEFAULT 0,
                first_seen DATETIME,
                last_seen DATETIME,
                tags TEXT
            )
        """
        )

        self.conn.commit()

    async def save_message(
        self, user_id: str, message: str, group_id: str = None, metadata: Dict = None
    ):
        """保存用户消息"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_messages (user_id, message, timestamp, group_id, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                user_id,
                message,
                datetime.now(),
                group_id,
                json.dumps(metadata) if metadata else None,
            ),
        )

        # 更新用户统计
        self._update_user_stats(user_id)

        self.conn.commit()

    async def save_judgment(
        self,
        user_id: str,
        case: str,
        verdict: str,
        personality: str,
        group_id: str = None,
    ):
        """保存判决记录"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO judgment_records (user_id, case_text, verdict, personality, timestamp, group_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (user_id, case, verdict, personality, datetime.now(), group_id),
        )

        self.conn.commit()

    async def get_user_history(self, user_id: str, limit: int = 50) -> List[str]:
        """获取用户历史消息"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT message FROM user_messages
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (user_id, limit),
        )

        return [row[0] for row in cursor.fetchall()]

    async def get_user_judgments(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户的判决历史"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT case_text, verdict, personality, timestamp
            FROM judgment_records
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (user_id, limit),
        )

        return [
            {
                "case": row[0],
                "verdict": row[1],
                "personality": row[2],
                "timestamp": row[3],
            }
            for row in cursor.fetchall()
        ]

    async def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """获取用户统计信息"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT total_messages, total_judgments, first_seen, last_seen, tags
            FROM user_stats
            WHERE user_id = ?
        """,
            (user_id,),
        )

        row = cursor.fetchone()
        if row:
            return {
                "total_messages": row[0],
                "total_judgments": row[1],
                "first_seen": row[2],
                "last_seen": row[3],
                "tags": json.loads(row[4]) if row[4] else [],
            }
        return None

    def _update_user_stats(self, user_id: str):
        """更新用户统计"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_stats (user_id, total_messages, first_seen, last_seen)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                total_messages = total_messages + 1,
                last_seen = ?
        """,
            (user_id, datetime.now(), datetime.now(), datetime.now()),
        )

    async def search_messages(
        self, query: str, user_id: str = None, limit: int = 20
    ) -> List[Dict]:
        """搜索消息"""
        cursor = self.conn.cursor()

        if user_id:
            cursor.execute(
                """
                SELECT user_id, message, timestamp
                FROM user_messages
                WHERE user_id = ? AND message LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (user_id, f"%{query}%", limit),
            )
        else:
            cursor.execute(
                """
                SELECT user_id, message, timestamp
                FROM user_messages
                WHERE message LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

        return [
            {"user_id": row[0], "message": row[1], "timestamp": row[2]}
            for row in cursor.fetchall()
        ]

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


# 示例使用
async def example_usage():
    """示例用法"""
    store = MemoryStore()

    # 保存消息
    await store.save_message("user_123", "4060 能跑 AI 吗？", group_id="group_456")

    # 保存判决
    await store.save_judgment(
        "user_123",
        "4060 能跑 AI 吗？",
        "鉴定为赛博乞丐。能跑是能跑，但建议自备灭火器。",
        "暴躁老哥",
    )

    # 查询历史
    history = await store.get_user_history("user_123")
    print(f"用户历史: {history}")

    # 查询统计
    stats = await store.get_user_stats("user_123")
    print(f"用户统计: {stats}")

    store.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
