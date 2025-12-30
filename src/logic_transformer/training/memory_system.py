"""
文件名: src/logic_transformer/training/memory_system.py
第二阶段：累积学习 - 经验回放缓冲区
实现记忆宫殿，解决灾难性遗忘问题
"""

import random
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """经验回放缓冲区 - 模型的记忆宫殿"""

    def __init__(self, capacity: int, save_path: Optional[str] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.save_path = save_path

        # 统计信息
        self.total_added = 0
        self.total_sampled = 0

        logger.info(f"🏛️ 记忆宫殿初始化完成，容量: {capacity}")

    def push(self, experience: Dict):
        """将单条经验存入缓冲区"""
        # 为经验添加时间戳和ID
        enhanced_experience = {
            **experience,
            "timestamp": self.total_added,
            "experience_id": f"exp_{self.total_added}",
        }

        self.buffer.append(enhanced_experience)
        self.total_added += 1

        if self.total_added % 1000 == 0:
            logger.debug(f"记忆宫殿已存储 {self.total_added} 条经验")

    def push_batch(self, experiences: List[Dict]):
        """批量存入经验"""
        for exp in experiences:
            self.push(exp)

    def sample(self, batch_size: int, strategy: str = "random") -> List[Dict]:
        """从缓冲区中抽样经验"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        if strategy == "random":
            sampled = random.sample(self.buffer, batch_size)
        elif strategy == "recent_bias":
            # 偏向于采样更新的经验
            sampled = self._sample_with_recent_bias(batch_size)
        elif strategy == "diverse":
            # 尽量采样多样化的经验
            sampled = self._sample_diverse(batch_size)
        else:
            sampled = random.sample(self.buffer, batch_size)

        self.total_sampled += len(sampled)
        return sampled

    def _sample_with_recent_bias(self, batch_size: int) -> List[Dict]:
        """带有新近偏向的采样"""
        buffer_list = list(self.buffer)

        # 为每个经验分配权重，越新的权重越高
        weights = []
        for i, exp in enumerate(buffer_list):
            # 使用指数衰减权重
            age = len(buffer_list) - i
            weight = np.exp(-age / (len(buffer_list) * 0.3))
            weights.append(weight)

        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()

        # 根据权重采样
        indices = np.random.choice(
            len(buffer_list), size=batch_size, replace=False, p=weights
        )

        return [buffer_list[i] for i in indices]

    def _sample_diverse(self, batch_size: int) -> List[Dict]:
        """多样化采样 - 尽量选择不同类型的经验"""
        buffer_list = list(self.buffer)

        # 简单的多样化策略：按时间戳分段采样
        segments = 5
        segment_size = len(buffer_list) // segments
        samples_per_segment = batch_size // segments

        sampled = []
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(buffer_list))
            segment_data = buffer_list[start_idx:end_idx]

            if segment_data:
                segment_samples = random.sample(
                    segment_data, min(samples_per_segment, len(segment_data))
                )
                sampled.extend(segment_samples)

        # 如果还需要更多样本，随机补充
        remaining = batch_size - len(sampled)
        if remaining > 0:
            remaining_pool = [exp for exp in buffer_list if exp not in sampled]
            if remaining_pool:
                additional = random.sample(
                    remaining_pool, min(remaining, len(remaining_pool))
                )
                sampled.extend(additional)

        return sampled[:batch_size]

    def get_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        if not self.buffer:
            return {"size": 0, "capacity": self.capacity}

        # 分析经验的多样性
        complexity_levels = {}
        for exp in self.buffer:
            level = exp.get("complexity_level", "unknown")
            complexity_levels[level] = complexity_levels.get(level, 0) + 1

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "complexity_distribution": complexity_levels,
            "oldest_timestamp": (
                min(exp["timestamp"] for exp in self.buffer) if self.buffer else None
            ),
            "newest_timestamp": (
                max(exp["timestamp"] for exp in self.buffer) if self.buffer else None
            ),
        }

    def save_to_disk(self, filepath: Optional[str] = None):
        """保存缓冲区到磁盘"""
        save_path = filepath or self.save_path
        if not save_path:
            logger.warning("没有指定保存路径，跳过保存")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = {
            "buffer": list(self.buffer),
            "capacity": self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"记忆宫殿已保存到: {save_path}")

    def load_from_disk(self, filepath: Optional[str] = None):
        """从磁盘加载缓冲区"""
        load_path = filepath or self.save_path
        if not load_path or not os.path.exists(load_path):
            logger.warning(f"文件不存在，无法加载: {load_path}")
            return

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.buffer = deque(data["buffer"], maxlen=self.capacity)
        self.total_added = data["total_added"]
        self.total_sampled = data["total_sampled"]

        logger.info(f"记忆宫殿已从磁盘加载: {load_path}")
        logger.info(f"加载了 {len(self.buffer)} 条经验")

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.total_added = 0
        self.total_sampled = 0
        logger.info("记忆宫殿已清空")

    def __len__(self):
        return len(self.buffer)


class CumulativeLearningSystem:
    """累积学习系统 - 整合记忆宫殿的训练系统"""

    def __init__(self, replay_buffer: ReplayBuffer, config: Dict):
        self.replay_buffer = replay_buffer
        self.config = config

        # 学习参数
        self.new_data_ratio = config.get(
            "new_data_ratio", 0.3
        )  # 新数据在训练批次中的比例
        self.training_iterations_per_loop = config.get(
            "training_iterations_per_loop", 5
        )
        self.min_buffer_size = config.get("min_buffer_size", 100)

        logger.info("🧠 累积学习系统初始化完成")
        logger.info(f"  新数据比例: {self.new_data_ratio}")
        logger.info(f"  每轮训练迭代: {self.training_iterations_per_loop}")

    def prepare_training_batch(
        self, new_samples: List[Dict], batch_size: int
    ) -> List[Dict]:
        """准备训练批次 - 新旧数据混合"""
        if len(self.replay_buffer) < self.min_buffer_size:
            # 缓冲区数据不足，主要使用新数据
            logger.debug("缓冲区数据不足，主要使用新数据")
            return new_samples[:batch_size]

        # 计算新旧数据的数量
        new_data_count = int(batch_size * self.new_data_ratio)
        old_data_count = batch_size - new_data_count

        # 获取新数据
        new_batch = new_samples[:new_data_count] if new_samples else []

        # 从记忆宫殿采样旧数据
        old_batch = self.replay_buffer.sample(old_data_count, strategy="diverse")

        # 混合并打乱
        mixed_batch = new_batch + old_batch
        random.shuffle(mixed_batch)

        logger.debug(f"训练批次组成: {len(new_batch)} 新数据 + {len(old_batch)} 旧数据")

        return mixed_batch

    def update_memory(self, new_experiences: List[Dict]):
        """更新记忆宫殿"""
        self.replay_buffer.push_batch(new_experiences)

        # 定期保存
        if self.replay_buffer.total_added % 1000 == 0:
            self.replay_buffer.save_to_disk()


def test_memory_system():
    """测试记忆系统"""
    print("🧪 测试累积学习记忆系统")
    print("=" * 50)

    # 创建记忆宫殿
    buffer = ReplayBuffer(capacity=1000, save_path="test_memory.pkl")

    # 添加一些测试经验
    test_experiences = [
        {
            "input_text": f"test_{i}",
            "target_text": f"target_{i}",
            "complexity_level": "simple",
        }
        for i in range(50)
    ]

    buffer.push_batch(test_experiences)
    print(f"✅ 添加了 {len(test_experiences)} 条经验")

    # 测试采样
    samples = buffer.sample(10, strategy="random")
    print(f"✅ 随机采样了 {len(samples)} 条经验")

    # 测试统计信息
    stats = buffer.get_stats()
    print(f"📊 缓冲区统计: {stats}")

    # 测试累积学习系统
    config = {
        "new_data_ratio": 0.3,
        "training_iterations_per_loop": 5,
        "min_buffer_size": 10,
    }

    learning_system = CumulativeLearningSystem(buffer, config)

    # 测试批次准备
    new_samples = [{"input_text": "new_1", "target_text": "new_target_1"}]
    batch = learning_system.prepare_training_batch(new_samples, batch_size=20)
    print(f"✅ 准备了大小为 {len(batch)} 的训练批次")

    print("\n🎯 累积学习的核心优势:")
    print("  🏛️ 记忆宫殿: 防止灾难性遗忘")
    print("  🔄 新旧混合: 温故而知新")
    print("  📈 持续积累: 知识不断增长")


if __name__ == "__main__":
    test_memory_system()
