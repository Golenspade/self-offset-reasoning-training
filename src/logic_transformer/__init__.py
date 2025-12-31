"""
Logic Transformer Package
自偏移推理训练的核心包
"""

__version__ = "1.0.0"
__author__ = "Logic Transformer Team"
__description__ = "A package for self-offset reasoning training in symbolic logic"

# 导出主要类和函数
from .data_utils import Tokenizer
from .logic_rules import rule_based_predict_corrected

# HybridModel 仅在部分场景需要，且依赖 models 子包；
# 为了兼容某些裁剪后的环境，这里做容错导入。
try:  # pragma: no cover - 在缺失 models 的环境下兜底
    from .models.hybrid_model import HybridModel
except Exception:
    HybridModel = None  # 类型: ignore

__all__ = ["Tokenizer", "rule_based_predict_corrected", "HybridModel"]
