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
from .models.hybrid_model import HybridModel

__all__ = [
    "Tokenizer",
    "rule_based_predict_corrected", 
    "HybridModel"
]
