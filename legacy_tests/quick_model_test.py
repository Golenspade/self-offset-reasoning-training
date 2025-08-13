"""
快速测试训练好的模型在训练集上的表现
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel


def quick_test_model():
    """快速测试模型表现"""
    print("🧪 快速测试训练好的模型...")
    
    # 初始化tokenizer
    tokenizer = Tokenizer()
    
    # 测试不同的模型
    models_to_test = [
        {
            'name': 'Level 1 鲁棒版',
            'model_path': 'outputs/trained_models/robust_model_Level_1_鲁棒版.npz',
            'data_path': 'data/train_level_1_鲁棒版.json'
        },
        {
            'name': 'Level 3 鲁棒版', 
            'model_path': 'outputs/trained_models/robust_model_Level_3_鲁棒版.npz',
            'data_path': 'data/train_level_3_鲁棒版.json'
        }
    ]
    
    for model_config in models_to_test:
        print(f"\n📊 测试 {model_config['name']}")
        print("-" * 40)
        
        # 加载模型
        model = ImprovedSimpleModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=128,
            max_length=50,
            learning_rate=0.003
        )
        
        if not model.load_model(model_config['model_path']):
            print(f"❌ 无法加载模型: {model_config['model_path']}")
            continue
        
        # 加载数据
        data = load_dataset(model_config['data_path'], tokenizer, 10)  # 只测试10个样本
        
        if not data:
            print(f"❌ 无法加载数据: {model_config['data_path']}")
            continue
        
        print(f"✅ 模型和数据加载成功")
        
        # 测试几个样本
        for i, sample in enumerate(data[:5]):
            try:
                predicted_tokens = model.predict(sample['input'], tokenizer)
                predicted_text = tokenizer.decode(predicted_tokens).strip()
                target_text = sample['target_text'].strip()
                
                print(f"\n  样本 {i+1}:")
                print(f"    输入: {sample['input_text']}")
                print(f"    目标: {target_text}")
                print(f"    预测: {predicted_text}")
                
                # 简单分析
                if predicted_text == target_text:
                    print(f"    结果: ✅ 完全匹配")
                elif '->' in predicted_text and len(predicted_text) > 5:
                    print(f"    结果: 🔄 格式正确但内容不同")
                elif predicted_text.startswith('-> -> ->'):
                    print(f"    结果: 🚨 陷入循环模式")
                else:
                    print(f"    结果: ❌ 格式错误")
                    
            except Exception as e:
                print(f"    结果: ❌ 预测出错: {e}")
        
        print(f"\n📈 {model_config['name']} 测试完成")


if __name__ == "__main__":
    quick_test_model()
