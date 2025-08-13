"""
深度调试序列生成机制
分析为什么模型陷入 "-> -> -> ..." 循环
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from logic_transformer.data_utils import Tokenizer, load_dataset
from logic_transformer.models.base_model import ImprovedSimpleModel


def debug_tokenizer():
    """调试tokenizer的token映射"""
    print("🔍 调试Tokenizer...")
    print("=" * 50)
    
    tokenizer = Tokenizer()
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"符号列表: {tokenizer.symbols}")
    print(f"PAD_TOKEN: {tokenizer.PAD_TOKEN}")
    print(f"START_TOKEN: {tokenizer.START_TOKEN}")
    print(f"END_TOKEN: {tokenizer.END_TOKEN}")
    
    print(f"\n字符到整数映射:")
    for char, idx in tokenizer.char_to_int.items():
        print(f"  '{char}' -> {idx}")
    
    print(f"\n整数到字符映射:")
    for idx, char in tokenizer.int_to_char.items():
        print(f"  {idx} -> '{char}'")
    
    # 测试编码解码
    test_text = "~p -> q"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n编码解码测试:")
    print(f"  原文: '{test_text}'")
    print(f"  编码: {encoded}")
    print(f"  解码: '{decoded}'")
    
    return tokenizer


def debug_model_prediction_step_by_step(model, tokenizer, input_text):
    """逐步调试模型预测过程"""
    print(f"\n🔍 逐步调试预测过程...")
    print(f"输入: '{input_text}'")
    print("=" * 60)
    
    # 编码输入
    input_sequence = tokenizer.encode(input_text)
    print(f"1. 输入编码: {input_sequence}")
    print(f"   对应字符: {[tokenizer.int_to_char[token] for token in input_sequence]}")
    
    # 编码
    encoded = model.encode(input_sequence)
    print(f"2. 编码后的隐藏状态形状: {encoded.shape}")
    print(f"   编码值范围: [{encoded.min():.3f}, {encoded.max():.3f}]")
    
    # 开始解码
    output_sequence = []
    current_token = tokenizer.START_TOKEN
    print(f"3. 开始解码，初始token: {current_token} ('{tokenizer.int_to_char[current_token]}')")
    
    for step in range(10):  # 只调试前10步
        print(f"\n  步骤 {step + 1}:")
        print(f"    当前token: {current_token} ('{tokenizer.int_to_char.get(current_token, 'UNKNOWN')}')")
        
        # 解码步骤
        new_hidden, output_probs = model.decode_step(encoded, current_token)
        
        print(f"    新隐藏状态范围: [{new_hidden.min():.3f}, {new_hidden.max():.3f}]")
        print(f"    输出概率形状: {output_probs.shape}")
        print(f"    概率和: {output_probs.sum():.6f}")
        
        # 分析概率分布
        top_5_indices = np.argsort(output_probs)[-5:][::-1]
        print(f"    前5个最高概率:")
        for i, idx in enumerate(top_5_indices):
            char = tokenizer.int_to_char.get(idx, 'UNKNOWN')
            prob = output_probs[idx]
            print(f"      {i+1}. token {idx} ('{char}'): {prob:.4f}")
        
        # 选择下一个token
        next_token = int(np.argmax(output_probs))
        next_char = tokenizer.int_to_char.get(next_token, 'UNKNOWN')
        
        print(f"    选择的下一个token: {next_token} ('{next_char}')")
        
        # 检查终止条件
        if next_token == tokenizer.END_TOKEN:
            print(f"    遇到END_TOKEN，停止生成")
            break
        
        # 检查有效性
        if next_token >= tokenizer.vocab_size or next_token < 0:
            print(f"    无效token，替换为PAD_TOKEN")
            next_token = tokenizer.PAD_TOKEN
        
        # 添加到序列
        output_sequence.append(next_token)
        current_token = next_token
        
        # 检查是否陷入循环
        if len(output_sequence) >= 3:
            last_3 = output_sequence[-3:]
            if len(set(last_3)) == 1:  # 最后3个token都相同
                print(f"    ⚠️  检测到循环模式: {last_3}")
                break
    
    # 解码输出序列
    decoded_output = tokenizer.decode(output_sequence)
    print(f"\n4. 最终输出序列: {output_sequence}")
    print(f"   解码结果: '{decoded_output}'")
    
    return output_sequence, decoded_output


def analyze_weight_patterns(model, tokenizer):
    """分析模型权重模式"""
    print(f"\n🔍 分析模型权重模式...")
    print("=" * 50)
    
    print(f"模型参数统计:")
    print(f"  embedding形状: {model.embedding.shape}")
    print(f"  encoder_weights形状: {model.encoder_weights.shape}")
    print(f"  decoder_weights形状: {model.decoder_weights.shape}")
    print(f"  output_weights形状: {model.output_weights.shape}")
    
    # 分析输出权重
    print(f"\n输出权重分析:")
    print(f"  权重范围: [{model.output_weights.min():.3f}, {model.output_weights.max():.3f}]")
    print(f"  权重均值: {model.output_weights.mean():.3f}")
    print(f"  权重标准差: {model.output_weights.std():.3f}")
    
    # 查看特定token的输出权重
    arrow_token = tokenizer.char_to_int.get('>', -1)
    dash_token = tokenizer.char_to_int.get('-', -1)
    end_token = tokenizer.END_TOKEN
    
    if arrow_token >= 0:
        print(f"\n'>' token ({arrow_token}) 的输出权重:")
        arrow_weights = model.output_weights[:, arrow_token]
        print(f"  范围: [{arrow_weights.min():.3f}, {arrow_weights.max():.3f}]")
        print(f"  均值: {arrow_weights.mean():.3f}")
    
    if dash_token >= 0:
        print(f"\n'-' token ({dash_token}) 的输出权重:")
        dash_weights = model.output_weights[:, dash_token]
        print(f"  范围: [{dash_weights.min():.3f}, {dash_weights.max():.3f}]")
        print(f"  均值: {dash_weights.mean():.3f}")
    
    print(f"\nEND_TOKEN ({end_token}) 的输出权重:")
    end_weights = model.output_weights[:, end_token]
    print(f"  范围: [{end_weights.min():.3f}, {end_weights.max():.3f}]")
    print(f"  均值: {end_weights.mean():.3f}")


def main():
    """主调试函数"""
    print("🐛 深度调试序列生成机制")
    print("=" * 60)
    
    # 1. 调试tokenizer
    tokenizer = debug_tokenizer()
    
    # 2. 加载模型
    print(f"\n🔍 加载模型...")
    model = ImprovedSimpleModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        max_length=50,
        learning_rate=0.003
    )
    
    model_path = 'outputs/trained_models/robust_model_Level_1_鲁棒版.npz'
    if not model.load_model(model_path):
        print(f"❌ 无法加载模型: {model_path}")
        return
    
    print(f"✅ 模型加载成功")
    
    # 3. 分析权重模式
    analyze_weight_patterns(model, tokenizer)
    
    # 4. 逐步调试预测过程
    test_inputs = [
        "p -> q",
        "~p -> r"
    ]
    
    for input_text in test_inputs:
        debug_model_prediction_step_by_step(model, tokenizer, input_text)
    
    print(f"\n🎯 调试总结:")
    print(f"通过以上分析，我们可以确定序列生成循环的具体原因...")


if __name__ == "__main__":
    main()
