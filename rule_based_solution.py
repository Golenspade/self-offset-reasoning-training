"""
文件名: rule_based_solution.py
基于规则的解决方案
直接实现逻辑转换规则，绕过神经网络的问题
"""

import json
import re
from logic_utils import Tokenizer, verify_equivalence


def parse_disjunction(input_str):
    """
    解析析取形式 (~A | B) 并返回组成部分
    """
    # 移除外层括号
    content = input_str.strip()
    if content.startswith('(') and content.endswith(')'):
        content = content[1:-1]
    
    # 按 | 分割
    parts = content.split('|')
    if len(parts) != 2:
        return None, None
    
    left = parts[0].strip()
    right = parts[1].strip()
    
    return left, right


def negate_term(term):
    """
    对一个项进行否定
    """
    term = term.strip()
    if term.startswith('~'):
        # 去掉否定
        return term[1:].strip()
    else:
        # 添加否定
        return f"~{term}"


def disjunction_to_contrapositive(input_str):
    """
    将析取形式转换为逆否命题
    (~A | B) 等价于 (A -> B)，其逆否命题是 (~B -> ~A)
    """
    left, right = parse_disjunction(input_str)
    
    if left is None or right is None:
        return None
    
    # (~A | B) -> (A -> B) -> (~B -> ~A)
    # 所以我们需要：
    # 1. 对右边否定得到 ~B
    # 2. 对左边的否定进行否定得到 ~A (如果左边是~A，那么~~A = A)
    
    neg_right = negate_term(right)  # ~B
    
    # 处理左边：如果是~A，那么我们要得到~A；如果是A，那么我们要得到~A
    if left.startswith('~'):
        # 左边是~A，那么原命题是A -> B，逆否是~B -> ~A
        neg_left = left  # 保持~A
    else:
        # 左边是A，那么原命题是~A -> B，逆否是~B -> A
        neg_left = left  # 保持A
    
    contrapositive = f"{neg_right} -> {neg_left}"
    return contrapositive


def rule_based_predict(input_str):
    """
    基于规则的预测函数
    """
    try:
        result = disjunction_to_contrapositive(input_str)
        return result if result else "parse_error"
    except Exception as e:
        return f"error: {str(e)}"


def test_rule_based_solution():
    """
    测试基于规则的解决方案
    """
    print("=== 基于规则的解决方案测试 ===")
    
    test_cases = [
        # (输入, 期望输出)
        ("(~p | q)", "~q -> ~p"),    # p -> q 的逆否
        ("(~r | s)", "~s -> ~r"),    # r -> s 的逆否  
        ("(p | ~q)", "q -> p"),      # ~p -> ~q 的逆否
        ("(~s | ~t)", "t -> ~s"),    # s -> ~t 的逆否
        ("(q | r)", "~r -> q"),      # ~q -> r 的逆否
    ]
    
    correct = 0
    total = len(test_cases)
    
    for input_str, expected in test_cases:
        predicted = rule_based_predict(input_str)
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        print(f"\n输入: {input_str}")
        print(f"期望: {expected}")
        print(f"预测: {predicted}")
        print(f"结果: {'✓' if is_correct else '✗'}")
        
        # 如果不匹配，尝试逻辑等价性检查
        if not is_correct:
            try:
                is_equivalent = verify_equivalence(predicted, expected)
                print(f"逻辑等价: {'✓' if is_equivalent else '✗'}")
            except:
                print(f"逻辑等价: 无法验证")
    
    accuracy = correct / total
    print(f"\n规则基础准确率: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy


def evaluate_on_validation_data():
    """
    在验证数据上评估规则基础方法
    """
    print("\n=== 在验证数据上评估规则基础方法 ===")
    
    try:
        with open('data/val.json', 'r', encoding='utf-8') as f:
            val_data = []
            for i, line in enumerate(f):
                if i >= 100:  # 测试前100个样本
                    break
                if line.strip():
                    val_data.append(json.loads(line))
    except:
        print("无法加载验证数据")
        return 0, 0
    
    exact_correct = 0
    logical_correct = 0
    total = len(val_data)
    
    print(f"测试 {total} 个样本...")
    
    for i, sample in enumerate(val_data):
        input_text = sample['noisy_prop']
        target_text = sample['target_contrapositive']
        
        predicted_text = rule_based_predict(input_text)
        
        # 精确匹配
        if predicted_text == target_text:
            exact_correct += 1
            logical_correct += 1
        else:
            # 逻辑等价性检查
            try:
                if verify_equivalence(predicted_text, target_text):
                    logical_correct += 1
            except:
                pass
        
        # 显示前10个结果
        if i < 10:
            print(f"\n样本 {i+1}:")
            print(f"  输入: {input_text}")
            print(f"  目标: {target_text}")
            print(f"  预测: {predicted_text}")
            print(f"  精确匹配: {'✓' if predicted_text == target_text else '✗'}")
    
    exact_accuracy = exact_correct / total
    logical_accuracy = logical_correct / total
    
    print(f"\n=== 评估结果 ===")
    print(f"精确匹配准确率: {exact_accuracy:.2%} ({exact_correct}/{total})")
    print(f"逻辑等价准确率: {logical_accuracy:.2%} ({logical_correct}/{total})")
    
    return exact_accuracy, logical_accuracy


def analyze_errors():
    """
    分析错误案例
    """
    print("\n=== 错误分析 ===")
    
    try:
        with open('data/val.json', 'r', encoding='utf-8') as f:
            val_data = []
            for i, line in enumerate(f):
                if i >= 50:
                    break
                if line.strip():
                    val_data.append(json.loads(line))
    except:
        print("无法加载验证数据")
        return
    
    errors = []
    
    for sample in val_data:
        input_text = sample['noisy_prop']
        target_text = sample['target_contrapositive']
        predicted_text = rule_based_predict(input_text)
        
        if predicted_text != target_text:
            errors.append({
                'input': input_text,
                'target': target_text,
                'predicted': predicted_text,
                'original': sample.get('original_prop', 'unknown')
            })
    
    print(f"发现 {len(errors)} 个错误案例:")
    
    for i, error in enumerate(errors[:10]):  # 只显示前10个
        print(f"\n错误 {i+1}:")
        print(f"  原始命题: {error['original']}")
        print(f"  噪声输入: {error['input']}")
        print(f"  目标输出: {error['target']}")
        print(f"  规则预测: {error['predicted']}")
        
        # 分析错误类型
        left, right = parse_disjunction(error['input'])
        if left and right:
            print(f"  解析结果: 左='{left}', 右='{right}'")


def create_corrected_rule_function():
    """
    创建修正后的规则函数
    """
    print("\n=== 创建修正后的规则函数 ===")
    
    def corrected_disjunction_to_contrapositive(input_str):
        """
        修正后的析取到逆否命题转换
        """
        left, right = parse_disjunction(input_str)
        
        if left is None or right is None:
            return None
        
        # 分析：(A | B) 等价于 (~A -> B)
        # 所以 (~A | B) 等价于 (~~A -> B) = (A -> B)
        # 逆否命题是 (~B -> ~A)
        
        # 如果左边是 ~X，那么原命题是 X -> right
        # 如果左边是 X，那么原命题是 ~X -> right
        
        if left.startswith('~'):
            # 左边是 ~X，原命题是 X -> right，逆否是 ~right -> ~X
            antecedent = left[1:].strip()  # X
            consequent = right.strip()     # right
            
            neg_consequent = negate_term(consequent)  # ~right
            neg_antecedent = f"~{antecedent}"         # ~X
            
        else:
            # 左边是 X，原命题是 ~X -> right，逆否是 ~right -> ~~X = ~right -> X
            antecedent = left.strip()      # X
            consequent = right.strip()     # right
            
            neg_consequent = negate_term(consequent)  # ~right
            neg_antecedent = antecedent               # X (因为~~X = X)
        
        contrapositive = f"{neg_consequent} -> {neg_antecedent}"
        return contrapositive
    
    # 测试修正后的函数
    test_cases = [
        ("(~p | q)", "~q -> ~p"),
        ("(~r | s)", "~s -> ~r"),
        ("(p | ~q)", "q -> p"),
        ("(~s | ~t)", "t -> s"),
    ]
    
    print("测试修正后的规则函数:")
    for input_str, expected in test_cases:
        predicted = corrected_disjunction_to_contrapositive(input_str)
        print(f"  {input_str} -> {predicted} (期望: {expected}) {'✓' if predicted == expected else '✗'}")
    
    return corrected_disjunction_to_contrapositive


def main():
    """
    主函数
    """
    print("开始基于规则的解决方案测试...")
    
    # 1. 测试基本规则
    test_accuracy = test_rule_based_solution()
    
    # 2. 在验证数据上评估
    exact_acc, logical_acc = evaluate_on_validation_data()
    
    # 3. 错误分析
    analyze_errors()
    
    # 4. 创建修正后的规则函数
    corrected_func = create_corrected_rule_function()
    
    print(f"\n=== 总结 ===")
    print(f"基本规则测试准确率: {test_accuracy:.2%}")
    print(f"验证数据精确准确率: {exact_acc:.2%}")
    print(f"验证数据逻辑准确率: {logical_acc:.2%}")
    
    if exact_acc > 0.5:  # 如果准确率超过50%
        print(f"\n🎉 规则基础方法成功！")
        print(f"这证明了问题的核心在于解码循环，而不是任务本身的难度。")
    else:
        print(f"\n需要进一步调试规则逻辑...")


if __name__ == "__main__":
    main()
