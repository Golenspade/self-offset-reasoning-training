"""
逻辑规则模块 - 经过验证的核心逻辑转换函数
基于 rule_based_solution.py 中的修正逻辑
"""

import re


def parse_disjunction(input_str):
    """
    解析析取形式 (~A | B) 并返回组成部分
    """
    # 移除外层括号
    content = input_str.strip()
    if content.startswith("(") and content.endswith(")"):
        content = content[1:-1]

    # 按 | 分割
    parts = content.split("|")
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
    if term.startswith("~"):
        # 去掉否定
        return term[1:].strip()
    else:
        # 添加否定
        return f"~{term}"


def corrected_disjunction_to_contrapositive(input_str):
    """
    修正后的析取到逆否命题转换

    核心逻辑：
    - (A | B) 等价于 (~A -> B)
    - (~A | B) 等价于 (~~A -> B) = (A -> B)，逆否命题是 (~B -> ~A)
    - (A | B) 等价于 (~A -> B)，逆否命题是 (~B -> ~~A) = (~B -> A)
    """
    left, right = parse_disjunction(input_str)

    if left is None or right is None:
        return None

    if left.startswith("~"):
        # 左边是 ~X，原命题是 X -> right，逆否是 ~right -> ~X
        antecedent = left[1:].strip()  # X
        consequent = right.strip()  # right

        neg_consequent = negate_term(consequent)  # ~right
        neg_antecedent = f"~{antecedent}"  # ~X

    else:
        # 左边是 X，原命题是 ~X -> right，逆否是 ~right -> ~~X = ~right -> X
        antecedent = left.strip()  # X
        consequent = right.strip()  # right

        neg_consequent = negate_term(consequent)  # ~right
        neg_antecedent = antecedent  # X (因为~~X = X)

    contrapositive = f"{neg_consequent} -> {neg_antecedent}"
    return contrapositive


def rule_based_predict_corrected(input_str):
    """
    修正后的基于规则的预测函数
    """
    try:
        result = corrected_disjunction_to_contrapositive(input_str)
        return result if result else "parse_error"
    except Exception as e:
        return f"error: {str(e)}"


def validate_rule_logic():
    """
    验证规则逻辑的正确性
    """
    test_cases = [
        # (输入, 期望输出, 说明)
        ("(~p | q)", "~q -> ~p", "p -> q 的逆否"),
        ("(~r | s)", "~s -> ~r", "r -> s 的逆否"),
        ("(p | ~q)", "q -> p", "~p -> ~q 的逆否"),
        ("(~s | ~t)", "t -> ~s", "s -> ~t 的逆否"),
        ("(q | r)", "~r -> q", "~q -> r 的逆否"),
        ("(~p | ~s)", "s -> ~p", "p -> ~s 的逆否"),
    ]

    print("=== 验证修正后的规则逻辑 ===")

    correct = 0
    total = len(test_cases)

    for input_str, expected, description in test_cases:
        predicted = rule_based_predict_corrected(input_str)
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        print(
            f"✓ {input_str} -> {predicted} ({description})"
            if is_correct
            else f"✗ {input_str} -> {predicted} (期望: {expected}, {description})"
        )

    accuracy = correct / total
    print(f"\n验证结果: {accuracy:.2%} ({correct}/{total})")

    return accuracy == 1.0


if __name__ == "__main__":
    # 运行验证
    success = validate_rule_logic()

    if success:
        print("\n🎉 所有测试通过！规则逻辑完全正确。")
    else:
        print("\n❌ 存在错误，需要进一步调试。")
