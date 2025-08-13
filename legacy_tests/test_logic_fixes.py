"""
测试修复后的logic_utils.py中的关键函数
验证evaluate_formula, negate_formula, add_noise_type2的修复效果
"""

from logic_utils import evaluate_formula, negate_formula, add_noise_type2, verify_equivalence


def test_evaluate_formula():
    """测试修复后的evaluate_formula函数"""
    print("=== 测试 evaluate_formula 修复效果 ===")
    
    test_cases = [
        # (公式, 变量赋值, 期望结果)
        ("p", {"p": True}, True),
        ("p", {"p": False}, False),
        ("~p", {"p": True}, False),
        ("~p", {"p": False}, True),
        ("p & q", {"p": True, "q": True}, True),
        ("p & q", {"p": True, "q": False}, False),
        ("p | q", {"p": False, "q": True}, True),
        ("p | q", {"p": False, "q": False}, False),
        ("p -> q", {"p": True, "q": True}, True),
        ("p -> q", {"p": True, "q": False}, False),
        ("p -> q", {"p": False, "q": True}, True),
        ("p -> q", {"p": False, "q": False}, True),
        # 复杂表达式测试
        ("(p & q) -> r", {"p": True, "q": True, "r": False}, False),
        ("(p & q) -> r", {"p": True, "q": True, "r": True}, True),
        ("~(p & q)", {"p": True, "q": False}, True),
        ("~(p & q)", {"p": True, "q": True}, False),
        # 优先级测试
        ("p | q & r", {"p": False, "q": True, "r": False}, False),  # 应该是 p | (q & r)
        ("(p | q) & r", {"p": True, "q": False, "r": False}, False),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for formula, assignment, expected in test_cases:
        try:
            result = evaluate_formula(formula, assignment)
            if result == expected:
                print(f"✓ {formula} with {assignment} = {result}")
                passed += 1
            else:
                print(f"✗ {formula} with {assignment} = {result} (期望: {expected})")
        except Exception as e:
            print(f"✗ {formula} with {assignment} 出错: {e}")
    
    print(f"\nevaluate_formula 测试结果: {passed}/{total} 通过")
    return passed == total


def test_negate_formula():
    """测试修复后的negate_formula函数"""
    print("\n=== 测试 negate_formula 修复效果 ===")
    
    test_cases = [
        # (输入, 期望输出)
        ("p", "~p"),
        ("~p", "p"),
        ("~q", "q"),
        ("(p & q)", "~(p & q)"),
        ("~(p & q)", "p & q"),  # 修复后应该正确处理
        ("p & q", "~(p & q)"),  # 复合表达式应该加括号
        ("p | q", "~(p | q)"),
        ("~~p", "~p"),  # 双重否定
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_formula, expected in test_cases:
        try:
            result = negate_formula(input_formula)
            if result == expected:
                print(f"✓ negate('{input_formula}') = '{result}'")
                passed += 1
            else:
                print(f"✗ negate('{input_formula}') = '{result}' (期望: '{expected}')")
        except Exception as e:
            print(f"✗ negate('{input_formula}') 出错: {e}")
    
    print(f"\nnegate_formula 测试结果: {passed}/{total} 通过")
    return passed == total


def test_add_noise_type2():
    """测试修复后的add_noise_type2函数"""
    print("\n=== 测试 add_noise_type2 修复效果 ===")
    
    test_cases = [
        "p -> q",
        "pr -> q",  # 测试不会错误替换 pr 中的 p
        "p & pr",   # 测试边界情况
        "(p | q) -> r",
    ]
    
    passed = 0
    total = len(test_cases)
    
    for formula in test_cases:
        try:
            # 多次测试以确保随机性
            results = []
            for _ in range(5):
                result = add_noise_type2(formula)
                results.append(result)
            
            # 检查是否有变化（应该有一些结果包含~~）
            has_double_negation = any('~~' in r for r in results)
            
            # 检查是否没有错误替换（如 pr 变成 ~~pr）
            no_wrong_replacement = all('~~pr' not in r and '~~pq' not in r for r in results)
            
            if has_double_negation and no_wrong_replacement:
                print(f"✓ '{formula}' -> 示例结果: {results[0]}")
                passed += 1
            else:
                print(f"✗ '{formula}' -> 结果: {results}")
                print(f"    双重否定: {has_double_negation}, 无错误替换: {no_wrong_replacement}")
        except Exception as e:
            print(f"✗ '{formula}' 出错: {e}")
    
    print(f"\nadd_noise_type2 测试结果: {passed}/{total} 通过")
    return passed == total


def test_verify_equivalence():
    """测试修复后的verify_equivalence函数"""
    print("\n=== 测试 verify_equivalence 修复效果 ===")
    
    test_cases = [
        # (公式1, 公式2, 期望结果)
        ("p -> q", "~p | q", True),  # 基本等价
        ("~(p & q)", "~p | ~q", True),  # 德摩根定律
        ("p & q", "q & p", True),  # 交换律
        ("p | q", "q | p", True),  # 交换律
        ("p -> q", "~q -> ~p", True),  # 逆否命题
        ("p", "q", False),  # 不等价
        ("p & q", "p | q", False),  # 不等价
        # 复杂等价性
        ("(p & q) -> r", "~(p & q) | r", True),
        ("~(p | q)", "~p & ~q", True),  # 德摩根定律
    ]
    
    passed = 0
    total = len(test_cases)
    
    for formula1, formula2, expected in test_cases:
        try:
            result = verify_equivalence(formula1, formula2)
            if result == expected:
                print(f"✓ '{formula1}' ≡ '{formula2}': {result}")
                passed += 1
            else:
                print(f"✗ '{formula1}' ≡ '{formula2}': {result} (期望: {expected})")
        except Exception as e:
            print(f"✗ '{formula1}' ≡ '{formula2}' 出错: {e}")
    
    print(f"\nverify_equivalence 测试结果: {passed}/{total} 通过")
    return passed == total


def main():
    """主测试函数"""
    print("开始测试修复后的logic_utils.py函数...")
    print("=" * 60)
    
    tests = [
        test_evaluate_formula,
        test_negate_formula,
        test_add_noise_type2,
        test_verify_equivalence,
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            if test():
                passed_tests += 1
                print("✅ 测试通过\n")
            else:
                print("❌ 测试失败\n")
        except Exception as e:
            print(f"❌ 测试异常: {e}\n")
    
    print("=" * 60)
    print(f"测试总结: {passed_tests}/{total_tests} 个测试套件通过")
    
    if passed_tests == total_tests:
        print("🎉 所有修复都成功！logic_utils.py 现在完全可靠。")
        return True
    else:
        print("⚠️  仍有问题需要修复。")
        return False


if __name__ == "__main__":
    main()
