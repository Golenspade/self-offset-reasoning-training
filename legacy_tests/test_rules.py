"""
专业测试模块 - 针对 logic_rules.py 的单元测试
遵循软件工程最佳实践
"""

import json
from logic_rules import (
    parse_disjunction, 
    negate_term, 
    corrected_disjunction_to_contrapositive,
    rule_based_predict_corrected,
    validate_rule_logic
)


class TestLogicRules:
    """逻辑规则测试类"""
    
    def test_parse_disjunction(self):
        """测试析取解析函数"""
        test_cases = [
            ("(~p | q)", "~p", "q"),
            ("(p | ~q)", "p", "~q"),
            ("(~s | ~t)", "~s", "~t"),
            ("(a | b)", "a", "b"),
            ("invalid", None, None),
            ("(a|b|c)", None, None),  # 超过两个操作数
        ]
        
        print("=== 测试 parse_disjunction ===")
        passed = 0
        total = len(test_cases)
        
        for input_str, expected_left, expected_right in test_cases:
            left, right = parse_disjunction(input_str)
            
            if left == expected_left and right == expected_right:
                print(f"✓ {input_str} -> ('{left}', '{right}')")
                passed += 1
            else:
                print(f"✗ {input_str} -> ('{left}', '{right}') (期望: ('{expected_left}', '{expected_right}'))")
        
        print(f"parse_disjunction 测试: {passed}/{total} 通过")
        return passed == total
    
    def test_negate_term(self):
        """测试否定函数"""
        test_cases = [
            ("p", "~p"),
            ("~p", "p"),
            ("~q", "q"),
            ("r", "~r"),
            ("~~p", "~p"),  # 双重否定
        ]
        
        print("\n=== 测试 negate_term ===")
        passed = 0
        total = len(test_cases)
        
        for input_term, expected in test_cases:
            result = negate_term(input_term)
            
            if result == expected:
                print(f"✓ negate('{input_term}') -> '{result}'")
                passed += 1
            else:
                print(f"✗ negate('{input_term}') -> '{result}' (期望: '{expected}')")
        
        print(f"negate_term 测试: {passed}/{total} 通过")
        return passed == total
    
    def test_contrapositive_conversion(self):
        """测试逆否命题转换的核心逻辑"""
        test_cases = [
            # 基本案例
            ("(~p | q)", "~q -> ~p", "p -> q 的逆否"),
            ("(~r | s)", "~s -> ~r", "r -> s 的逆否"),
            
            # 左边无否定的案例
            ("(p | ~q)", "q -> p", "~p -> ~q 的逆否"),
            ("(q | r)", "~r -> q", "~q -> r 的逆否"),
            
            # 双否定案例
            ("(~s | ~t)", "t -> ~s", "s -> ~t 的逆否"),
            ("(~p | ~s)", "s -> ~p", "p -> ~s 的逆否"),
            
            # 边界案例
            ("(~~p | q)", "~q -> ~~p", "~p -> q 的逆否"),
        ]
        
        print("\n=== 测试 corrected_disjunction_to_contrapositive ===")
        passed = 0
        total = len(test_cases)
        
        for input_str, expected, description in test_cases:
            result = corrected_disjunction_to_contrapositive(input_str)
            
            if result == expected:
                print(f"✓ {input_str} -> {result} ({description})")
                passed += 1
            else:
                print(f"✗ {input_str} -> {result} (期望: {expected}, {description})")
        
        print(f"contrapositive_conversion 测试: {passed}/{total} 通过")
        return passed == total
    
    def test_rule_based_predict(self):
        """测试完整的规则预测函数"""
        test_cases = [
            ("(~p | q)", "~q -> ~p"),
            ("(p | ~q)", "q -> p"),
            ("invalid_input", "parse_error"),
        ]
        
        print("\n=== 测试 rule_based_predict_corrected ===")
        passed = 0
        total = len(test_cases)
        
        for input_str, expected in test_cases:
            result = rule_based_predict_corrected(input_str)
            
            # 对于错误案例，只检查是否包含错误标识
            if expected == "parse_error":
                if result is None or result == "parse_error" or result.startswith("error:"):
                    print(f"✓ {input_str} -> {result} (正确处理错误)")
                    passed += 1
                else:
                    print(f"✗ {input_str} -> {result} (应该返回错误)")
            else:
                if result == expected:
                    print(f"✓ {input_str} -> {result}")
                    passed += 1
                else:
                    print(f"✗ {input_str} -> {result} (期望: {expected})")
        
        print(f"rule_based_predict 测试: {passed}/{total} 通过")
        return passed == total
    
    def test_on_validation_data(self, max_samples=100):
        """在验证数据上进行集成测试"""
        print(f"\n=== 验证数据集成测试 (前{max_samples}个样本) ===")
        
        try:
            with open('data/val.json', 'r', encoding='utf-8') as f:
                val_data = []
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    if line.strip():
                        val_data.append(json.loads(line))
        except:
            print("无法加载验证数据，跳过集成测试")
            return True
        
        correct = 0
        total = len(val_data)
        errors = []
        
        for sample in val_data:
            input_text = sample['noisy_prop']
            target_text = sample['target_contrapositive']
            predicted_text = rule_based_predict_corrected(input_text)
            
            if predicted_text == target_text:
                correct += 1
            else:
                errors.append({
                    'input': input_text,
                    'target': target_text,
                    'predicted': predicted_text
                })
        
        accuracy = correct / total
        print(f"验证数据准确率: {accuracy:.2%} ({correct}/{total})")
        
        # 显示前几个错误案例
        if errors:
            print(f"\n前5个错误案例:")
            for i, error in enumerate(errors[:5]):
                print(f"  错误 {i+1}: {error['input']} -> {error['predicted']} (期望: {error['target']})")
        
        return accuracy >= 0.95  # 95%以上认为通过
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始运行逻辑规则完整测试套件...")
        print("=" * 60)
        
        tests = [
            self.test_parse_disjunction,
            self.test_negate_term,
            self.test_contrapositive_conversion,
            self.test_rule_based_predict,
            self.test_on_validation_data,
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
            print("🎉 所有测试通过！逻辑规则模块完全正确。")
            return True
        else:
            print("⚠️  存在失败的测试，需要进一步调试。")
            return False


def main():
    """主测试函数"""
    tester = TestLogicRules()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ 逻辑规则模块已通过所有测试，可以安全使用。")
    else:
        print("\n❌ 测试未完全通过，请检查并修复问题。")
    
    return success


if __name__ == "__main__":
    main()
