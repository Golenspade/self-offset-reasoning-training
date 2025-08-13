"""
测试高级数据生成功能
验证递归生成、新噪声类型和配置驱动系统
"""

from logic_utils import (
    generate_recursive_implication,
    add_noise_type3,
    add_noise_type4,
    add_noise,
    verify_equivalence
)


def test_recursive_generation():
    """测试递归命题生成"""
    print("=== 测试递归命题生成 ===")
    
    for depth in [1, 2, 3, 4]:
        print(f"\n深度 {depth} 的命题示例:")
        for i in range(3):
            prop = generate_recursive_implication(max_depth=depth)
            print(f"  {i+1}. {prop}")
    
    print("✅ 递归生成测试完成")


def test_new_noise_types():
    """测试新的噪声类型"""
    print("\n=== 测试新噪声类型 ===")
    
    test_props = [
        "p -> q",
        "(p & q) -> r",
        "p | q -> s"
    ]
    
    for prop in test_props:
        print(f"\n原始命题: {prop}")
        
        # 测试噪声类型3
        noisy3 = add_noise_type3(prop)
        print(f"  噪声类型3 (括号): {noisy3}")
        
        # 测试噪声类型4
        noisy4 = add_noise_type4(prop)
        print(f"  噪声类型4 (交换): {noisy4}")
    
    print("✅ 新噪声类型测试完成")


def test_multi_noise_application():
    """测试多次噪声应用"""
    print("\n=== 测试多次噪声应用 ===")
    
    original = "p -> q"
    noise_types = ['type1', 'type2', 'type3', 'type4']
    
    for num_apps in [1, 2, 3]:
        print(f"\n应用 {num_apps} 次噪声:")
        for i in range(3):
            noisy = add_noise(original, noise_types, num_apps)
            print(f"  {i+1}. {noisy}")
    
    print("✅ 多次噪声应用测试完成")


def test_complex_equivalence():
    """测试复杂命题的等价性验证"""
    print("\n=== 测试复杂等价性验证 ===")
    
    test_cases = [
        # 递归生成的复杂命题
        ("((p & q) | r) -> s", "~s -> ~((p & q) | r)"),
        ("(p -> q) & (r -> s)", "(p -> q) & (r -> s)"),  # 自身等价
        # 带噪声的等价性
        ("p -> q", "(~p | q)"),  # 基本等价
        ("~~p -> q", "p -> q"),  # 双重否定
    ]
    
    for prop1, prop2 in test_cases:
        try:
            is_equiv = verify_equivalence(prop1, prop2)
            print(f"  '{prop1}' ≡ '{prop2}': {is_equiv}")
        except Exception as e:
            print(f"  '{prop1}' ≡ '{prop2}': 错误 - {e}")
    
    print("✅ 复杂等价性验证测试完成")


def main():
    """主测试函数"""
    print("🧪 开始测试高级数据生成功能...")
    print("=" * 60)
    
    test_recursive_generation()
    test_new_noise_types()
    test_multi_noise_application()
    test_complex_equivalence()
    
    print("\n🎉 所有测试完成！")


if __name__ == "__main__":
    main()
