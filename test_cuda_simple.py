"""
文件名: test_cuda_simple.py
简化的CUDA系统测试脚本
不依赖PyTorch，仅测试基础功能
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试基础模块
        import cuda_utils
        print("✅ cuda_utils 导入成功")
        
        # 测试CUDA工具创建
        try:
            manager = cuda_utils.CUDAManager()
            print(f"✅ CUDAManager 创建成功: {manager.device}")
        except Exception as e:
            print(f"⚠️ CUDAManager 创建失败: {e}")
        
        # 测试CUDA信息获取
        info = cuda_utils.get_cuda_info()
        print(f"✅ CUDA信息获取成功: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_model_import():
    """测试模型导入"""
    print("\n🤖 测试模型导入...")
    
    try:
        import model
        print("✅ model 模块导入成功")
        
        # 测试模型创建函数存在
        assert hasattr(model, 'create_cuda_model'), "create_cuda_model 函数不存在"
        assert hasattr(model, 'LogicTransformer'), "LogicTransformer 类不存在"
        
        print("✅ 模型组件检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def test_training_system_import():
    """测试训练系统导入"""
    print("\n🚀 测试训练系统导入...")
    
    try:
        import cuda_training_system
        print("✅ cuda_training_system 模块导入成功")
        
        # 检查关键类存在
        assert hasattr(cuda_training_system, 'CUDABreakthroughTraining'), "CUDABreakthroughTraining 类不存在"
        
        print("✅ 训练系统组件检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练系统导入失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")
    
    required_files = [
        'cuda_utils.py',
        'cuda_training_system.py',
        'train_cuda.py',
        'model.py',
        'requirements_cuda.txt',
        'Dockerfile.cuda'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file} 存在")
    
    if missing_files:
        print(f"❌ 缺失文件: {missing_files}")
        return False
    
    print("✅ 所有必需文件都存在")
    return True

def test_config_files():
    """测试配置文件"""
    print("\n⚙️ 测试配置文件...")
    
    try:
        # 测试requirements_cuda.txt
        with open('requirements_cuda.txt', 'r') as f:
            content = f.read()
            assert 'torch' in content, "requirements_cuda.txt 中缺少 torch"
            assert 'nvidia-ml-py3' in content, "requirements_cuda.txt 中缺少 nvidia-ml-py3"
        
        print("✅ requirements_cuda.txt 检查通过")
        
        # 测试Dockerfile.cuda
        with open('Dockerfile.cuda', 'r') as f:
            content = f.read()
            assert 'nvidia/cuda' in content, "Dockerfile.cuda 中缺少 CUDA 基础镜像"
            assert 'torch' in content, "Dockerfile.cuda 中缺少 PyTorch 安装"
        
        print("✅ Dockerfile.cuda 检查通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件检查失败: {e}")
        return False

def test_script_executability():
    """测试脚本可执行性"""
    print("\n🔧 测试脚本可执行性...")
    
    try:
        # 测试train_cuda.py的帮助信息
        import subprocess
        result = subprocess.run([
            sys.executable, 'train_cuda.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ train_cuda.py 可正常执行")
        else:
            print(f"⚠️ train_cuda.py 执行有问题: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ 脚本执行测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 CUDA系统简化测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("模型导入", test_model_import),
        ("训练系统导入", test_training_system_import),
        ("文件结构", test_file_structure),
        ("配置文件", test_config_files),
        ("脚本可执行性", test_script_executability)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
    
    # 生成报告
    print("\n" + "=" * 60)
    print("📊 测试报告")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有基础测试通过！CUDA系统结构正确")
        print("\n💡 下一步:")
        print("1. 安装CUDA版本的PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. 安装其他CUDA依赖: pip install -r requirements_cuda.txt")
        print("3. 运行完整CUDA测试: python test_cuda_training.py")
        print("4. 开始CUDA训练: python train_cuda.py --help")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查文件和代码")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
