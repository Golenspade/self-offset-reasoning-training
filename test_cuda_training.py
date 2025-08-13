"""
文件名: test_cuda_training.py
CUDA训练系统测试脚本
验证GPU加速功能和性能
"""
import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

try:
    import torch
    from cuda_utils import CUDAManager, print_cuda_summary
    from cuda_training_system import CUDABreakthroughTraining
    from model import create_cuda_model
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CUDA相关模块导入失败: {e}")
    CUDA_AVAILABLE = False


class CUDATrainingTester:
    """CUDA训练系统测试器"""
    
    def __init__(self):
        self.test_dir = None
        self.logger = self._setup_logging()
        self.test_results = {}
    
    def _setup_logging(self):
        """设置测试日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def test_cuda_environment(self) -> bool:
        """测试CUDA环境"""
        self.logger.info("🔍 测试CUDA环境...")
        
        try:
            if not CUDA_AVAILABLE:
                self.logger.warning("⚠️ CUDA模块不可用")
                self.test_results['cuda_environment'] = False
                return False
            
            # 检查PyTorch CUDA支持
            assert torch.cuda.is_available(), "CUDA不可用"
            
            device_count = torch.cuda.device_count()
            assert device_count > 0, "未检测到GPU设备"
            
            # 测试基本GPU操作
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            
            assert result.device == device, "GPU计算失败"
            
            self.logger.info(f"✅ CUDA环境测试通过 ({device_count} GPU)")
            self.test_results['cuda_environment'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CUDA环境测试失败: {e}")
            self.test_results['cuda_environment'] = False
            return False
    
    def test_cuda_manager(self) -> bool:
        """测试CUDA管理器"""
        self.logger.info("🔧 测试CUDA管理器...")
        
        try:
            # 创建CUDA管理器
            cuda_manager = CUDAManager()
            
            # 测试设备选择
            assert cuda_manager.device is not None, "设备选择失败"
            
            # 测试内存信息获取
            memory_info = cuda_manager.get_memory_info()
            if cuda_manager.device.type == 'cuda':
                assert 'total_memory' in memory_info, "内存信息获取失败"
                assert memory_info['total_memory'] > 0, "GPU内存信息无效"
            
            # 测试内存监控
            with cuda_manager.memory_monitor("测试操作"):
                if cuda_manager.device.type == 'cuda':
                    test_tensor = torch.randn(1000, 1000, device=cuda_manager.device)
                    del test_tensor
            
            # 测试缓存清理
            cuda_manager.clear_cache()
            
            self.logger.info("✅ CUDA管理器测试通过")
            self.test_results['cuda_manager'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CUDA管理器测试失败: {e}")
            self.test_results['cuda_manager'] = False
            return False
    
    def test_cuda_model(self) -> bool:
        """测试CUDA模型"""
        self.logger.info("🤖 测试CUDA模型...")
        
        try:
            # 创建CUDA模型
            model, device = create_cuda_model(
                vocab_size=1000,
                device='auto',
                d_model=64,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2
            )
            
            # 验证模型在正确设备上
            for param in model.parameters():
                assert param.device == device, f"模型参数设备不匹配: {param.device} vs {device}"
            
            # 测试前向传播
            batch_size = 4
            seq_len = 10
            
            src = torch.randint(0, 1000, (seq_len, batch_size), device=device)
            tgt = torch.randint(0, 1000, (seq_len, batch_size), device=device)
            
            with torch.no_grad():
                output = model(src, tgt[:-1])
                assert output.device == device, "模型输出设备不匹配"
                assert output.shape == (seq_len-1, batch_size, 1000), f"输出形状错误: {output.shape}"
            
            self.logger.info("✅ CUDA模型测试通过")
            self.test_results['cuda_model'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CUDA模型测试失败: {e}")
            self.test_results['cuda_model'] = False
            return False
    
    def test_mixed_precision(self) -> bool:
        """测试混合精度训练"""
        self.logger.info("🔥 测试混合精度训练...")
        
        try:
            from torch.cuda.amp import GradScaler, autocast
            
            # 创建简单模型
            model, device = create_cuda_model(
                vocab_size=100,
                device='auto',
                d_model=32,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1
            )
            
            if device.type != 'cuda':
                self.logger.info("ℹ️ CPU模式，跳过混合精度测试")
                self.test_results['mixed_precision'] = True
                return True
            
            # 检查是否支持混合精度
            cuda_manager = CUDAManager()
            if not cuda_manager.supports_mixed_precision():
                self.logger.info("ℹ️ GPU不支持混合精度，跳过测试")
                self.test_results['mixed_precision'] = True
                return True
            
            # 创建优化器和scaler
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scaler = GradScaler()
            criterion = torch.nn.CrossEntropyLoss()
            
            # 测试混合精度训练步骤
            model.train()
            
            src = torch.randint(0, 100, (5, 2), device=device)
            tgt = torch.randint(0, 100, (5, 2), device=device)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(src, tgt[:-1])
                loss = criterion(output.reshape(-1, 100), tgt[1:].reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            assert not torch.isnan(loss), "混合精度训练产生NaN"
            
            self.logger.info("✅ 混合精度训练测试通过")
            self.test_results['mixed_precision'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 混合精度训练测试失败: {e}")
            self.test_results['mixed_precision'] = False
            return False
    
    def test_cuda_training_system(self) -> bool:
        """测试CUDA训练系统"""
        self.logger.info("🚀 测试CUDA训练系统...")
        
        try:
            # 创建测试配置
            config = {
                'hidden_size': 64,
                'num_heads': 4,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'dim_feedforward': 128,
                'max_length': 20,
                'batch_size': 4,
                'learning_rate': 0.01,
                'weight_decay': 1e-5,
                'use_mixed_precision': True,
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'early_stopping_patience': 5,
                'epochs': 3
            }
            
            # 创建训练系统
            trainer = CUDABreakthroughTraining(config)
            
            # 创建测试数据
            test_data = []
            for i in range(20):
                test_data.append({
                    'noisy_prop': f"p{i} -> q{i}",
                    'target_contrapositive': f"~q{i} -> ~p{i}",
                    'complexity': 'simple'
                })
            
            # 测试批次准备
            batch = test_data[:4]
            src_batch, tgt_input, tgt_output = trainer.prepare_batch_cuda(batch)
            
            assert src_batch is not None, "批次准备失败"
            assert src_batch.device == trainer.device, "批次数据设备不匹配"
            
            # 测试训练步骤
            metrics = trainer.train_step_cuda(batch)
            
            assert 'loss' in metrics, "训练步骤未返回损失"
            assert not torch.isnan(torch.tensor(metrics['loss'])), "训练损失为NaN"
            
            # 测试验证
            val_metrics = trainer.validate_cuda(test_data[:10])
            assert 'val_loss' in val_metrics, "验证未返回损失"
            
            self.logger.info("✅ CUDA训练系统测试通过")
            self.test_results['cuda_training_system'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ CUDA训练系统测试失败: {e}")
            self.test_results['cuda_training_system'] = False
            return False
    
    def test_performance_comparison(self) -> bool:
        """测试性能对比"""
        self.logger.info("⚡ 测试性能对比...")
        
        try:
            if not torch.cuda.is_available():
                self.logger.info("ℹ️ CUDA不可用，跳过性能对比")
                self.test_results['performance_comparison'] = True
                return True
            
            # 创建测试模型
            vocab_size = 1000
            batch_size = 16
            seq_len = 50
            
            # CPU模型
            cpu_model, _ = create_cuda_model(vocab_size, device='cpu', d_model=128)
            
            # GPU模型
            gpu_model, _ = create_cuda_model(vocab_size, device='cuda', d_model=128)
            
            # 创建测试数据
            cpu_src = torch.randint(0, vocab_size, (seq_len, batch_size))
            cpu_tgt = torch.randint(0, vocab_size, (seq_len, batch_size))
            
            gpu_src = cpu_src.cuda()
            gpu_tgt = cpu_tgt.cuda()
            
            # CPU性能测试
            cpu_model.eval()
            with torch.no_grad():
                cpu_start = time.time()
                for _ in range(10):
                    _ = cpu_model(cpu_src, cpu_tgt[:-1])
                cpu_time = time.time() - cpu_start
            
            # GPU性能测试
            gpu_model.eval()
            with torch.no_grad():
                # 预热
                _ = gpu_model(gpu_src, gpu_tgt[:-1])
                torch.cuda.synchronize()
                
                gpu_start = time.time()
                for _ in range(10):
                    _ = gpu_model(gpu_src, gpu_tgt[:-1])
                torch.cuda.synchronize()
                gpu_time = time.time() - gpu_start
            
            speedup = cpu_time / gpu_time
            
            self.logger.info(f"📊 性能对比结果:")
            self.logger.info(f"  CPU时间: {cpu_time:.3f}s")
            self.logger.info(f"  GPU时间: {gpu_time:.3f}s")
            self.logger.info(f"  加速比: {speedup:.2f}x")
            
            # GPU应该比CPU快（至少不能慢太多）
            assert speedup > 0.5, f"GPU性能异常，加速比仅为{speedup:.2f}x"
            
            self.test_results['performance_comparison'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 性能对比测试失败: {e}")
            self.test_results['performance_comparison'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有CUDA测试"""
        self.logger.info("🧪 开始CUDA训练系统完整测试")
        self.logger.info("=" * 60)
        
        # 首先打印CUDA环境信息
        print_cuda_summary()
        
        # 运行测试
        tests = [
            ('CUDA环境', self.test_cuda_environment),
            ('CUDA管理器', self.test_cuda_manager),
            ('CUDA模型', self.test_cuda_model),
            ('混合精度', self.test_mixed_precision),
            ('CUDA训练系统', self.test_cuda_training_system),
            ('性能对比', self.test_performance_comparison)
        ]
        
        for test_name, test_func in tests:
            self.logger.info(f"\n🔍 测试: {test_name}")
            test_func()
        
        # 生成测试报告
        self._generate_test_report()
        
        return self.test_results
    
    def _generate_test_report(self):
        """生成测试报告"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 CUDA测试报告")
        self.logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            self.logger.info(f"{test_name}: {status}")
        
        self.logger.info("-" * 60)
        self.logger.info(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            self.logger.info("🎉 所有CUDA测试通过！GPU加速训练系统准备就绪")
        else:
            self.logger.warning("⚠️ 部分CUDA测试失败，请检查GPU环境和驱动")


def main():
    """主函数"""
    print("🧪 CUDA训练系统测试工具")
    print("=" * 60)
    
    # 创建测试器
    tester = CUDATrainingTester()
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 返回结果
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    
    print(f"\n🏁 CUDA测试完成，退出码: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
