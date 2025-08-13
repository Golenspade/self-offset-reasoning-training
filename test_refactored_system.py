"""
文件名: test_refactored_system.py
重构后突破性训练系统的测试脚本
验证所有修复和改进是否正常工作
"""
import sys
import os
import json
import tempfile
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from breakthrough_training_system_refactored import (
    BreakthroughTrainingSystem, 
    create_breakthrough_config,
    ExperienceReplayBuffer,
    AdaptiveLearningRateScheduler
)


class RefactoredSystemTester:
    """重构后系统测试器"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {}
    
    def _setup_logging(self):
        """设置测试日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def test_config_system(self) -> bool:
        """测试配置系统"""
        self.logger.info("🔧 测试配置系统...")
        
        try:
            # 测试配置创建
            config = create_breakthrough_config()
            
            # 验证嵌套配置结构
            assert 'model' in config, "配置应包含model部分"
            assert 'training' in config, "配置应包含training部分"
            assert 'precision' in config, "配置应包含precision部分"
            assert 'replay' in config, "配置应包含replay部分"
            
            # 验证配置值
            assert config['model']['hidden_size'] > 0, "隐藏层大小应大于0"
            assert config['training']['initial_lr'] > 0, "学习率应大于0"
            assert config['precision']['lr_patience'] > 0, "学习率耐心值应大于0"
            
            self.logger.info("✅ 配置系统测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 配置系统测试失败: {e}")
            return False
    
    def test_model_interface(self) -> bool:
        """测试模型接口改进"""
        self.logger.info("🤖 测试模型接口...")
        
        try:
            from src.logic_transformer.models.base_model import ImprovedSimpleModel
            from src.logic_transformer.data_utils import Tokenizer
            
            tokenizer = Tokenizer()
            model = ImprovedSimpleModel(
                vocab_size=tokenizer.vocab_size,
                hidden_size=64,
                max_length=50,
                learning_rate=0.001
            )
            
            # 测试权重管理接口
            weights = model.get_weights()
            assert isinstance(weights, dict), "get_weights应返回字典"
            assert 'embedding' in weights, "权重应包含embedding"
            assert 'encoder_weights' in weights, "权重应包含encoder_weights"
            
            # 测试权重设置
            original_weights = model.get_weights()
            model.set_weights(original_weights)
            
            # 测试评估方法（不更新权重）
            input_tokens = [1, 2, 3]
            target_tokens = [2, 3, 4]
            
            # 获取评估前的权重
            weights_before = model.get_weights()
            
            # 执行评估步骤
            loss = model.evaluate_step(input_tokens, target_tokens, tokenizer)
            
            # 获取评估后的权重
            weights_after = model.get_weights()
            
            # 验证权重没有改变（评估不应更新权重）
            import numpy as np
            for key in weights_before:
                assert np.array_equal(weights_before[key], weights_after[key]), \
                    f"评估步骤不应改变权重: {key}"
            
            assert isinstance(loss, (int, float)), "损失应为数值"
            assert loss >= 0, "损失应为非负数"
            
            self.logger.info("✅ 模型接口测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型接口测试失败: {e}")
            return False
    
    def test_experience_replay(self) -> bool:
        """测试经验回放改进"""
        self.logger.info("💾 测试经验回放...")
        
        try:
            # 创建经验回放缓冲区
            buffer = ExperienceReplayBuffer(capacity=100)
            
            # 测试样本添加
            samples = [
                {'input': 'p -> q', 'target': '~q -> ~p', 'complexity': 'simple'},
                {'input': 'r -> s', 'target': '~s -> ~r', 'complexity': 'medium'},
                {'input': 'a -> b', 'target': '~b -> ~a', 'complexity': 'complex'}
            ]
            
            buffer.push_batch(samples)
            assert len(buffer) == 3, "缓冲区应包含3个样本"
            
            # 测试采样
            sampled = buffer.sample(2)
            assert len(sampled) == 2, "应采样2个样本"
            
            # 测试利用率
            utilization = buffer.utilization()
            assert 0 <= utilization <= 1, "利用率应在0-1之间"
            assert utilization == 0.03, f"利用率应为0.03，实际为{utilization}"
            
            self.logger.info("✅ 经验回放测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 经验回放测试失败: {e}")
            return False
    
    def test_learning_rate_scheduler(self) -> bool:
        """测试学习率调度器"""
        self.logger.info("📈 测试学习率调度器...")
        
        try:
            scheduler = AdaptiveLearningRateScheduler(
                initial_lr=0.001,
                patience=2,
                factor=0.5,
                min_lr=1e-6
            )
            
            # 测试初始状态
            assert scheduler.current_lr == 0.001, "初始学习率应为0.001"
            assert scheduler.best_loss == float('inf'), "初始最佳损失应为无穷大"
            
            # 测试学习率调整
            # 第一次：损失改善，不调整
            adjusted = scheduler.step(0.5)
            assert not adjusted, "损失改善时不应调整学习率"
            assert scheduler.current_lr == 0.001, "学习率不应改变"
            
            # 连续几次损失不改善，应该调整学习率
            adjusted1 = scheduler.step(0.6)  # 损失变差
            adjusted2 = scheduler.step(0.7)  # 损失继续变差
            adjusted3 = scheduler.step(0.8)  # 达到patience，应该调整

            # 检查是否在某个步骤进行了调整
            assert adjusted1 or adjusted2 or adjusted3, "达到patience时应调整学习率"
            assert scheduler.current_lr == 0.0005, f"学习率应调整为0.0005，实际为{scheduler.current_lr}"
            
            self.logger.info("✅ 学习率调度器测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 学习率调度器测试失败: {e}")
            return False
    
    def test_training_system_integration(self) -> bool:
        """测试训练系统集成"""
        self.logger.info("🚀 测试训练系统集成...")
        
        try:
            # 创建配置
            config = create_breakthrough_config()
            config['model']['hidden_size'] = 32  # 使用小模型加快测试
            
            # 创建训练系统
            trainer = BreakthroughTrainingSystem(config)
            
            # 验证系统初始化
            assert trainer.model is not None, "模型应被初始化"
            assert trainer.target_model is not None, "目标模型应被初始化"
            assert trainer.lr_scheduler is not None, "学习率调度器应被初始化"
            assert trainer.replay_buffer is not None, "经验回放缓冲区应被初始化"
            
            # 创建测试数据
            test_data = [
                {
                    'noisy_prop': 'p -> q',
                    'target_contrapositive': '~q -> ~p',
                    'complexity': 'simple'
                },
                {
                    'noisy_prop': 'r -> s',
                    'target_contrapositive': '~s -> ~r',
                    'complexity': 'simple'
                }
            ] * 10  # 20个样本
            
            # 测试单步训练
            loss, clipped = trainer.train_step(test_data[0])
            assert isinstance(loss, (int, float)), "训练步骤应返回数值损失"
            assert isinstance(clipped, bool), "训练步骤应返回布尔梯度裁剪标志"
            
            # 测试单步评估
            eval_loss = trainer.evaluate_step(test_data[0])
            assert isinstance(eval_loss, (int, float)), "评估步骤应返回数值损失"
            
            # 测试数据准备（课程学习）
            new_data, replay_data = trainer.prepare_training_data(test_data, epoch=0)
            assert isinstance(new_data, list), "新数据应为列表"
            assert isinstance(replay_data, list), "回放数据应为列表"
            
            # 测试验证评估
            val_loss = trainer.evaluate_validation(test_data[:5])
            assert isinstance(val_loss, (int, float)), "验证损失应为数值"
            assert val_loss >= 0, "验证损失应为非负数"
            
            self.logger.info("✅ 训练系统集成测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 训练系统集成测试失败: {e}")
            return False
    
    def test_bug_fixes(self) -> bool:
        """测试关键Bug修复"""
        self.logger.info("🐛 测试Bug修复...")
        
        try:
            # 创建配置和训练系统
            config = create_breakthrough_config()
            config['model']['hidden_size'] = 32
            trainer = BreakthroughTrainingSystem(config)
            
            # 创建测试数据
            test_sample = {
                'noisy_prop': 'p -> q',
                'target_contrapositive': '~q -> ~p',
                'complexity': 'simple'
            }
            
            # 测试Bug修复1: 验证集不应训练模型
            # 获取模型权重
            weights_before_eval = trainer.model.get_weights()
            
            # 执行验证评估
            val_loss = trainer.evaluate_validation([test_sample])
            
            # 获取评估后的权重
            weights_after_eval = trainer.model.get_weights()
            
            # 验证权重没有改变
            import numpy as np
            weights_changed = False
            for key in weights_before_eval:
                if not np.array_equal(weights_before_eval[key], weights_after_eval[key]):
                    weights_changed = True
                    break
            
            assert not weights_changed, "验证评估不应改变模型权重"
            
            # 测试Bug修复2: 目标网络安全更新
            original_target_weights = trainer.target_model.get_weights()
            trainer.target_model.soft_update_from(trainer.model, tau=0.1)
            updated_target_weights = trainer.target_model.get_weights()
            
            # 验证目标网络权重确实更新了
            weights_updated = False
            for key in original_target_weights:
                if not np.array_equal(original_target_weights[key], updated_target_weights[key]):
                    weights_updated = True
                    break
            
            assert weights_updated, "目标网络权重应该被更新"
            
            self.logger.info("✅ Bug修复测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bug修复测试失败: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """运行所有测试"""
        self.logger.info("🧪 开始重构后系统完整测试")
        self.logger.info("=" * 60)
        
        tests = [
            ('配置系统', self.test_config_system),
            ('模型接口', self.test_model_interface),
            ('经验回放', self.test_experience_replay),
            ('学习率调度器', self.test_learning_rate_scheduler),
            ('训练系统集成', self.test_training_system_integration),
            ('Bug修复验证', self.test_bug_fixes)
        ]
        
        for test_name, test_func in tests:
            self.logger.info(f"\n🔍 测试: {test_name}")
            self.test_results[test_name] = test_func()
        
        # 生成测试报告
        self._generate_test_report()
        
        return self.test_results
    
    def _generate_test_report(self):
        """生成测试报告"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 重构后系统测试报告")
        self.logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            self.logger.info(f"{test_name}: {status}")
        
        self.logger.info("-" * 60)
        self.logger.info(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            self.logger.info("🎉 所有测试通过！重构后系统质量优秀")
            self.logger.info("✨ 关键改进验证:")
            self.logger.info("  - ✅ 验证集训练Bug已修复")
            self.logger.info("  - ✅ 模型权重管理安全可靠")
            self.logger.info("  - ✅ 配置系统统一一致")
            self.logger.info("  - ✅ 经验回放机制改进")
            self.logger.info("  - ✅ 异常处理增强")
        else:
            self.logger.warning("⚠️ 部分测试失败，需要进一步检查")


def main():
    """主函数"""
    print("🧪 重构后突破性训练系统测试工具")
    print("=" * 60)
    
    # 创建测试器
    tester = RefactoredSystemTester()
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 返回结果
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    
    print(f"\n🏁 测试完成，退出码: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
