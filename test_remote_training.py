"""
文件名: test_remote_training.py
远程训练系统测试脚本
验证所有组件的功能和集成
"""
import os
import sys
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List

# 添加项目路径
sys.path.append('.')
sys.path.append('src')

from remote_training_config import RemoteTrainingConfig
from sync_data_to_remote import DataSyncManager
from remote_training_main import RemoteTrainingManager


class RemoteTrainingTester:
    """远程训练系统测试器"""
    
    def __init__(self):
        self.test_dir = None
        self.config = None
        self.logger = self._setup_logging()
        self.test_results = {}
    
    def _setup_logging(self):
        """设置测试日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def setup_test_environment(self):
        """设置测试环境"""
        self.logger.info("🧪 设置测试环境...")
        
        # 创建临时测试目录
        self.test_dir = tempfile.mkdtemp(prefix='remote_training_test_')
        self.logger.info(f"📁 测试目录: {self.test_dir}")
        
        # 设置环境变量
        os.environ.update({
            'REMOTE_DATA_PATH': os.path.join(self.test_dir, 'data'),
            'REMOTE_MODEL_PATH': os.path.join(self.test_dir, 'models'),
            'REMOTE_OUTPUT_PATH': os.path.join(self.test_dir, 'outputs'),
            'REMOTE_CHECKPOINT_PATH': os.path.join(self.test_dir, 'checkpoints'),
            'EPOCHS': '3',
            'BATCH_SIZE': '4',
            'LEARNING_RATE': '0.01',
            'CLOUD_PROVIDER': 'local',
            'DEBUG_MODE': 'true',
            'DRY_RUN': 'true'
        })
        
        # 创建配置
        self.config = RemoteTrainingConfig()
        self.config.create_directories()
        
        self.logger.info("✅ 测试环境设置完成")
    
    def test_config_system(self) -> bool:
        """测试配置系统"""
        self.logger.info("🔧 测试配置系统...")
        
        try:
            # 测试配置创建
            config = RemoteTrainingConfig()
            
            # 测试配置验证
            assert config.batch_size > 0, "批次大小应该大于0"
            assert config.epochs > 0, "训练轮次应该大于0"
            assert config.learning_rate > 0, "学习率应该大于0"
            
            # 测试路径获取
            paths = config.get_full_paths()
            assert 'data_path' in paths, "应该包含数据路径"
            assert 'model_path' in paths, "应该包含模型路径"
            
            # 测试配置保存
            config_file = os.path.join(self.test_dir, 'test_config.json')
            config.save_config(config_file)
            assert os.path.exists(config_file), "配置文件应该被保存"
            
            self.test_results['config_system'] = True
            self.logger.info("✅ 配置系统测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 配置系统测试失败: {e}")
            self.test_results['config_system'] = False
            return False
    
    def test_data_sync(self) -> bool:
        """测试数据同步系统"""
        self.logger.info("📊 测试数据同步系统...")
        
        try:
            # 创建测试数据
            test_data = [
                {
                    "original_prop": "p -> q",
                    "noisy_prop": "(~p | q)",
                    "target_contrapositive": "~q -> ~p",
                    "complexity": "simple"
                }
            ] * 10
            
            # 保存测试数据
            train_file = os.path.join(self.test_dir, 'train_test.json')
            val_file = os.path.join(self.test_dir, 'val_test.json')
            
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f)
            
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(test_data[:5], f)
            
            # 测试数据同步管理器
            sync_manager = DataSyncManager(self.config)
            
            # 测试本地文件复制（模拟云存储）
            success = sync_manager.storage_manager._copy_local_file(
                train_file, 
                os.path.join(self.config.remote_data_path, 'train_data.json')
            )
            
            assert success, "数据复制应该成功"
            assert os.path.exists(os.path.join(self.config.remote_data_path, 'train_data.json')), "目标文件应该存在"
            
            self.test_results['data_sync'] = True
            self.logger.info("✅ 数据同步系统测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据同步系统测试失败: {e}")
            self.test_results['data_sync'] = False
            return False
    
    def test_training_system(self) -> bool:
        """测试训练系统"""
        self.logger.info("🚀 测试训练系统...")
        
        try:
            # 准备训练数据
            train_data = [
                {
                    "original_prop": "p -> q",
                    "noisy_prop": "(~p | q)",
                    "target_contrapositive": "~q -> ~p",
                    "complexity": "simple"
                }
            ] * 20
            
            val_data = train_data[:10]
            
            # 保存数据文件
            train_file = os.path.join(self.config.remote_data_path, 'train_data.json')
            val_file = os.path.join(self.config.remote_data_path, 'val_data.json')
            
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f)
            
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(val_data, f)
            
            # 创建训练管理器
            manager = RemoteTrainingManager()
            manager.config = self.config
            manager.setup_logging()
            
            # 测试环境验证
            manager.validate_environment()
            
            # 测试训练系统初始化
            manager.initialize_training_system()
            assert manager.training_system is not None, "训练系统应该被初始化"
            
            # 测试模拟训练
            manager.config.dry_run = True
            manager.run_training()
            
            self.test_results['training_system'] = True
            self.logger.info("✅ 训练系统测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 训练系统测试失败: {e}")
            self.test_results['training_system'] = False
            return False
    
    def test_checkpoint_system(self) -> bool:
        """测试检查点系统"""
        self.logger.info("💾 测试检查点系统...")
        
        try:
            from breakthrough_training_system import BreakthroughTrainingSystem
            
            # 创建训练系统
            config = {
                'hidden_size': 64,
                'max_length': 20,
                'initial_lr': 0.01,
                'batch_size': 4
            }
            
            training_system = BreakthroughTrainingSystem(config)
            
            # 测试检查点保存
            checkpoint_info = training_system.save_remote_checkpoint(0, self.config)
            
            assert 'path' in checkpoint_info, "检查点信息应该包含路径"
            assert 'epoch' in checkpoint_info, "检查点信息应该包含轮次"
            assert os.path.exists(checkpoint_info['path']), "检查点文件应该存在"
            
            # 测试检查点加载
            state_path = checkpoint_info.get('state_path')
            if state_path:
                result = training_system.load_remote_checkpoint(
                    checkpoint_info['path'], 
                    state_path
                )
                assert result is not None, "检查点加载应该成功"
            
            self.test_results['checkpoint_system'] = True
            self.logger.info("✅ 检查点系统测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 检查点系统测试失败: {e}")
            self.test_results['checkpoint_system'] = False
            return False
    
    def test_error_handling(self) -> bool:
        """测试错误处理"""
        self.logger.info("🚨 测试错误处理...")
        
        try:
            # 测试无效配置
            try:
                os.environ['BATCH_SIZE'] = '0'  # 无效值
                invalid_config = RemoteTrainingConfig()
                assert False, "应该抛出配置验证错误"
            except ValueError:
                pass  # 预期的错误
            finally:
                os.environ['BATCH_SIZE'] = '4'  # 恢复有效值
            
            # 测试文件不存在
            sync_manager = DataSyncManager(self.config)
            result = sync_manager.storage_manager.upload_file(
                'nonexistent_file.txt', 
                'remote_path.txt'
            )
            assert not result, "上传不存在的文件应该失败"
            
            self.test_results['error_handling'] = True
            self.logger.info("✅ 错误处理测试通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 错误处理测试失败: {e}")
            self.test_results['error_handling'] = False
            return False
    
    def cleanup_test_environment(self):
        """清理测试环境"""
        self.logger.info("🧹 清理测试环境...")
        
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            self.logger.info(f"🗑️ 删除测试目录: {self.test_dir}")
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        self.logger.info("🧪 开始远程训练系统完整测试")
        self.logger.info("=" * 60)
        
        try:
            # 设置测试环境
            self.setup_test_environment()
            
            # 运行测试
            tests = [
                ('配置系统', self.test_config_system),
                ('数据同步', self.test_data_sync),
                ('训练系统', self.test_training_system),
                ('检查点系统', self.test_checkpoint_system),
                ('错误处理', self.test_error_handling)
            ]
            
            for test_name, test_func in tests:
                self.logger.info(f"\n🔍 测试: {test_name}")
                test_func()
            
            # 生成测试报告
            self._generate_test_report()
            
        finally:
            # 清理环境
            self.cleanup_test_environment()
        
        return self.test_results
    
    def _generate_test_report(self):
        """生成测试报告"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 测试报告")
        self.logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            self.logger.info(f"{test_name}: {status}")
        
        self.logger.info("-" * 60)
        self.logger.info(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            self.logger.info("🎉 所有测试通过！远程训练系统准备就绪")
        else:
            self.logger.warning("⚠️ 部分测试失败，请检查相关组件")


def main():
    """主函数"""
    print("🧪 远程训练系统测试工具")
    print("=" * 60)
    
    # 创建测试器
    tester = RemoteTrainingTester()
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 返回结果
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    
    print(f"\n🏁 测试完成，退出码: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
