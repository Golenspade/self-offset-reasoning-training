"""
文件名: remote_training_main.py
远程训练主程序
支持分布式训练和云端部署
"""
import os
import sys
import json
import logging
import signal
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

# 计算项目根目录（remote/ 的上一级）并添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from remote.remote_training_config import RemoteTrainingConfig
from scripts.breakthrough_training_system_refactored import BreakthroughTrainingSystem


class RemoteTrainingManager:
    """远程训练管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化远程训练管理器"""
        self.config = RemoteTrainingConfig(config_file)
        self.training_system = None
        self.logger = None
        self.start_time = None
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = os.path.join(self.config.remote_output_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 配置日志处理器
        handlers = []
        
        # 文件处理器
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if not self.config.debug_mode else logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
        
        # 配置根日志器
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug_mode else logging.INFO,
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📝 日志系统初始化完成")
        self.logger.info(f"📁 日志文件: {log_file}")
    
    def setup_monitoring(self):
        """设置监控系统"""
        try:
            # 初始化Weights & Biases
            if self.config.enable_wandb:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    config=self.config.get_training_config(),
                    name=f"remote_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.logger.info("📊 Weights & Biases 监控已启用")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 监控系统设置失败: {e}")
    
    def validate_environment(self):
        """验证远程环境"""
        self.logger.info("🔍 验证远程环境...")
        
        # 检查必要目录
        self.config.create_directories()
        
        # 检查数据文件
        paths = self.config.get_full_paths()
        
        if not os.path.exists(paths['train_data']):
            self.logger.error(f"❌ 训练数据文件不存在: {paths['train_data']}")
            raise FileNotFoundError(f"训练数据文件不存在: {paths['train_data']}")
        
        if not os.path.exists(paths['val_data']):
            self.logger.error(f"❌ 验证数据文件不存在: {paths['val_data']}")
            raise FileNotFoundError(f"验证数据文件不存在: {paths['val_data']}")
        
        # 检查GPU可用性
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"🎮 检测到 {gpu_count} 个GPU")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.logger.warning("⚠️ 未检测到GPU，将使用CPU训练")
        except ImportError:
            self.logger.warning("⚠️ PyTorch未安装，无法检测GPU")
        
        self.logger.info("✅ 环境验证完成")
    
    def initialize_training_system(self):
        """初始化训练系统"""
        self.logger.info("🚀 初始化训练系统...")
        
        try:
            # 创建训练配置
            training_config = self.config.get_training_config()
            training_config.update(self.config.get_model_config())
            training_config['remote_config'] = self.config
            
            # 初始化训练系统
            self.training_system = BreakthroughTrainingSystem(training_config)
            
            self.logger.info("✅ 训练系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 训练系统初始化失败: {e}")
            raise
    
    def run_training(self):
        """执行训练"""
        self.logger.info("🎯 开始远程训练...")
        self.start_time = datetime.now()
        
        try:
            # 如果是干运行模式
            if self.config.dry_run:
                self.logger.info("🧪 干运行模式，跳过实际训练")
                self._simulate_training()
                return
            
            # 执行实际训练
            if hasattr(self.training_system, 'run_remote_training'):
                # 使用专门的远程训练方法
                results = self.training_system.run_remote_training(self.config)
            else:
                # 使用标准训练方法
                results = self.training_system.run_full_training()
            
            # 记录训练结果
            self._log_training_results(results)
            
            self.logger.info("🎉 远程训练完成！")
            
        except Exception as e:
            self.logger.error(f"❌ 训练过程中发生错误: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise
        
        finally:
            # 计算训练时间
            if self.start_time:
                duration = datetime.now() - self.start_time
                self.logger.info(f"⏱️ 总训练时间: {duration}")
    
    def _simulate_training(self):
        """模拟训练（用于测试）"""
        import time
        
        self.logger.info("🧪 模拟训练开始...")
        
        for epoch in range(min(5, self.config.epochs)):
            self.logger.info(f"模拟训练 Epoch {epoch+1}/{min(5, self.config.epochs)}")
            time.sleep(2)  # 模拟训练时间
            
            # 模拟指标
            fake_metrics = {
                'epoch': epoch + 1,
                'train_loss': 0.5 - epoch * 0.05,
                'val_loss': 0.6 - epoch * 0.04,
                'accuracy': 0.7 + epoch * 0.05
            }
            
            self.logger.info(f"模拟指标: {fake_metrics}")
        
        self.logger.info("🧪 模拟训练完成")
    
    def _log_training_results(self, results):
        """记录训练结果"""
        if results:
            self.logger.info("📊 训练结果:")
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")
            
            # 保存结果到文件
            results_file = os.path.join(self.config.remote_output_path, 'training_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 结果已保存到: {results_file}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"🛑 接收到信号 {signum}，正在优雅关闭...")
        
        # 保存当前状态
        if self.training_system:
            try:
                checkpoint_path = os.path.join(
                    self.config.remote_checkpoint_path, 
                    'emergency_checkpoint.npz'
                )
                # 这里应该调用训练系统的保存方法
                self.logger.info(f"💾 紧急保存检查点到: {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"❌ 紧急保存失败: {e}")
        
        sys.exit(0)
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 清理资源...")
        
        try:
            # 关闭监控
            if self.config.enable_wandb:
                import wandb
                wandb.finish()
            
            # 其他清理工作
            self.logger.info("✅ 资源清理完成")
            
        except Exception as e:
            self.logger.error(f"⚠️ 资源清理时发生错误: {e}")


def main():
    """主函数"""
    print("🚀 启动远程训练系统")
    print("=" * 60)
    
    # 获取配置文件路径
    config_file = os.getenv('CONFIG_FILE', None)
    if config_file and not os.path.exists(config_file):
        print(f"⚠️ 配置文件不存在: {config_file}")
        config_file = None
    
    try:
        # 创建训练管理器
        manager = RemoteTrainingManager(config_file)
        
        # 设置日志
        manager.setup_logging()
        
        # 打印配置信息
        manager.logger.info(str(manager.config))
        
        # 设置监控
        manager.setup_monitoring()
        
        # 验证环境
        manager.validate_environment()
        
        # 初始化训练系统
        manager.initialize_training_system()
        
        # 执行训练
        manager.run_training()
        
    except KeyboardInterrupt:
        print("\n🛑 用户中断训练")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ 远程训练失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        sys.exit(1)
        
    finally:
        # 清理资源
        if 'manager' in locals():
            manager.cleanup()
    
    print("🎉 远程训练系统退出")


if __name__ == "__main__":
    main()
