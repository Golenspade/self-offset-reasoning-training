"""
文件名: remote_training_config.py
远程训练配置文件
支持环境变量配置和分布式训练参数
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class RemoteTrainingConfig:
    """远程训练配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化远程训练配置
        
        Args:
            config_file: 可选的配置文件路径，用于覆盖环境变量
        """
        # 加载基础配置
        self._load_base_config()
        
        # 如果提供了配置文件，加载并覆盖
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # 验证配置
        self._validate_config()
    
    def _load_base_config(self):
        """从环境变量加载基础配置"""
        
        # ==================== 远程环境路径配置 ====================
        self.remote_data_path = os.getenv('REMOTE_DATA_PATH', '/data/logic_training')
        self.remote_model_path = os.getenv('REMOTE_MODEL_PATH', '/models/logic_models')
        self.remote_output_path = os.getenv('REMOTE_OUTPUT_PATH', '/outputs/training_results')
        self.remote_checkpoint_path = os.getenv('REMOTE_CHECKPOINT_PATH', '/checkpoints')
        
        # ==================== 训练超参数配置 ====================
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        self.epochs = int(os.getenv('EPOCHS', '50'))
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.001'))
        self.weight_decay = float(os.getenv('WEIGHT_DECAY', '1e-5'))
        self.gradient_clip_norm = float(os.getenv('GRADIENT_CLIP_NORM', '1.0'))
        
        # ==================== 分布式训练配置 ====================
        self.use_distributed = os.getenv('USE_DISTRIBUTED', 'false').lower() == 'true'
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.rank = int(os.getenv('RANK', '0'))
        self.local_rank = int(os.getenv('LOCAL_RANK', '0'))
        self.master_addr = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '12355')
        
        # ==================== 检查点和日志配置 ====================
        self.checkpoint_frequency = int(os.getenv('CHECKPOINT_FREQ', '10'))
        self.log_frequency = int(os.getenv('LOG_FREQ', '100'))
        self.save_best_only = os.getenv('SAVE_BEST_ONLY', 'true').lower() == 'true'
        self.early_stopping_patience = int(os.getenv('EARLY_STOPPING_PATIENCE', '20'))
        
        # ==================== 数据配置 ====================
        self.train_data_file = os.getenv('TRAIN_DATA_FILE', 'train_data.json')
        self.val_data_file = os.getenv('VAL_DATA_FILE', 'val_data.json')
        self.data_workers = int(os.getenv('DATA_WORKERS', '4'))
        self.prefetch_factor = int(os.getenv('PREFETCH_FACTOR', '2'))
        
        # ==================== 模型配置 ====================
        self.model_type = os.getenv('MODEL_TYPE', 'breakthrough')  # breakthrough, simple, hybrid
        self.hidden_size = int(os.getenv('HIDDEN_SIZE', '256'))
        self.num_layers = int(os.getenv('NUM_LAYERS', '4'))
        self.dropout_rate = float(os.getenv('DROPOUT_RATE', '0.1'))
        
        # ==================== 云存储配置 ====================
        self.cloud_provider = os.getenv('CLOUD_PROVIDER', 'local')  # aws, gcp, azure, aliyun, local
        self.cloud_bucket = os.getenv('CLOUD_BUCKET', '')
        self.cloud_region = os.getenv('CLOUD_REGION', '')
        self.cloud_access_key = os.getenv('CLOUD_ACCESS_KEY', '')
        self.cloud_secret_key = os.getenv('CLOUD_SECRET_KEY', '')
        
        # ==================== 监控和通知配置 ====================
        self.enable_wandb = os.getenv('ENABLE_WANDB', 'false').lower() == 'true'
        self.wandb_project = os.getenv('WANDB_PROJECT', 'logic-training')
        self.wandb_entity = os.getenv('WANDB_ENTITY', '')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK', '')
        self.email_notifications = os.getenv('EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
        
        # ==================== 资源配置 ====================
        self.gpu_memory_limit = os.getenv('GPU_MEMORY_LIMIT', '')
        self.cpu_limit = int(os.getenv('CPU_LIMIT', '0'))  # 0表示不限制
        self.memory_limit = os.getenv('MEMORY_LIMIT', '')  # 如 "8Gi"
        
        # ==================== 调试和开发配置 ====================
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.profile_training = os.getenv('PROFILE_TRAINING', 'false').lower() == 'true'
        self.dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
    
    def _load_config_file(self, config_file: str):
        """从JSON配置文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}")
    
    def _validate_config(self):
        """验证配置的有效性"""
        errors = []
        
        # 验证必要路径
        if not self.remote_data_path:
            errors.append("remote_data_path 不能为空")
        
        # 验证训练参数
        if self.batch_size <= 0:
            errors.append("batch_size 必须大于0")
        
        if self.epochs <= 0:
            errors.append("epochs 必须大于0")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate 必须大于0")
        
        # 验证分布式配置
        if self.use_distributed:
            if self.world_size <= 1:
                errors.append("分布式训练时 world_size 必须大于1")
            
            if self.rank >= self.world_size:
                errors.append("rank 必须小于 world_size")
        
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
    
    def get_full_paths(self) -> Dict[str, str]:
        """获取完整的路径配置"""
        return {
            'data_path': self.remote_data_path,
            'model_path': self.remote_model_path,
            'output_path': self.remote_output_path,
            'checkpoint_path': self.remote_checkpoint_path,
            'train_data': os.path.join(self.remote_data_path, self.train_data_file),
            'val_data': os.path.join(self.remote_data_path, self.val_data_file)
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练相关配置"""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm,
            'checkpoint_frequency': self.checkpoint_frequency,
            'early_stopping_patience': self.early_stopping_patience
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型相关配置"""
        return {
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate
        }
    
    def get_distributed_config(self) -> Dict[str, Any]:
        """获取分布式训练配置"""
        return {
            'use_distributed': self.use_distributed,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'master_addr': self.master_addr,
            'master_port': self.master_port
        }
    
    def save_config(self, output_path: str):
        """保存当前配置到文件"""
        config_dict = {}
        
        # 获取所有配置属性
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                config_dict[attr_name] = getattr(self, attr_name)
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {output_path}")
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.remote_data_path,
            self.remote_model_path,
            self.remote_output_path,
            self.remote_checkpoint_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 创建目录: {directory}")
    
    def __str__(self) -> str:
        """配置信息的字符串表示"""
        config_info = [
            "🔧 远程训练配置信息:",
            f"  数据路径: {self.remote_data_path}",
            f"  模型路径: {self.remote_model_path}",
            f"  输出路径: {self.remote_output_path}",
            f"  批次大小: {self.batch_size}",
            f"  训练轮次: {self.epochs}",
            f"  学习率: {self.learning_rate}",
            f"  分布式训练: {'是' if self.use_distributed else '否'}",
            f"  云服务商: {self.cloud_provider}",
            f"  调试模式: {'是' if self.debug_mode else '否'}"
        ]
        
        return "\n".join(config_info)


def create_default_config_file(output_path: str = "configs/remote_training_config.json"):
    """创建默认的远程训练配置文件"""
    default_config = {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001,
        "model_type": "breakthrough",
        "hidden_size": 256,
        "num_layers": 4,
        "checkpoint_frequency": 10,
        "early_stopping_patience": 20,
        "cloud_provider": "local",
        "enable_wandb": False,
        "debug_mode": False
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 默认配置文件已创建: {output_path}")


if __name__ == "__main__":
    # 测试配置系统
    print("🧪 测试远程训练配置系统")
    
    # 创建默认配置文件
    create_default_config_file()
    
    # 创建配置实例
    config = RemoteTrainingConfig()
    print(config)
    
    # 创建必要目录
    config.create_directories()
    
    # 保存配置
    config.save_config("outputs/current_remote_config.json")
