"""
文件名: sync_data_to_remote.py
数据同步到远程存储
支持多种云存储平台的数据上传和下载
"""
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import sys

# 计算项目根目录（remote/ 的上一级），确保可以导入 src/ 和 scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 云存储客户端导入
try:
    import boto3  # AWS S3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs  # Google Cloud Storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient  # Azure Blob Storage
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import oss2  # 阿里云OSS
    ALIYUN_AVAILABLE = True
except ImportError:
    ALIYUN_AVAILABLE = False

from remote.remote_training_config import RemoteTrainingConfig


class CloudStorageManager:
    """云存储管理器"""
    
    def __init__(self, config: RemoteTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        # 初始化云存储客户端
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化云存储客户端"""
        provider = self.config.cloud_provider.lower()
        
        try:
            if provider == 'aws' and AWS_AVAILABLE:
                self._init_aws_client()
            elif provider == 'gcp' and GCP_AVAILABLE:
                self._init_gcp_client()
            elif provider == 'azure' and AZURE_AVAILABLE:
                self._init_azure_client()
            elif provider == 'aliyun' and ALIYUN_AVAILABLE:
                self._init_aliyun_client()
            elif provider == 'local':
                self.logger.info("使用本地存储")
            else:
                self.logger.warning(f"不支持的云存储提供商: {provider}")
                
        except Exception as e:
            self.logger.error(f"云存储客户端初始化失败: {e}")
    
    def _init_aws_client(self):
        """初始化AWS S3客户端"""
        self.client = boto3.client(
            's3',
            aws_access_key_id=self.config.cloud_access_key,
            aws_secret_access_key=self.config.cloud_secret_key,
            region_name=self.config.cloud_region
        )
        self.logger.info("AWS S3客户端初始化成功")
    
    def _init_gcp_client(self):
        """初始化Google Cloud Storage客户端"""
        self.client = gcs.Client()
        self.logger.info("Google Cloud Storage客户端初始化成功")
    
    def _init_azure_client(self):
        """初始化Azure Blob Storage客户端"""
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.config.cloud_access_key};AccountKey={self.config.cloud_secret_key};EndpointSuffix=core.windows.net"
        self.client = BlobServiceClient.from_connection_string(connection_string)
        self.logger.info("Azure Blob Storage客户端初始化成功")
    
    def _init_aliyun_client(self):
        """初始化阿里云OSS客户端"""
        auth = oss2.Auth(self.config.cloud_access_key, self.config.cloud_secret_key)
        endpoint = f"https://oss-{self.config.cloud_region}.aliyuncs.com"
        self.client = oss2.Bucket(auth, endpoint, self.config.cloud_bucket)
        self.logger.info("阿里云OSS客户端初始化成功")
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """上传文件到云存储"""
        if self.config.cloud_provider == 'local':
            return self._copy_local_file(local_path, remote_path)
        
        try:
            provider = self.config.cloud_provider.lower()
            
            if provider == 'aws':
                return self._upload_to_s3(local_path, remote_path)
            elif provider == 'gcp':
                return self._upload_to_gcs(local_path, remote_path)
            elif provider == 'azure':
                return self._upload_to_azure(local_path, remote_path)
            elif provider == 'aliyun':
                return self._upload_to_oss(local_path, remote_path)
            
        except Exception as e:
            self.logger.error(f"文件上传失败: {e}")
            return False
        
        return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """从云存储下载文件"""
        if self.config.cloud_provider == 'local':
            return self._copy_local_file(remote_path, local_path)
        
        try:
            provider = self.config.cloud_provider.lower()
            
            if provider == 'aws':
                return self._download_from_s3(remote_path, local_path)
            elif provider == 'gcp':
                return self._download_from_gcs(remote_path, local_path)
            elif provider == 'azure':
                return self._download_from_azure(remote_path, local_path)
            elif provider == 'aliyun':
                return self._download_from_oss(remote_path, local_path)
            
        except Exception as e:
            self.logger.error(f"文件下载失败: {e}")
            return False
        
        return False
    
    def _upload_to_s3(self, local_path: str, remote_path: str) -> bool:
        """上传到AWS S3"""
        try:
            self.client.upload_file(local_path, self.config.cloud_bucket, remote_path)
            self.logger.info(f"文件上传到S3成功: {remote_path}")
            return True
        except ClientError as e:
            self.logger.error(f"S3上传失败: {e}")
            return False
    
    def _upload_to_gcs(self, local_path: str, remote_path: str) -> bool:
        """上传到Google Cloud Storage"""
        try:
            bucket = self.client.bucket(self.config.cloud_bucket)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            self.logger.info(f"文件上传到GCS成功: {remote_path}")
            return True
        except Exception as e:
            self.logger.error(f"GCS上传失败: {e}")
            return False
    
    def _upload_to_azure(self, local_path: str, remote_path: str) -> bool:
        """上传到Azure Blob Storage"""
        try:
            blob_client = self.client.get_blob_client(
                container=self.config.cloud_bucket,
                blob=remote_path
            )
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            self.logger.info(f"文件上传到Azure成功: {remote_path}")
            return True
        except Exception as e:
            self.logger.error(f"Azure上传失败: {e}")
            return False
    
    def _upload_to_oss(self, local_path: str, remote_path: str) -> bool:
        """上传到阿里云OSS"""
        try:
            self.client.put_object_from_file(remote_path, local_path)
            self.logger.info(f"文件上传到OSS成功: {remote_path}")
            return True
        except Exception as e:
            self.logger.error(f"OSS上传失败: {e}")
            return False
    
    def _copy_local_file(self, src: str, dst: str) -> bool:
        """本地文件复制"""
        try:
            import shutil
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            self.logger.info(f"本地文件复制成功: {src} -> {dst}")
            return True
        except Exception as e:
            self.logger.error(f"本地文件复制失败: {e}")
            return False


class DataSyncManager:
    """数据同步管理器"""
    
    def __init__(self, config: RemoteTrainingConfig):
        self.config = config
        self.storage_manager = CloudStorageManager(config)
        self.logger = logging.getLogger(__name__)
    
    def sync_training_data(self, force_regenerate: bool = False) -> bool:
        """同步训练数据到云端"""
        self.logger.info("🔄 开始同步训练数据...")
        
        try:
            # 检查本地数据是否存在
            local_train_data = "data/train_level_3_鲁棒版.json"
            local_val_data = "data/val_level_3_鲁棒版.json"
            
            if force_regenerate or not os.path.exists(local_train_data) or not os.path.exists(local_val_data):
                self.logger.info("📊 生成新的训练数据...")
                self._generate_training_data()
            
            # 上传训练数据
            remote_train_path = f"data/{self.config.train_data_file}"
            remote_val_path = f"data/{self.config.val_data_file}"
            
            success = True
            success &= self.storage_manager.upload_file(local_train_data, remote_train_path)
            success &= self.storage_manager.upload_file(local_val_data, remote_val_path)
            
            if success:
                self.logger.info("✅ 训练数据同步成功")
                return True
            else:
                self.logger.error("❌ 训练数据同步失败")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 数据同步过程中发生错误: {e}")
            return False
    
    def _generate_training_data(self):
        """生成训练数据"""
        try:
            # 导入当前推荐的数据生成函数（位于 scripts/ 目录）
            from scripts.generate_robust_dataset import generate_robust_dataset

            # 直接生成复杂度为 "complex" 的鲁棒数据集，对应原来的 L3 级别
            train_data = generate_robust_dataset(size=10000, complexity_level="complex")
            val_data = generate_robust_dataset(size=2000, complexity_level="complex")

            # 保存数据
            os.makedirs("data", exist_ok=True)

            with open("data/train_level_3_鲁棒版.json", "w", encoding="utf-8") as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)

            with open("data/val_level_3_鲁棒版.json", "w", encoding="utf-8") as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"✅ 生成训练数据: {len(train_data)} 训练样本, {len(val_data)} 验证样本"
            )
            
        except Exception as e:
            self.logger.error(f"❌ 数据生成失败: {e}")
            raise
    
    def download_training_data(self) -> bool:
        """从云端下载训练数据"""
        self.logger.info("⬇️ 从云端下载训练数据...")
        
        try:
            # 创建本地数据目录
            os.makedirs(self.config.remote_data_path, exist_ok=True)
            
            # 下载文件
            remote_train_path = f"data/{self.config.train_data_file}"
            remote_val_path = f"data/{self.config.val_data_file}"
            
            local_train_path = os.path.join(self.config.remote_data_path, self.config.train_data_file)
            local_val_path = os.path.join(self.config.remote_data_path, self.config.val_data_file)
            
            success = True
            success &= self.storage_manager.download_file(remote_train_path, local_train_path)
            success &= self.storage_manager.download_file(remote_val_path, local_val_path)
            
            if success:
                self.logger.info("✅ 训练数据下载成功")
                return True
            else:
                self.logger.error("❌ 训练数据下载失败")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 数据下载过程中发生错误: {e}")
            return False
    
    def sync_model_checkpoints(self, checkpoint_dir: str) -> bool:
        """同步模型检查点"""
        self.logger.info("💾 同步模型检查点...")
        
        try:
            if not os.path.exists(checkpoint_dir):
                self.logger.warning(f"检查点目录不存在: {checkpoint_dir}")
                return True
            
            success_count = 0
            total_count = 0
            
            for file_name in os.listdir(checkpoint_dir):
                if file_name.endswith(('.npz', '.json')):
                    local_path = os.path.join(checkpoint_dir, file_name)
                    remote_path = f"checkpoints/{file_name}"
                    
                    total_count += 1
                    if self.storage_manager.upload_file(local_path, remote_path):
                        success_count += 1
            
            self.logger.info(f"📊 检查点同步完成: {success_count}/{total_count}")
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"❌ 检查点同步失败: {e}")
            return False


def main():
    """主函数 - 数据同步工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据同步工具")
    parser.add_argument('--action', choices=['upload', 'download', 'sync-checkpoints'], 
                       required=True, help="操作类型")
    parser.add_argument('--config', help="配置文件路径")
    parser.add_argument('--force-regenerate', action='store_true', 
                       help="强制重新生成数据")
    parser.add_argument('--checkpoint-dir', default='outputs/checkpoints',
                       help="检查点目录")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = RemoteTrainingConfig(args.config)
    
    # 创建数据同步管理器
    sync_manager = DataSyncManager(config)
    
    # 执行操作
    if args.action == 'upload':
        success = sync_manager.sync_training_data(args.force_regenerate)
    elif args.action == 'download':
        success = sync_manager.download_training_data()
    elif args.action == 'sync-checkpoints':
        success = sync_manager.sync_model_checkpoints(args.checkpoint_dir)
    
    if success:
        print("✅ 操作成功完成")
    else:
        print("❌ 操作失败")
        exit(1)


if __name__ == "__main__":
    main()
