"""
文件名: cuda_utils.py
CUDA工具模块
处理GPU设备检测、内存管理、性能优化等
"""
import os
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

# 尝试导入CUDA相关库
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch未安装，CUDA功能不可用")

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("nvidia-ml-py3未安装，GPU监控功能受限")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class CUDAManager:
    """CUDA设备和内存管理器"""
    
    def __init__(self, memory_fraction: float = 0.8, auto_optimize: bool = True):
        """
        初始化CUDA管理器
        
        Args:
            memory_fraction: GPU内存使用比例 (0.0-1.0)
            auto_optimize: 是否自动优化GPU设置
        """
        self.memory_fraction = memory_fraction
        self.auto_optimize = auto_optimize
        self.device = None
        self.device_properties = {}
        
        # 初始化NVML
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                logger.warning(f"NVML初始化失败: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
        
        # 获取最佳设备
        self.device = self.get_best_device()
        
        # 自动优化
        if self.auto_optimize and self.device.type == 'cuda':
            self.optimize_cuda_settings()
    
    def get_best_device(self):
        """获取最佳计算设备"""
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch未安装，使用CPU")
            return 'cpu'
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA不可用，使用CPU")
            return 'cpu'
        
        device_count = torch.cuda.device_count()
        logger.info(f"🚀 发现 {device_count} 个CUDA设备")
        
        if device_count == 0:
            logger.warning("⚠️ 未发现CUDA设备，使用CPU")
            return 'cpu'
        
        # 获取所有GPU信息
        gpu_info = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            
            # 获取当前内存使用情况
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = props.total_memory
            memory_free = memory_total - memory_allocated
            
            gpu_info.append({
                'id': i,
                'name': props.name,
                'total_memory': memory_total,
                'free_memory': memory_free,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            })
            
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  内存: {memory_total/1e9:.1f}GB (可用: {memory_free/1e9:.1f}GB)")
            logger.info(f"  计算能力: {props.major}.{props.minor}")
            logger.info(f"  多处理器数量: {props.multi_processor_count}")
        
        # 选择最佳GPU（优先考虑可用内存）
        best_gpu = max(gpu_info, key=lambda x: x['free_memory'])
        best_device_id = best_gpu['id']
        
        if TORCH_AVAILABLE:
            device = torch.device(f'cuda:{best_device_id}')
        else:
            device = f'cuda:{best_device_id}'

        self.device_properties = best_gpu

        logger.info(f"✅ 选择设备: {device} ({best_gpu['name']})")
        return device
    
    def optimize_cuda_settings(self):
        """优化CUDA设置"""
        if not TORCH_AVAILABLE or (hasattr(self.device, 'type') and self.device.type != 'cuda') or (isinstance(self.device, str) and not self.device.startswith('cuda')):
            return
        
        try:
            # 设置当前设备
            torch.cuda.set_device(self.device)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 设置内存分配策略
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device)
                logger.info(f"🔧 设置GPU内存使用比例: {self.memory_fraction}")
            
            # 启用cudnn基准模式（如果输入大小固定）
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                logger.info("🚀 启用cuDNN基准模式")
            
            # 启用cudnn确定性模式（可选，会降低性能但提高可重现性）
            # torch.backends.cudnn.deterministic = True
            
            logger.info("✅ CUDA设置优化完成")
            
        except Exception as e:
            logger.error(f"❌ CUDA设置优化失败: {e}")
    
    def get_memory_info(self, device_id: Optional[int] = None):
        """获取GPU内存信息"""
        if self.device.type != 'cuda':
            return {'error': 'CUDA不可用'}
        
        if device_id is None:
            device_id = self.device.index
        
        try:
            # PyTorch内存信息
            allocated = torch.cuda.memory_allocated(device_id) / 1e9
            reserved = torch.cuda.memory_reserved(device_id) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(device_id) / 1e9
            max_reserved = torch.cuda.max_memory_reserved(device_id) / 1e9
            
            # 设备属性
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory / 1e9
            
            memory_info = {
                'device_id': device_id,
                'device_name': props.name,
                'total_memory': total,
                'allocated_memory': allocated,
                'reserved_memory': reserved,
                'free_memory': total - allocated,
                'max_allocated': max_allocated,
                'max_reserved': max_reserved,
                'utilization_percent': (allocated / total) * 100
            }
            
            # 如果NVML可用，获取更详细信息
            if self.nvml_initialized:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # GPU使用率
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info['gpu_utilization'] = util.gpu
                    memory_info['memory_utilization'] = util.memory
                    
                    # 温度
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    memory_info['temperature'] = temp
                    
                    # 功耗
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    memory_info['power_usage'] = power
                    
                except Exception as e:
                    logger.debug(f"NVML信息获取失败: {e}")
            
            return memory_info
            
        except Exception as e:
            logger.error(f"获取GPU内存信息失败: {e}")
            return {'error': str(e)}
    
    def get_all_gpu_info(self):
        """获取所有GPU信息"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return []
        
        gpu_list = []
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            gpu_info = self.get_memory_info(i)
            gpu_list.append(gpu_info)
        
        return gpu_list
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("🧹 GPU缓存已清理")
    
    def reset_peak_memory_stats(self):
        """重置峰值内存统计"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.reset_accumulated_memory_stats(self.device)
            logger.info("📊 GPU内存统计已重置")
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "操作"):
        """内存监控上下文管理器"""
        if self.device.type != 'cuda':
            yield
            return
        
        # 记录开始状态
        self.reset_peak_memory_stats()
        start_memory = self.get_memory_info()
        
        logger.info(f"🔍 开始监控 '{operation_name}' 的GPU内存使用")
        logger.info(f"初始内存: {start_memory['allocated_memory']:.2f}GB")
        
        try:
            yield
        finally:
            # 记录结束状态
            end_memory = self.get_memory_info()
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1e9
            
            logger.info(f"📊 '{operation_name}' 内存使用报告:")
            logger.info(f"  结束内存: {end_memory['allocated_memory']:.2f}GB")
            logger.info(f"  峰值内存: {peak_memory:.2f}GB")
            logger.info(f"  内存增量: {end_memory['allocated_memory'] - start_memory['allocated_memory']:.2f}GB")
    
    def check_memory_available(self, required_gb: float) -> bool:
        """检查是否有足够的GPU内存"""
        if self.device.type != 'cuda':
            return True  # CPU模式总是返回True
        
        memory_info = self.get_memory_info()
        available_gb = memory_info.get('free_memory', 0)
        
        if available_gb >= required_gb:
            logger.info(f"✅ GPU内存充足: 需要{required_gb:.1f}GB, 可用{available_gb:.1f}GB")
            return True
        else:
            logger.warning(f"⚠️ GPU内存不足: 需要{required_gb:.1f}GB, 可用{available_gb:.1f}GB")
            return False
    
    def get_optimal_batch_size(self, model_memory_gb: float, max_batch_size: int = 128) -> int:
        """根据GPU内存自动计算最优批次大小"""
        if self.device.type != 'cuda':
            return min(32, max_batch_size)  # CPU默认批次大小
        
        memory_info = self.get_memory_info()
        available_memory = memory_info.get('free_memory', 0) * self.memory_fraction
        
        # 估算每个样本需要的内存（包括模型、梯度、优化器状态等）
        memory_per_sample = model_memory_gb * 4  # 经验值：模型大小的4倍
        
        if memory_per_sample <= 0:
            return min(32, max_batch_size)
        
        optimal_batch_size = int(available_memory / memory_per_sample)
        optimal_batch_size = max(1, min(optimal_batch_size, max_batch_size))
        
        logger.info(f"🎯 推荐批次大小: {optimal_batch_size} (基于{available_memory:.1f}GB可用内存)")
        return optimal_batch_size
    
    def supports_mixed_precision(self) -> bool:
        """检查是否支持混合精度训练"""
        if self.device.type != 'cuda':
            return False
        
        # 检查计算能力（需要7.0以上支持Tensor Cores）
        props = torch.cuda.get_device_properties(self.device)
        compute_capability = props.major + props.minor * 0.1
        
        supports_fp16 = compute_capability >= 7.0
        
        if supports_fp16:
            logger.info(f"✅ 支持混合精度训练 (计算能力: {props.major}.{props.minor})")
        else:
            logger.info(f"⚠️ 不支持混合精度训练 (计算能力: {props.major}.{props.minor}, 需要7.0+)")
        
        return supports_fp16
    
    def __str__(self) -> str:
        """返回CUDA管理器的字符串表示"""
        if self.device.type == 'cpu':
            return "CUDAManager(device=CPU)"
        
        memory_info = self.get_memory_info()
        return (f"CUDAManager(device={self.device}, "
                f"memory={memory_info.get('allocated_memory', 0):.1f}GB/"
                f"{memory_info.get('total_memory', 0):.1f}GB)")


def get_cuda_info():
    """获取CUDA环境信息"""
    info = {
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'device_count': 0,
        'current_device': None,
        'cuda_version': None,
        'cudnn_version': None
    }
    
    if TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            info['current_device'] = torch.cuda.current_device()
            info['cuda_version'] = torch.version.cuda
            
            if hasattr(torch.backends.cudnn, 'version'):
                info['cudnn_version'] = torch.backends.cudnn.version()
    
    return info


def print_cuda_summary():
    """打印CUDA环境摘要"""
    print("🔍 CUDA环境检查")
    print("=" * 50)
    
    info = get_cuda_info()
    
    print(f"PyTorch: {'✅ 已安装' if info['torch_available'] else '❌ 未安装'}")
    
    if info['torch_available']:
        print(f"CUDA: {'✅ 可用' if info['cuda_available'] else '❌ 不可用'}")
        
        if info['cuda_available']:
            print(f"CUDA版本: {info['cuda_version']}")
            print(f"cuDNN版本: {info['cudnn_version']}")
            print(f"GPU数量: {info['device_count']}")
            print(f"当前设备: {info['current_device']}")
            
            # 显示所有GPU信息
            manager = CUDAManager()
            gpu_list = manager.get_all_gpu_info()
            
            for gpu in gpu_list:
                if 'error' not in gpu:
                    print(f"\nGPU {gpu['device_id']}: {gpu['device_name']}")
                    print(f"  内存: {gpu['allocated_memory']:.1f}GB / {gpu['total_memory']:.1f}GB")
                    print(f"  使用率: {gpu['utilization_percent']:.1f}%")
                    
                    if 'temperature' in gpu:
                        print(f"  温度: {gpu['temperature']}°C")
                    if 'power_usage' in gpu:
                        print(f"  功耗: {gpu['power_usage']:.1f}W")


if __name__ == "__main__":
    # 测试CUDA工具
    print_cuda_summary()
    
    # 创建CUDA管理器
    manager = CUDAManager()
    print(f"\n{manager}")
    
    # 测试内存监控
    with manager.memory_monitor("测试操作"):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # 创建一个测试张量
            test_tensor = torch.randn(1000, 1000, device=manager.device)
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
