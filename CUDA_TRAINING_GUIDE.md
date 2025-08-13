# 🚀 CUDA加速训练使用指南

## 📋 概述

本指南介绍如何使用自偏移推理训练项目的CUDA加速功能，实现GPU高性能训练。

## 🎯 CUDA系统特性

### ⚡ 核心优化特性
- **自动设备检测**: 智能选择最佳GPU设备
- **混合精度训练**: FP16加速训练，节省内存
- **内存管理**: 智能GPU内存分配和清理
- **批次优化**: 根据GPU内存自动调整批次大小
- **梯度累积**: 支持大批次等效训练
- **性能监控**: 实时GPU使用率和内存监控

### 🏗️ 系统架构
```
CUDA工具层 (cuda_utils.py)
    ↓
模型层 (model.py + CUDA支持)
    ↓
训练系统层 (cuda_training_system.py)
    ↓
主训练脚本 (train_cuda.py)
```

## 🛠️ 环境准备

### 1. 硬件要求
- **GPU**: NVIDIA GPU (计算能力 >= 6.0)
- **内存**: 推荐 >= 8GB GPU内存
- **CUDA**: CUDA 11.8 或更高版本

### 2. 软件依赖安装

#### 基础环境
```bash
# 确保有NVIDIA驱动和CUDA
nvidia-smi  # 检查GPU状态
nvcc --version  # 检查CUDA版本
```

#### Python依赖
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他CUDA依赖
pip install -r requirements_cuda.txt
```

#### 验证安装
```bash
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 🚀 快速开始

### 1. 环境检查
```bash
# 运行简化测试（不需要PyTorch）
python3 test_cuda_simple.py

# 运行完整CUDA测试（需要PyTorch）
python3 test_cuda_training.py
```

### 2. 基础训练
```bash
# 查看所有可用参数
python3 train_cuda.py --help

# 基础CUDA训练
python3 train_cuda.py \
    --data-dir data \
    --output-dir outputs/cuda_training \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# 自动优化批次大小
python3 train_cuda.py \
    --auto-batch-size \
    --use-mixed-precision \
    --epochs 100
```

### 3. 高级配置
```bash
# 大模型训练
python3 train_cuda.py \
    --hidden-size 512 \
    --num-heads 16 \
    --num-encoder-layers 6 \
    --num-decoder-layers 6 \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --use-mixed-precision

# 恢复训练
python3 train_cuda.py \
    --resume outputs/cuda_training/cuda_checkpoint_epoch_20.pth \
    --epochs 100
```

## 🔧 配置参数详解

### 模型参数
- `--hidden-size`: 隐藏层大小 (默认: 256)
- `--num-heads`: 注意力头数 (默认: 8)
- `--num-encoder-layers`: 编码器层数 (默认: 4)
- `--num-decoder-layers`: 解码器层数 (默认: 4)
- `--max-length`: 最大序列长度 (默认: 128)

### 训练参数
- `--batch-size`: 批次大小 (默认: 32)
- `--learning-rate`: 学习率 (默认: 0.001)
- `--epochs`: 训练轮次 (默认: 100)
- `--gradient-accumulation-steps`: 梯度累积步数 (默认: 1)

### CUDA参数
- `--use-mixed-precision`: 启用混合精度训练
- `--gpu-memory-fraction`: GPU内存使用比例 (默认: 0.8)
- `--auto-batch-size`: 自动优化批次大小

### 优化参数
- `--weight-decay`: 权重衰减 (默认: 1e-5)
- `--max-grad-norm`: 梯度裁剪阈值 (默认: 1.0)
- `--label-smoothing`: 标签平滑 (默认: 0.1)

## 📊 性能优化建议

### 1. 批次大小优化
```bash
# 让系统自动选择最优批次大小
python3 train_cuda.py --auto-batch-size

# 手动调整（根据GPU内存）
# 8GB GPU: batch-size 16-32
# 16GB GPU: batch-size 32-64
# 24GB GPU: batch-size 64-128
```

### 2. 混合精度训练
```bash
# 启用混合精度（推荐用于V100/A100等现代GPU）
python3 train_cuda.py --use-mixed-precision

# 检查GPU是否支持混合精度
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'计算能力: {props.major}.{props.minor}')
    print(f'支持混合精度: {props.major >= 7}')
"
```

### 3. 梯度累积
```bash
# 模拟大批次训练（等效batch_size = 32 * 4 = 128）
python3 train_cuda.py \
    --batch-size 32 \
    --gradient-accumulation-steps 4
```

### 4. 内存优化
```bash
# 降低GPU内存使用
python3 train_cuda.py \
    --gpu-memory-fraction 0.7 \
    --batch-size 16

# 使用更小的模型
python3 train_cuda.py \
    --hidden-size 128 \
    --num-heads 4 \
    --num-encoder-layers 3
```

## 🐳 Docker部署

### 1. 构建CUDA镜像
```bash
# 构建镜像
docker build -f Dockerfile.cuda -t logic-training-cuda:latest .

# 验证镜像
docker run --gpus all --rm logic-training-cuda:latest \
    python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 运行CUDA容器
```bash
# 基础运行
docker run --gpus all -it --rm \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/outputs:/app/outputs \
    logic-training-cuda:latest \
    python3 train_cuda.py --epochs 50

# 后台运行
docker run --gpus all -d \
    --name cuda-training \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/outputs:/app/outputs \
    logic-training-cuda:latest \
    python3 train_cuda.py --epochs 100 --batch-size 64

# 查看训练日志
docker logs -f cuda-training
```

## 📈 监控和调试

### 1. GPU监控
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 或使用nvtop（如果安装）
nvtop
```

### 2. 训练监控
```bash
# 查看训练日志
tail -f outputs/cuda_training/training.log

# 查看TensorBoard（如果启用）
tensorboard --logdir outputs/cuda_training/tensorboard
```

### 3. 内存调试
```python
# 在Python中监控GPU内存
import torch
print(f"已分配: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"已缓存: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

## 🚨 故障排除

### 常见问题

#### 1. CUDA不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA安装
nvcc --version

# 重新安装PyTorch CUDA版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. GPU内存不足
```bash
# 减小批次大小
python3 train_cuda.py --batch-size 8

# 启用梯度累积
python3 train_cuda.py --batch-size 8 --gradient-accumulation-steps 4

# 减小模型大小
python3 train_cuda.py --hidden-size 128 --num-heads 4
```

#### 3. 训练速度慢
```bash
# 启用混合精度
python3 train_cuda.py --use-mixed-precision

# 增大批次大小
python3 train_cuda.py --auto-batch-size

# 检查数据加载瓶颈
python3 train_cuda.py --log-frequency 10
```

#### 4. 模型不收敛
```bash
# 调整学习率
python3 train_cuda.py --learning-rate 0.0005

# 增加模型容量
python3 train_cuda.py --hidden-size 512 --num-heads 16

# 减少正则化
python3 train_cuda.py --weight-decay 1e-6 --label-smoothing 0.05
```

## 📊 性能基准

### 预期性能提升
- **GPU vs CPU**: 5-20x 加速
- **混合精度**: 1.5-2x 额外加速
- **批次优化**: 10-30% 性能提升

### 不同GPU性能参考
| GPU型号 | 推荐批次大小 | 预期训练时间 |
|---------|-------------|-------------|
| GTX 1080 Ti | 16-32 | 基准 |
| RTX 3080 | 32-64 | 0.6x |
| RTX 4090 | 64-128 | 0.4x |
| V100 | 32-64 | 0.5x |
| A100 | 64-128 | 0.3x |

## 🎯 最佳实践

1. **开始训练前**:
   - 运行 `test_cuda_simple.py` 检查环境
   - 使用 `--auto-batch-size` 找到最优批次大小
   - 启用混合精度训练（如果GPU支持）

2. **训练过程中**:
   - 监控GPU内存使用率
   - 定期保存检查点
   - 观察训练损失曲线

3. **性能调优**:
   - 根据GPU内存调整批次大小
   - 使用梯度累积模拟大批次
   - 适当调整模型大小

---

**🎉 现在您可以享受GPU加速的高性能训练了！**
