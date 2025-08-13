# 🚀 远程算力训练使用指南

## 📋 概述

本指南介绍如何使用自偏移推理训练项目的远程算力训练系统，支持多种云平台和容器化部署。

## 🎯 系统架构

```
本地开发环境
    ↓ (数据同步)
云端存储 (S3/OSS/GCS/Azure)
    ↓ (容器化部署)
Kubernetes集群 (GPU节点)
    ↓ (训练执行)
远程训练系统
    ↓ (结果同步)
云端存储 + 监控系统
```

## 🛠️ 快速开始

### 1. 环境准备

#### 本地环境
```bash
# 安装依赖
pip install -r requirements.txt
pip install -r requirements_remote.txt

# 安装Docker
# 安装kubectl (Kubernetes命令行工具)
# 安装云平台CLI工具
```

#### 云平台配置
```bash
# 阿里云
aliyun configure

# AWS
aws configure

# Google Cloud
gcloud auth login

# Azure
az login
```

### 2. 配置设置

#### 创建配置文件
```bash
python remote_training_config.py
```

#### 环境变量配置
```bash
# 训练参数
export EPOCHS=100
export BATCH_SIZE=64
export LEARNING_RATE=0.001

# 云存储配置
export CLOUD_PROVIDER=aliyun  # aws, gcp, azure, aliyun
export CLOUD_BUCKET=your-bucket-name
export CLOUD_ACCESS_KEY=your-access-key
export CLOUD_SECRET_KEY=your-secret-key

# 路径配置
export REMOTE_DATA_PATH=/data/logic_training
export REMOTE_OUTPUT_PATH=/outputs/training_results
```

### 3. 数据准备

#### 同步训练数据到云端
```bash
# 生成并上传数据
python sync_data_to_remote.py --action upload --force-regenerate

# 仅上传现有数据
python sync_data_to_remote.py --action upload
```

### 4. 部署选项

#### 选项A: 本地Docker测试
```bash
# 构建镜像
docker build -t logic-training:latest .

# 运行容器
docker run -it --rm \
  -e EPOCHS=20 \
  -e BATCH_SIZE=16 \
  -e DEBUG_MODE=true \
  -v $(pwd)/data:/data/logic_training:ro \
  -v $(pwd)/outputs:/outputs \
  logic-training:latest
```

#### 选项B: Docker Compose本地集群
```bash
# 创建必要目录
mkdir -p volumes/{models,outputs,checkpoints} logs

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f logic-training

# 停止服务
docker-compose down
```

#### 选项C: Kubernetes云端部署
```bash
# 部署到云端
chmod +x deploy_to_cloud.sh
./deploy_to_cloud.sh --cloud-provider aliyun --version v1.0

# 监控训练状态
kubectl get jobs -n logic-training
kubectl logs -f job/logic-training-job -n logic-training
```

## 📊 监控和管理

### 训练状态监控
```bash
# 查看Pod状态
kubectl get pods -n logic-training

# 查看训练日志
kubectl logs -f deployment/logic-training -n logic-training

# 查看资源使用
kubectl top pods -n logic-training
```

### 检查点管理
```bash
# 同步检查点到云端
python sync_data_to_remote.py --action sync-checkpoints

# 下载检查点
python sync_data_to_remote.py --action download
```

## 🔧 高级配置

### 分布式训练
```yaml
# k8s-training-job.yaml
env:
- name: USE_DISTRIBUTED
  value: "true"
- name: WORLD_SIZE
  value: "4"  # 4个GPU
```

### 自定义资源配置
```yaml
resources:
  requests:
    nvidia.com/gpu: "2"  # 请求2个GPU
    memory: "16Gi"
    cpu: "8"
  limits:
    nvidia.com/gpu: "2"
    memory: "32Gi"
    cpu: "16"
```

### 监控集成
```bash
# 启用Weights & Biases
export ENABLE_WANDB=true
export WANDB_PROJECT=logic-training
export WANDB_ENTITY=your-team

# 启用Slack通知
export SLACK_WEBHOOK=your-webhook-url
```

## 📈 性能优化

### 数据加载优化
```python
# 在配置中设置
DATA_WORKERS=8
PREFETCH_FACTOR=4
```

### GPU内存优化
```bash
export GPU_MEMORY_LIMIT=8Gi
export BATCH_SIZE=32  # 根据GPU内存调整
```

### 网络优化
```bash
# 使用更快的存储类
storageClassName: fast-ssd
```

## 🚨 故障排除

### 常见问题

#### 1. 容器启动失败
```bash
# 检查镜像
docker images | grep logic-training

# 检查日志
docker logs container-name

# 检查资源
kubectl describe pod pod-name -n logic-training
```

#### 2. 数据加载失败
```bash
# 检查数据文件
ls -la /data/logic_training/

# 检查权限
kubectl exec -it pod-name -n logic-training -- ls -la /data/
```

#### 3. GPU不可用
```bash
# 检查GPU节点
kubectl get nodes -l accelerator=nvidia-tesla-v100

# 检查GPU资源
kubectl describe node node-name
```

#### 4. 网络连接问题
```bash
# 检查网络策略
kubectl get networkpolicy -n logic-training

# 测试连接
kubectl exec -it pod-name -n logic-training -- ping google.com
```

### 调试模式
```bash
# 启用调试模式
export DEBUG_MODE=true

# 干运行测试
export DRY_RUN=true

# 详细日志
export LOG_LEVEL=DEBUG
```

## 📋 最佳实践

### 1. 资源管理
- 根据数据大小和模型复杂度合理配置资源
- 使用资源配额防止过度使用
- 定期清理不需要的检查点

### 2. 安全配置
- 使用Kubernetes Secrets管理敏感信息
- 配置网络策略限制访问
- 定期更新容器镜像

### 3. 成本优化
- 使用抢占式实例降低成本
- 配置自动缩放
- 监控资源使用情况

### 4. 数据管理
- 定期备份重要数据
- 使用版本控制管理数据集
- 优化数据传输效率

## 🎯 示例工作流

### 完整训练流程
```bash
# 1. 准备数据
python sync_data_to_remote.py --action upload --force-regenerate

# 2. 部署训练任务
./deploy_to_cloud.sh --cloud-provider aliyun --version v1.0

# 3. 监控训练
kubectl logs -f job/logic-training-job -n logic-training

# 4. 下载结果
python sync_data_to_remote.py --action sync-checkpoints

# 5. 清理资源
kubectl delete job logic-training-job -n logic-training
```

## 📞 支持和反馈

如果遇到问题或有改进建议，请：

1. 查看日志文件获取详细错误信息
2. 检查配置是否正确
3. 参考故障排除部分
4. 在GitHub仓库提交Issue

---

**🎉 现在您可以在云端进行大规模的自偏移推理训练了！**
