# 🚀 自偏移训练项目 - 环境设置指南

## ✅ 依赖问题已解决！

你的项目环境已经成功配置完成。以下是使用指南：

## 📁 新增文件说明

- `venv/` - Python虚拟环境目录
- `activate_env.sh` - 便捷的环境激活脚本
- `check_dependencies.py` - 依赖检查脚本
- `requirements.txt` - 更新后的依赖列表（适合macOS CPU环境）

## 🎯 快速开始

### 1. 激活虚拟环境
```bash
# 方法1: 使用便捷脚本
./activate_env.sh

# 方法2: 手动激活
source venv/bin/activate
```

### 2. 验证环境
```bash
python check_dependencies.py
```

### 3. 运行项目
```bash
# 生成数据集
python generate_robust_dataset.py

# 训练模型（推荐使用重构版）
python breakthrough_training_system_refactored.py

# 评估模型
python clean_evaluation_system.py
```

## 📦 已安装的主要依赖

- **PyTorch 2.8.0** - 深度学习框架（CPU版本）
- **NumPy 2.3.2** - 数值计算
- **Matplotlib 3.10.5** - 数据可视化
- **Pandas 2.3.1** - 数据处理
- **Scikit-learn 1.7.1** - 机器学习工具
- **TensorBoard 2.20.0** - 训练监控
- **其他工具包** - tqdm, seaborn, pytest等

## 🔧 环境管理

### 激活环境
```bash
source venv/bin/activate
```

### 退出环境
```bash
deactivate
```

### 重新安装依赖（如果需要）
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## 🚨 注意事项

1. **虚拟环境**: 项目使用独立的虚拟环境，避免与系统Python冲突
2. **macOS优化**: 依赖配置已针对macOS ARM64架构优化
3. **CPU训练**: 当前配置适合CPU训练，如需GPU请参考`requirements_cuda.txt`

## 🐛 故障排除

### 如果遇到导入错误
```bash
python check_dependencies.py
```

### 如果虚拟环境损坏
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 如果权限问题
```bash
chmod +x activate_env.sh
```

## 📚 项目结构提醒

- **推荐使用重构版脚本**: `breakthrough_training_system_refactored.py`
- **数据生成**: `generate_robust_dataset.py`
- **评估系统**: `clean_evaluation_system.py`
- **核心模块**: `src/logic_transformer/`

## 🎉 成功标志

如果看到以下输出，说明环境配置成功：
```
🎉 所有依赖检查通过！项目可以正常运行。
```

现在你可以开始使用项目了！
