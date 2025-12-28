# 文件名: Dockerfile
# 自偏移推理训练项目 - 远程训练容器化配置

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements_remote.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_remote.txt

# 复制项目代码
COPY src/ ./src/
COPY analysis/ ./analysis/
COPY configs/ ./configs/
COPY *.py ./
COPY scripts/ ./scripts/
COPY remote/ ./remote/

# 创建必要目录
RUN mkdir -p /data /models /outputs /checkpoints /logs

# 设置权限
RUN chmod +x *.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# 暴露端口（用于分布式训练通信）
EXPOSE 12355

# 默认命令：以模块形式启动远程训练主程序
CMD ["python", "-m", "remote.remote_training_main"]
