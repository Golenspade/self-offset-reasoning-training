#!/bin/bash
# 文件名: deploy_to_cloud.sh
# 云端部署脚本 - 自偏移推理训练项目

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
PROJECT_NAME="logic-training"
VERSION=${VERSION:-"latest"}
REGISTRY=${REGISTRY:-"registry.cn-hangzhou.aliyuncs.com"}
NAMESPACE=${NAMESPACE:-"your-namespace"}
IMAGE_NAME="${REGISTRY}/${NAMESPACE}/${PROJECT_NAME}:${VERSION}"

# 云平台配置
CLOUD_PROVIDER=${CLOUD_PROVIDER:-"aliyun"}  # aliyun, aws, gcp, azure
CLUSTER_NAME=${CLUSTER_NAME:-"logic-training-cluster"}
REGION=${REGION:-"cn-hangzhou"}

echo -e "${BLUE}🚀 开始部署自偏移推理训练项目到云端${NC}"
echo "=================================="
echo "项目名称: ${PROJECT_NAME}"
echo "版本: ${VERSION}"
echo "镜像: ${IMAGE_NAME}"
echo "云平台: ${CLOUD_PROVIDER}"
echo "=================================="

# 函数：检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 命令未找到，请先安装${NC}"
        exit 1
    fi
}

# 函数：构建Docker镜像
build_docker_image() {
    echo -e "${YELLOW}🔨 构建Docker镜像...${NC}"
    
    # 检查Dockerfile是否存在
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}❌ Dockerfile不存在${NC}"
        exit 1
    fi
    
    # 构建镜像
    docker build -t ${PROJECT_NAME}:${VERSION} .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Docker镜像构建成功${NC}"
    else
        echo -e "${RED}❌ Docker镜像构建失败${NC}"
        exit 1
    fi
}

# 函数：推送镜像到云端仓库
push_to_registry() {
    echo -e "${YELLOW}📤 推送镜像到云端仓库...${NC}"
    
    # 标记镜像
    docker tag ${PROJECT_NAME}:${VERSION} ${IMAGE_NAME}
    
    # 登录到镜像仓库（需要预先配置认证）
    echo "正在推送到: ${IMAGE_NAME}"
    docker push ${IMAGE_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 镜像推送成功${NC}"
    else
        echo -e "${RED}❌ 镜像推送失败${NC}"
        exit 1
    fi
}

# 函数：部署到Kubernetes
deploy_to_kubernetes() {
    echo -e "${YELLOW}☸️ 部署到Kubernetes集群...${NC}"
    
    # 检查kubectl是否可用
    check_command kubectl
    
    # 检查集群连接
    kubectl cluster-info &> /dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 无法连接到Kubernetes集群${NC}"
        exit 1
    fi
    
    # 创建命名空间（如果不存在）
    kubectl create namespace logic-training --dry-run=client -o yaml | kubectl apply -f -
    
    # 应用Kubernetes配置
    if [ -f "k8s-training-job.yaml" ]; then
        # 替换镜像名称
        sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s-training-job.yaml > k8s-training-job-deployed.yaml
        
        kubectl apply -f k8s-training-job-deployed.yaml
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Kubernetes部署成功${NC}"
        else
            echo -e "${RED}❌ Kubernetes部署失败${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️ k8s-training-job.yaml不存在，跳过Kubernetes部署${NC}"
    fi
}

# 函数：创建云存储
setup_cloud_storage() {
    echo -e "${YELLOW}💾 设置云存储...${NC}"
    
    case ${CLOUD_PROVIDER} in
        "aliyun")
            echo "设置阿里云OSS存储..."
            # 这里可以添加阿里云OSS的设置命令
            ;;
        "aws")
            echo "设置AWS S3存储..."
            # 这里可以添加AWS S3的设置命令
            ;;
        "gcp")
            echo "设置Google Cloud Storage..."
            # 这里可以添加GCS的设置命令
            ;;
        "azure")
            echo "设置Azure Blob Storage..."
            # 这里可以添加Azure的设置命令
            ;;
        *)
            echo -e "${YELLOW}⚠️ 未知的云平台: ${CLOUD_PROVIDER}${NC}"
            ;;
    esac
}

# 函数：监控部署状态
monitor_deployment() {
    echo -e "${YELLOW}👀 监控部署状态...${NC}"
    
    # 等待Pod启动
    echo "等待训练任务启动..."
    kubectl wait --for=condition=Ready pod -l job-name=logic-training-job --timeout=300s -n logic-training
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 训练任务已启动${NC}"
        
        # 显示Pod状态
        kubectl get pods -n logic-training -l job-name=logic-training-job
        
        # 显示日志
        echo -e "${BLUE}📋 训练日志:${NC}"
        kubectl logs -f job/logic-training-job -n logic-training
    else
        echo -e "${RED}❌ 训练任务启动超时${NC}"
        kubectl describe job logic-training-job -n logic-training
        exit 1
    fi
}

# 函数：清理资源
cleanup() {
    echo -e "${YELLOW}🧹 清理临时资源...${NC}"
    
    # 删除临时文件
    if [ -f "k8s-training-job-deployed.yaml" ]; then
        rm k8s-training-job-deployed.yaml
    fi
    
    echo -e "${GREEN}✅ 清理完成${NC}"
}

# 主要部署流程
main() {
    # 检查必要的命令
    check_command docker
    
    # 构建镜像
    build_docker_image
    
    # 推送镜像
    push_to_registry
    
    # 设置云存储
    setup_cloud_storage
    
    # 部署到Kubernetes
    deploy_to_kubernetes
    
    # 监控部署
    if [ "${MONITOR:-true}" = "true" ]; then
        monitor_deployment
    fi
    
    # 清理
    cleanup
    
    echo -e "${GREEN}🎉 部署完成！${NC}"
    echo "=================================="
    echo "镜像: ${IMAGE_NAME}"
    echo "集群: ${CLUSTER_NAME}"
    echo "命名空间: logic-training"
    echo "=================================="
    echo -e "${BLUE}💡 使用以下命令查看训练状态:${NC}"
    echo "kubectl get jobs -n logic-training"
    echo "kubectl logs -f job/logic-training-job -n logic-training"
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --cloud-provider)
            CLOUD_PROVIDER="$2"
            shift 2
            ;;
        --no-monitor)
            MONITOR="false"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --version VERSION        设置镜像版本 (默认: latest)"
            echo "  --registry REGISTRY      设置镜像仓库 (默认: registry.cn-hangzhou.aliyuncs.com)"
            echo "  --namespace NAMESPACE    设置命名空间 (默认: your-namespace)"
            echo "  --cloud-provider PROVIDER 设置云平台 (默认: aliyun)"
            echo "  --no-monitor             不监控部署状态"
            echo "  --help                   显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 执行主函数
main
