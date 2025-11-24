# 赛博裁判长 - Makefile

.PHONY: help install crawl clean distill run-bot test

help:
	@echo "赛博裁判长 - 可用命令:"
	@echo "  make install    - 安装所有依赖"
	@echo "  make crawl      - 运行爬虫抓取数据"
	@echo "  make clean      - 清洗数据"
	@echo "  make distill    - LLM 蒸馏数据"
	@echo "  make run-bot    - 启动 Bot"
	@echo "  make test       - 运行测试"
	@echo "  make all        - 完整流程（爬虫->清洗->蒸馏）"

install:
	@echo "📦 安装 Python 依赖..."
	pip install -r requirements.txt
	@echo "📦 安装 Go 依赖..."
	cd harvester && go mod download
	@echo "✅ 依赖安装完成"

crawl:
	@echo "🕷️  启动爬虫..."
	cd harvester && go run main.go

clean:
	@echo "🧹 清洗数据..."
	python refinery/cleaner.py

distill:
	@echo "🧪 蒸馏数据..."
	python refinery/distiller.py

run-bot:
	@echo "🤖 启动赛博裁判长..."
	cd body && python bot.py

test:
	@echo "🧪 运行测试..."
	pytest tests/ -v

all: crawl clean distill
	@echo "✅ 数据准备完成！现在可以运行 make run-bot 启动 Bot"

# 开发相关
format:
	@echo "🎨 格式化代码..."
	black .
	cd harvester && go fmt ./...

lint:
	@echo "🔍 代码检查..."
	flake8 .
	cd harvester && go vet ./...

# 清理
clean-data:
	@echo "🗑️  清理数据..."
	rm -rf data/raw/*
	rm -rf data/processed/*

clean-all: clean-data
	@echo "🗑️  清理所有生成文件..."
	rm -rf data/memory.db
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

