# 赛博裁判长 - Makefile

.PHONY: help install crawl clean distill run-bot test venv

# 虚拟环境路径
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "赛博裁判长 - 可用命令:"
	@echo "  make venv           - 创建虚拟环境"
	@echo "  make install        - 安装所有依赖（自动创建虚拟环境）"
	@echo "  make install-browser - 安装 Playwright 浏览器"
	@echo "  make crawl          - 运行爬虫抓取数据 (Go 版本)"
	@echo "  make crawl-pw       - 运行 Playwright 爬虫 (推荐)"
	@echo "  make clean          - 清洗数据"
	@echo "  make distill        - LLM 蒸馏数据"
	@echo "  make run-bot        - 启动 Bot"
	@echo "  make test           - 运行测试"
	@echo "  make all            - 完整流程（爬虫->清洗->蒸馏）"

venv:
	@echo "🔧 创建虚拟环境..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "✨ 使用 uv 创建虚拟环境..."; \
		uv venv $(VENV); \
	else \
		echo "📦 使用 python3 -m venv 创建虚拟环境..."; \
		python3 -m venv $(VENV); \
	fi
	@echo "✅ 虚拟环境创建完成: $(VENV)"

install: venv
	@echo "📦 安装 Python 依赖..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "✨ 使用 uv 安装依赖（极速）..."; \
		uv pip install -r requirements.txt; \
	else \
		echo "📦 使用 pip 安装依赖..."; \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.txt; \
	fi
	@echo "📦 安装 Go 依赖..."
	cd harvester && go mod download
	@echo "✅ 依赖安装完成"
	@echo ""
	@echo "💡 提示: 使用以下命令激活虚拟环境:"
	@echo "   source $(VENV)/bin/activate"

crawl:
	@echo "🕷️  启动爬虫 (Go 版本)..."
	cd harvester && go run main.go

crawl-pw:
	@echo "🎭 启动 Playwright 爬虫..."
	$(PYTHON) harvester/crawler_playwright.py

crawl-interactive:
	@echo "🎭 启动交互式爬虫（需要手动完成验证）..."
	$(PYTHON) harvester/interactive_crawler.py

crawl-bandori:
	@echo "🎯 深度爬取邦多利怀孕吧（包含回复）..."
	$(PYTHON) harvester/crawl_bandori_deep.py

install-browser:
	@echo "📦 安装 Playwright 浏览器..."
	$(VENV)/bin/playwright install chromium

clean:
	@echo "🧹 清洗数据..."
	$(PYTHON) refinery/cleaner.py

distill:
	@echo "🧪 蒸馏数据..."
	$(PYTHON) refinery/distiller.py

run-bot:
	@echo "🤖 启动赛博裁判长..."
	cd body && $(PYTHON) bot.py

test:
	@echo "🧪 运行测试..."
	$(VENV)/bin/pytest tests/ -v

all: crawl clean distill
	@echo "✅ 数据准备完成！现在可以运行 make run-bot 启动 Bot"

# 开发相关
format:
	@echo "🎨 格式化代码..."
	$(VENV)/bin/black .
	cd harvester && go fmt ./...

lint:
	@echo "🔍 代码检查..."
	$(VENV)/bin/flake8 .
	cd harvester && go vet ./...

# 清理
clean-data:
	@echo "🗑️  清理数据..."
	rm -rf data/raw/*
	rm -rf data/processed/*

clean-venv:
	@echo "🗑️  清理虚拟环境..."
	rm -rf $(VENV)

clean-all: clean-data clean-venv
	@echo "🗑️  清理所有生成文件..."
	rm -rf data/memory.db
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

