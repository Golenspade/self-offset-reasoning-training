# 🤖 赛博裁判长 (Cyber-Judge)

**项目代号**: Cyber-Judge  
**核心理念**: 邦多利怀孕 (Bandori Pregnancy) - 模块化、可组合、高并发  
**目标**: 开发一个极速、毒舌、具备"查成分"和"定性"能力的 QQ/Discord 机器人

## 🎯 核心体验

- ⚡ **秒回**: <1s 响应时间
- 🧠 **有记忆**: 记录用户历史发言
- 🎭 **逻辑自洽**: 基于 ReAct 循环的推理能力
- 💬 **说话好听**: 指怪话多

## 🏗️ 架构设计

### 混合技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                    赛博裁判长系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  The Harvester│───▶│ The Refinery │───▶│  The Brain   │ │
│  │   (Go爬虫)    │    │ (Python清洗) │    │ (LLM推理)    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         └────────────────────┴────────────────────┘         │
│                              │                              │
│                    ┌─────────▼─────────┐                   │
│                    │    The Body       │                   │
│                    │  (NoneBot2 后端)  │                   │
│                    └─────────┬─────────┘                   │
│                              │                              │
│                    ┌─────────▼─────────┐                   │
│                    │   The Memory      │                   │
│                    │ (SQLite/ChromaDB) │                   │
│                    └───────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 模块说明

1. **The Harvester (语料掠夺)**
   - 技术栈: Go + Colly
   - 功能: 高并发抓取贴吧精品内容
   - 产出: `raw_judgments.json`

2. **The Refinery (数据清洗)**
   - 技术栈: Python + Pandas + LLM API
   - 功能: 数据蒸馏和格式化
   - 产出: `production_examples.txt`, `train.jsonl`

3. **The Brain (大脑)**
   - 生产环境: Cerebras API (Llama 3.1 8B / 3.3 70B)
   - 实验环境: RTX 4060 + Unsloth (Qwen 2.5 3B)
   - 模式: System Prompt + Few-Shot / LoRA 微调

4. **The Body (躯干)**
   - 技术栈: Python + NoneBot2 + NapCatQQ
   - 功能: 对接 NTQQ 协议，处理消息

5. **The Memory (记忆)**
   - 技术栈: SQLite (轻量) / ChromaDB (向量检索)
   - 功能: 用户历史记录存储和检索

## 📁 项目结构

```
cyber_judge/
├── harvester/          # Go 爬虫模块
│   ├── main.go
│   ├── crawler/
│   └── config/
├── refinery/           # Python 数据清洗模块
│   ├── cleaner.py
│   ├── distiller.py
│   └── formatter.py
├── brain/              # LLM 推理模块
│   ├── cerebras_client.py
│   ├── local_model.py
│   ├── prompts/
│   └── react_loop.py
├── body/               # NoneBot2 后端
│   ├── bot.py
│   ├── plugins/
│   └── config.yml
├── memory/             # 数据存储模块
│   ├── sqlite_store.py
│   ├── vector_store.py
│   └── schemas/
├── configs/            # 配置文件
├── data/               # 数据目录
│   ├── raw/
│   ├── processed/
│   └── examples/
└── tests/              # 测试
```

## 🚀 快速开始

### 环境要求

- Go 1.21+
- Python 3.10+
- (可选) CUDA 12.1+ for RTX 4060

### 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt

# Go 依赖
cd harvester && go mod download
```

### 运行

```bash
# 1. 启动爬虫
cd harvester && go run main.go

# 2. 清洗数据
python refinery/cleaner.py

# 3. 启动 Bot
python body/bot.py
```

## 📊 开发路线图

- [ ] Phase 1: 语料掠夺模块 (Go Crawler)
- [ ] Phase 2: 数据清洗模块 (Python Refinery)
- [ ] Phase 3: 大脑模块 (Cerebras + Local)
- [ ] Phase 4: 躯干模块 (NoneBot2 + NapCatQQ)
- [ ] Phase 5: 记忆模块 (SQLite/ChromaDB)
- [ ] Phase 6: Agentic 回路 (ReAct Loop)

## 💡 进阶特性

- **并发多重人格**: 同时请求 3 个不同人格，选择最佳回复
- **自动归档**: 每日总结群聊十大逆天言论
- **延迟套利**: 利用 Cerebras 极速 API 实现亚秒级响应

## 📝 License

MIT

