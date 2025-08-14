# 📌 项目 ToDoList（按路线图精细拆解）

说明：
- 优先级：P0=必须优先、P1=重要、P2=可选
- 任务粒度：尽量控制在单个开发者 0.5–2 小时可完成
- 每个任务含：目标、完成标准(验收)、依赖、产出物

---

## P0｜近期（1–2周）

### A. 提升精确匹配率（后处理与约束解码）
- [ ] A1 创建括号/符号平衡校验器（logic_utils）
  - 目标：提供 `check_balance(expr:str)->bool` 与 `auto_fix(expr)->str`
  - 验收：新增 tests 通过，非法表达式能被检测，常见缺失可被修复
  - 依赖：无
  - 产出：logic_utils.py、tests/test_logic_sanity.py
- [ ] A2 约束解码（simple_model/base_model 解码阶段屏蔽非法 token）
  - 目标：按语法约束屏蔽不可能 token（操作符位置、括号配对、箭头两侧）
  - 验收：单元测试覆盖 5+ 典型约束；推断时非法序列比例明显下降
  - 依赖：A1
  - 产出：simple_model.py or src/.../models/base_model.py
- [ ] A3 统一后处理流水线（normalize -> balance -> dedup spaces）
  - 目标：实现 `postprocess(expr)->expr` 管道，训练/评估共用
  - 验收：clean_evaluation_system 调用后处理并写入报告
  - 依赖：A1
  - 产出：logic_utils.py、clean_evaluation_system.py

### B. 训练稳定性（调度/裁剪/早停/曲线）
- [ ] B1 学习率余弦退火（breakthrough_training_system_refactored）
  - 目标：新增 `--lr-schedule cos`，可视化 lr 变化
  - 验收：训练日志含 lr；图表保存 outputs/figures/lr_schedule.png
  - 依赖：无
  - 产出：breakthrough_training_system_refactored.py
- [ ] B2 梯度裁剪（全局范数）
  - 目标：参数 `--grad-clip 1.0`；日志记录裁剪统计
  - 验收：无 NaN/爆炸；训练收敛更稳定
  - 依赖：无
  - 产出：breakthrough_training_system_refactored.py
- [ ] B3 早停+最佳模型保存
  - 目标：监控 val_loss/logic_acc，保存 best.ckpt
  - 验收：早停生效；best.ckpt 与最后权重区分
  - 依赖：无
  - 产出：breakthrough_training_system_refactored.py, outputs/trained_models/
- [ ] B4 训练曲线可视化（loss/acc）
  - 目标：保存 outputs/figures/training_curves.png
  - 验收：曲线包含 train/val；随日志一起生成
  - 依赖：B1/B3
  - 产出：analysis 或训练脚本内的绘图函数

### C. 数据增强（更多逻辑等价变换）
- [ ] C1 双重否定与蕴含消除规则实现
  - 目标：在数据生成加入 `~~A=A`, `A->B = ~A|B`
  - 验收：生成样本中出现新变换；统计报告包含占比
  - 依赖：无
  - 产出：generate_robust_dataset.py
- [ ] C2 结合/分配律等价变换（受控概率注入）
  - 目标：实现 (A&(B&C))↔((A&B)&C) 等规则；可配置注入比例
  - 验收：数据质量检查通过；不引入不合法表达式
  - 依赖：C1
  - 产出：generate_robust_dataset.py、docs/transform_rules.md
- [ ] C3 逻辑等价性回归测试
  - 目标：对每种规则提供可自动验证的等价单测
  - 验收：tests 通过；CI 内可运行
  - 依赖：C1/C2
  - 产出：tests/test_equivalences.py

### D. 评估与报告（指标定义与统一格式）
- [ ] D1 指标定义文档（等价/语法/可读性）
  - 目标：明确三类指标的公式与计算接口
  - 验收：docs/metrics_spec.md 创建；评审通过
  - 依赖：无
- [ ] D2 评估实现与 JSON Schema 统一
  - 目标：clean_evaluation_system 输出统一 Schema v2
  - 验收：outputs/reports/eval_v2.json 含三类指标与版本号
  - 依赖：D1
- [ ] D3 分析面板增强（complete_experiment_summary_refactored）
  - 目标：新增多指标对比、趋势图、效率指标
  - 验收：图表/报告落盘，脚本无交互即可生成
  - 依赖：D2

### E. 工程化（测试与CI）
- [ ] E1 最小单元测试集
  - 目标：覆盖 logic_utils, data 生成、评估核心入口
  - 验收：pytest 全绿；覆盖率>40%
  - 依赖：A/C/D
- [ ] E2 GitHub Actions 基础 CI
  - 目标：lint+type+test 三步流水
  - 验收：PR 自动运行；状态徽章加入 README
  - 依赖：E1

---

## P1｜中期（1–2月）

### F. 模型升级与抽象
- [ ] F1 新增 Transformer 模型（PyTorch）
  - 目标：`src/.../models/transformer_model.py` 与选择开关
  - 验收：能在小数据上跑通 1 epoch；评估可用
  - 依赖：E1
- [ ] F2 统一模型接口（forward/encode/decode）
  - 目标：simple_model 与 transformer 共用训练/评估管线
  - 验收：同一训练脚本可切换模型
  - 依赖：F1

### G. 复杂逻辑扩展（受控子集）
- [ ] G1 一阶逻辑最小子集规格定义
  - 目标：语法/词表/约束文档；可解析范式
  - 验收：docs/fol_subset_spec.md；解析器骨架
  - 依赖：D1
- [ ] G2 解析与生成原型
  - 目标：支持该子集的数据生成与基本验证
  - 验收：生成10k样本；评估通过
  - 依赖：G1, C

### H. 端到端流水线与分布式
- [ ] H1 Makefile/Invoke 任务
  - 目标：`make data/train/eval/analyze`
  - 验收：一条命令串起端到端流程
  - 依赖：A–D
- [ ] H2 DDP/FSDP 原型（多卡）
  - 目标：train_cuda.py 支持多卡参数；记录吞吐
  - 验收：2卡验证通过；速度线性提升>1.6x
  - 依赖：F1

### I. 实验追踪
- [ ] I1 集成 Weights & Biases/MLflow（可配置）
  - 目标：打点超参/指标/图表/模型
  - 验收：README 配置说明；能在本地/远程落盘
  - 依赖：B/D

---

## P2｜远期（3–6月）

### J. 自然语言迁移与多步推理
- [ ] J1 小规模 NLI 数据映射实验
  - 目标：从符号到自然语言的对齐原型
  - 验收：能在 SNLI 子集跑通逻辑等价评估
  - 依赖：F1, I
- [ ] J2 链式推理与验证器
  - 目标：定义 chain-of-thought 约束与可验证器
  - 验收：给出示例与自动验证脚本
  - 依赖：A/D/F

### K. 自我修复训练与超参搜索
- [ ] K1 失败回退与参数搜索脚本
  - 目标：训练失败自动回退到上次 best 并网格/贝叶斯搜索
  - 验收：提供 `--auto-repair` 开关与日志
  - 依赖：B3, I

### L. 发布与复现
- [ ] L1 发布版 Docker 与最小 Benchmark
  - 目标：提供一键复现脚本与镜像
  - 验收：新环境30分钟内复现核心结果
  - 依赖：H/I

---

## 管理与状态
- 每周维护 Roadmap→ToDoList 映射；完成即勾选并在 PR 描述引用任务编号（如 A2, D3）
- 所有产出（报告/图表/配置）写入 outputs/ 或 docs/，并在 README 链接

