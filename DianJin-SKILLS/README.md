# 百技图

面向金融行业的 AI Agent 技能定义库，覆盖银行、保险、证券/资管三大业态，10 个专业角色、130+ 个标准化技能。每个技能以结构化 Markdown 定义，可直接对接大模型 Agent 框架进行解析与编排，也可作为 Prompt 规范独立使用。

> [!IMPORTANT]
> 本仓库所有技能定义仅供 AI Agent 编排与集成参考，不构成任何金融投资、保险或信贷建议。所有技能的 AI 输出均需经过专业人员审核后，方可用于实际业务决策。使用方须自行确保符合所在地区的监管合规要求。

---

## What's in the Repo

本仓库包含 10 个金融角色下的 133 个标准化技能定义，每个技能以独立目录存在，核心文件为 `SKILL.md`，包含 YAML Frontmatter 元数据（触发词、依赖关系、风险等级）及标准化 Markdown 正文（执行流程、输出格式、合规约束、审计追踪）。

---

## Repository Layout

```
./
├── corporate-banker/                # AI对公客户经理（8个技能）
├── credit-review-expert/            # AI信审专家（7个技能）
├── wealth-copilot/                  # AI理财经理（28个技能，按业务流程分8层）
├── credit-risk-manager/             # AI信贷风险管理专家（8个技能）
├── insurance-agent/                 # AI保险代理人（12个技能）
├── claim-expert/                    # AI保险理赔专家（8个技能）
├── underwriting-expert/             # AI保险核保专家（5个技能）
├── investment-researcher/           # AI研究员（34个技能）
├── investment-advisor/              # AI投资顾问（9个技能）
└── financial-engineering-expert/    # AI金融工程专家（14个技能）
```

每个技能目录的标准结构：

```
skill-name/
├── SKILL.md           # 技能定义（核心文件）
├── PROVENANCE.md      # 来源与变更记录
├── assets/            # 模板与输出示例
├── references/        # 参考资料（行业指南、话术库等）
├── scripts/           # 辅助脚本
└── tests/             # 测试用例
```

---

## Roles & Skills

### Roles Overview

| Role | Directory | Skills | Domain |
|------|-----------|--------|--------|
| AI对公客户经理 | `corporate-banker` | 8 | 银行·对公业务 |
| AI信审专家 | `credit-review-expert` | 7 | 银行·信贷审批 |
| AI理财经理 | `wealth-copilot` | 28 | 银行·财富管理 |
| AI信贷风险管理专家 | `credit-risk-manager` | 8 | 银行·风控 |
| AI保险代理人 | `insurance-agent` | 12 | 保险·销售 |
| AI保险理赔专家 | `claim-expert` | 8 | 保险·理赔 |
| AI保险核保专家 | `underwriting-expert` | 5 | 保险·核保 |
| AI研究员 | `investment-researcher` | 34 | 证券·研究 |
| AI投资顾问 | `investment-advisor` | 9 | 证券·投顾 |
| AI金融工程专家 | `financial-engineering-expert` | 14 | 证券·金工 |

---

### AI对公客户经理（corporate-banker）— 8 个技能

覆盖从访前规划到授信申请的对公信贷全流程。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 客户经理访前规划 | `corporate-banker/pre-visit-plan` | 生成客户拜访前的背景分析与拜访目标清单 |
| 产品路演方案生成 | `corporate-banker/product-roadshow` | 生成结构化的产品路演演示方案 |
| 走访纪要生成 | `corporate-banker/visit-memo` | 将现场拜访口述观察结构化为贷前尽调笔记，支持多轮增量更新 |
| 企业信贷尽职调查报告 | `corporate-banker/credit-due-diligence` | 生成符合信贷审批标准的尽职调查报告 |
| 财报分析 | `corporate-banker/financial-report-analysis` | 对企业财务报表进行多维度分析与风险识别 |
| 行业深度分析 | `corporate-banker/credit-industry-analysis` | 对企业所在行业进行信贷视角的深度分析 |
| 股权穿透与关联方分析 | `corporate-banker/equity-penetration-analysis` | 识别企业股权结构中的关联方与实际控制人 |
| 提交授信申请 | `corporate-banker/submit-credit-application` | 生成标准化的授信申请材料 |

---

### AI信审专家（credit-review-expert）— 7 个技能

覆盖从任务规划到押品评估的信贷审查全链路。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 信贷风控任务规划 | `credit-review-expert/ai_risk_planning` | 规划信贷审查的任务分解与执行路径 |
| 访前穿透式风险分析 | `credit-review-expert/pre-visit-credit-analysis` | 拜访前对客户进行深度风险穿透分析 |
| 走访观察纪要 | `credit-review-expert/reviewer-visit-memo` | 信审视角的走访记录，侧重风险信号捕捉 |
| 准入规则扫描 | `credit-review-expert/admission-rules-scan` | 扫描信贷准入规则的合规性 |
| 进件合规检查 | `credit-review-expert/credit-case-intake-check` | 检查信贷进件材料的完整性与合规性 |
| 关联交易识别 | `credit-review-expert/credit-related-party-detection` | 识别客户关联交易与利益输送风险 |
| 押品风险评估 | `credit-review-expert/credit-collateral-risk-mgmt` | 评估押品价值与风险覆盖充足度 |

---

### AI理财经理（wealth-copilot）— 28 个技能

覆盖从商机发现到工作复盘的完整财富管理链路，按业务流程分为 8 层。

**L2-1 客户商机识别**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 客户KYC画像 | `wealth-copilot/L2-1_opportunity/client-kyc-profiling` | 构建客户风险偏好与资产状况的结构化画像 |
| 客户商机扫描 | `wealth-copilot/L2-1_opportunity/client-opportunity-scan` | 基于客户画像与市场动态识别交叉销售与追加销售机会 |
| 到期客户承接 | `wealth-copilot/L2-1_opportunity/maturity-client-alert` | 监控产品到期节点并生成承接方案 |

**L2-2 客户触达**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 客户关怀话术 | `wealth-copilot/L2-2_outreach/client-care-script` | 生成节日/生日/事件驱动的客户关怀沟通话术 |
| 主推产品卖点 | `wealth-copilot/L2-2_outreach/hot-product-briefing` | 提取主推产品核心卖点与适合客群匹配 |
| 市场事件解读 | `wealth-copilot/L2-2_outreach/market-event-interpreter` | 将重大市场事件转化为客户可理解的投资影响说明 |
| 晨会市场早报 | `wealth-copilot/L2-2_outreach/market-morning-brief` | 生成每日市场要闻速览供晨会使用 |
| 电话触达话术生成器 | `wealth-copilot/L2-2_outreach/outreach-script-generator` | 基于客户画像与沟通目的生成个性化电话沟通脚本 |
| 朋友圈内容创作 | `wealth-copilot/L2-2_outreach/wechat-moments-creator` | 生成专业且合规的社交媒体营销内容 |

**L2-3 销售策略**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 客户分群策略 | `wealth-copilot/L2-3_strategy/client-segmentation-strategy` | 基于多维标签生成客群细分与差异化服务策略 |
| 竞品对比分析 | `wealth-copilot/L2-3_strategy/competitor-product-compare` | 对比本行与竞品同类产品的优劣势 |
| 销售SOP导航 | `wealth-copilot/L2-3_strategy/sales-sop-navigator` | 根据当前销售阶段引导客户经理执行标准操作流程 |

**L2-4 知识问答**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 合规风控问答 | `wealth-copilot/L2-4_qa/compliance-risk-qa` | 回答销售过程中的合规与风控相关问题 |
| 市场行情问答 | `wealth-copilot/L2-4_qa/market-insight-qa` | 回答市场走势、行业动态等行情相关问题 |
| 产品知识问答 | `wealth-copilot/L2-4_qa/product-knowledge-qa` | 回答金融产品的条款、费率、风险等级等问题 |

**L2-5 持仓诊断**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 基金深度尽调 | `wealth-copilot/L2-5_diagnosis/fund-deep-research` | 对基金产品进行深度尽调分析（经理/持仓/业绩/风格） |
| 持仓健康诊断 | `wealth-copilot/L2-5_diagnosis/portfolio-health-check` | 评估客户投资组合的健康度与改进空间 |
| 组合风险雷达 | `wealth-copilot/L2-5_diagnosis/portfolio-risk-radar` | 多维度扫描投资组合的潜在风险暴露 |

**L2-6 资产配置**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 资产配置优化 | `wealth-copilot/L2-6_allocation/asset-allocation-optimizer` | 基于客户风险偏好与市场观点生成优化配置方案 |
| 家庭理财规划 | `wealth-copilot/L2-6_allocation/family-financial-planner` | 生成覆盖保障、教育、养老等目标的家庭综合理财规划 |
| 投资收益模拟 | `wealth-copilot/L2-6_allocation/investment-simulation` | 模拟不同配置方案在未来场景下的收益与回撤 |
| 产品智能匹配 | `wealth-copilot/L2-6_allocation/smart-product-matching` | 基于客户画像与需求智能匹配最适合的金融产品 |

**L2-7 工作复盘**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 日终工作复盘 | `wealth-copilot/L2-7_summary/daily-work-review` | 汇总当日工作成果与待办，生成结构化复盘报告 |
| 服务记录小结 | `wealth-copilot/L2-7_summary/service-summary-generator` | 将客户服务过程自动生成标准化服务记录 |
| 明日工作计划 | `wealth-copilot/L2-7_summary/tomorrow-plan-generator` | 基于今日待办与客户日程生成次日工作计划 |

**L2-8 伴随服务**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 投教知识科普 | `wealth-copilot/L2-8_companion/investor-education-qa` | 以通俗语言解答客户投资理财基础知识问题 |
| 市场热点速递 | `wealth-copilot/L2-8_companion/market-hotspot-digest` | 精炼推送当日市场热点事件及投资启示 |
| 持仓异动解读 | `wealth-copilot/L2-8_companion/portfolio-alert-narrator` | 将持仓异动信号转化为客户可理解的自然语言解读 |

---

### AI信贷风险管理专家（credit-risk-manager）— 8 个技能

覆盖从政策分析到贷后管理的信贷风控全周期。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 信贷政策环境分析 | `credit-risk-manager/credit-policy-analysis` | 分析当前信贷政策环境对业务的影响 |
| 行业风险规则生成 | `credit-risk-manager/credit-industry-rule-gen` | 基于行业特征生成信贷风险识别规则 |
| 风险分析框架提取 | `credit-risk-manager/credit-risk-extraction` | 从历史案例中提取可复用的风险分析框架 |
| 风险逆向思维链(CoT) | `credit-risk-manager/credit-risk-cot` | 通过逆向推理识别潜在风险盲区 |
| 跨模态材料核验 | `credit-risk-manager/vlm-verifier` | 跨文本与图像模态核验信贷材料真实性 |
| 贷后风险监测 | `credit-risk-manager/loan-risk-monitor` | 持续监测贷后客户风险信号 |
| 贷后全周期管理 | `credit-risk-manager/post-loan-management` | 管理贷后检查、分类、催收等全周期任务 |
| 大额风险暴露管理 | `credit-risk-manager/credit-large-exposure-mgmt` | 监控与管理大额信贷风险暴露集中度 |

---

### AI保险代理人（insurance-agent）— 12 个技能

覆盖从客户画像到续期维护的保险销售全流程。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 客户画像分析 | `insurance-agent/customer-profiling` | 构建保险客户的家庭结构与保障需求画像 |
| 保障缺口顾问 | `insurance-agent/gap-advisor` | 识别客户现有保障与理想保障之间的缺口 |
| 客户咨询顾问 | `insurance-agent/client-consulting` | 提供专业保险咨询服务与方案建议 |
| 计划书生成器 | `insurance-agent/plan-generator` | 生成个性化保险保障计划书 |
| 异议处理助手 | `insurance-agent/objection-handler` | 提供专业异议处理话术与应对策略 |
| 保单条款解读 | `insurance-agent/policy-explanation` | 用通俗语言解读保单条款与权益 |
| 健康告知指导 | `insurance-agent/health-disclosure-guide` | 指导客户完成健康告知并评估核保影响 |
| 投保资料核查 | `insurance-agent/application-check` | 核查投保材料的完整性与准确性 |
| 续期维护助手 | `insurance-agent/renewal-assist` | 辅助保单续期与客户续保沟通 |
| 展业活动计划器 | `insurance-agent/activity-planner` | 规划保险营销活动的主题、流程与话术 |
| 产品知识库 | `insurance-agent/product-wiki` | 提供保险产品条款、费率、对比等结构化知识 |
| 社媒营销 | `insurance-agent/social-media` | 生成合规的保险营销社交媒体内容 |

---

### AI保险理赔专家（claim-expert）— 8 个技能

覆盖从报案登记到结案通知的理赔全流程。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 理赔报案登记 | `claim-expert/claim-case-registration` | 登记理赔报案信息并完成案件建档 |
| 理赔材料分析 | `claim-expert/claim-document-processing` | 识别与结构化解析理赔提交的各类材料 |
| 理赔责任认定 | `claim-expert/coverage-analysis` | 分析保单条款认定保险责任与赔付范围 |
| 医学审查 | `claim-expert/medical-review` | 从医学专业角度审查理赔案件中的医疗信息 |
| 理赔理算 | `claim-expert/claim-adjustment` | 根据保单条款与案件事实计算赔付金额 |
| 理赔欺诈检测 | `claim-expert/claim-fraud-detection` | 识别理赔申请中的欺诈风险信号 |
| 核赔全案复审 | `claim-expert/claim-adjudication-review` | 对理赔全案进行最终审核与裁决 |
| 理赔结案通知 | `claim-expert/claim-notification` | 生成理赔结案通知与赔付说明 |

---

### AI保险核保专家（underwriting-expert）— 5 个技能

覆盖从健康核验到回访跟进的核保全流程。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 健康核验 | `underwriting-expert/health-verification` | 核验投保人健康告知与体检信息的一致性 |
| 医学评估 | `underwriting-expert/medical-assessment` | 评估投保人健康状况对核保决定的影响 |
| 双录质检 | `underwriting-expert/recording-inspection` | 检查销售双录（录音录像）的合规性 |
| 结论解读 | `underwriting-expert/conclusion-interpretation` | 解读核保结论并向客户/代理人说明 |
| 回访跟进 | `underwriting-expert/follow-up-outreach` | 执行新保回访与核保条件件跟进 |

---

### AI研究员（investment-researcher）— 34 个技能

按研究条线分层，覆盖行业、宏观、策略、金工、固收五大研究方向。

**行业研究员**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 全球财经速览 | `investment-researcher/global-finance-brief` | 生成全球金融市场动态速览 |
| 财报点评生成 | `investment-researcher/earnings-commentary-generator` | 基于上市公司财报生成专业点评 |
| 公告分析 | `investment-researcher/announcement-analysis` | 解读上市公司重大公告的投资影响 |
| 调研提纲生成 | `investment-researcher/institutional-research-outline` | 生成上市公司调研访谈提纲 |
| 公司深度分析 | `investment-researcher/company-deep-analysis` | 生成上市公司深度研究报告 |
| 公司一页纸分析 | `investment-researcher/company-one-page-analysis` | 生成精简的公司核心投资逻辑摘要 |
| 行业深度分析 | `investment-researcher/industry-deep-analysis` | 生成行业深度研究报告 |
| 行业一页纸分析 | `investment-researcher/industry-one-page-analysis` | 生成精简的行业核心趋势摘要 |

**宏观研究员**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 宏观数据速览 | `investment-researcher/macro-data-brief` | 解读核心宏观经济数据发布 |
| 宏观政策分析 | `investment-researcher/macro-policy-analysis` | 分析央行与政府部门政策动向及影响 |
| 宏观日报 | `investment-researcher/macro-daily-briefing` | 生成每日宏观经济与政策要闻简报 |
| 宏观周报 | `investment-researcher/macro-weekly-briefing` | 生成每周宏观经济与政策深度回顾 |
| 宏观风险监测 | `investment-researcher/macro-risk-monitor` | 监测宏观经济尾部风险与系统性风险信号 |
| 全球宏观联动 | `investment-researcher/global-macro-linkage` | 分析全球宏观经济事件的跨境传导影响 |
| 宏观资产配置 | `investment-researcher/macro-asset-allocation` | 基于宏观研判生成大类资产配置建议 |

**策略研究员**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 策略日报 | `investment-researcher/strategy-daily-briefing` | 生成每日市场策略与板块观点简报 |
| 板块配置建议 | `investment-researcher/sector-allocation` | 生成行业板块配置权重与轮动建议 |
| 市场情绪监测 | `investment-researcher/market-sentiment-monitor` | 监测市场情绪指标与极端情绪信号 |
| 市场趋势风格分析 | `investment-researcher/market-trend-style-analysis` | 分析当前市场趋势与风格特征 |
| 政策快评 | `investment-researcher/policy-flash-briefing` | 对重大政策发布进行即时快评 |
| 估值景气跟踪 | `investment-researcher/valuation-prosperity-tracking` | 跟踪行业估值与景气度匹配状态 |
| 市场极简复盘 | `investment-researcher/market-minimal-review` | 生成精简的日度市场复盘摘要 |

**金工研究员**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 金工日报 | `investment-researcher/quant-daily-brief` | 生成量化因子与策略表现每日简报 |
| 因子表现跟踪 | `investment-researcher/quant-factor-tracker` | 跟踪 Barra 风格因子与 Alpha 因子表现 |
| 策略快速回测 | `investment-researcher/quant-strategy-quick-backtest` | 对量化策略进行快速回测与绩效评估 |
| 另类因子挖掘 | `investment-researcher/alternative-factor-mining` | 从非结构化数据中挖掘新型 Alpha 因子 |
| 金工周报 | `investment-researcher/quant-weekly-brief` | 生成量化策略与因子表现每周深度回顾 |

**固收研究员**

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 隔夜资金面简报 | `investment-researcher/fixed-income-overnight-brief` | 生成隔夜资金面与利率动态简报 |
| 利率债复盘 | `investment-researcher/fixed-income-rate-review` | 复盘利率债市场走势与驱动因素 |
| 城投债风险分析 | `investment-researcher/lgt-debt-risk-analysis` | 分析城投平台信用风险与区域偿债能力 |
| 产业债风控 | `investment-researcher/industry-bond-risk-control` | 评估产业类信用债的行业与主体风险 |
| 可转债估值 | `investment-researcher/convertible-bond-valuation` | 对可转债进行定价分析与转股价值评估 |
| 固收日报 | `investment-researcher/fixed-income-daily-brief` | 生成固定收益市场每日综合简报 |
| 固收周报 | `investment-researcher/fixed-income-weekly-brief` | 生成固定收益市场每周深度回顾 |

---

### AI投资顾问（investment-advisor）— 9 个技能

覆盖股票与基金的行情分析、因子筛选与综合诊断。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 股票资金面分析 | `investment-advisor/stock-fund-analysis` | 分析个股资金流向与主力资金动向 |
| 股票行情分析 | `investment-advisor/stock-quote-analysis` | 解读个股实时行情与盘口特征 |
| 股票技术面分析 | `investment-advisor/stock-tech-analysis` | 基于技术指标分析个股趋势与关键位 |
| 股东股本查询 | `investment-advisor/stock-shareholder-analysis` | 查询个股股东结构与股本变动 |
| 股票多因子筛选 | `investment-advisor/stock-multi-factor-filter` | 基于多因子模型筛选满足条件的股票池 |
| 基金多因子筛选 | `investment-advisor/fund-multi-factor-filter` | 基于多因子模型筛选满足条件的基金池 |
| 可比公司分析 | `investment-advisor/comparable-company-analysis` | 识别可比公司并进行估值与经营指标对比 |
| 基金综合诊断 | `investment-advisor/fund-diagnosis` | 对基金产品进行多维度综合诊断评估 |
| A股市场热点发现 | `investment-advisor/a-market-hotspot-discovery` | 发现当前 A 股市场热点板块与主题投资机会 |

---

### AI金融工程专家（financial-engineering-expert）— 14 个技能

覆盖从数据探索到模型对比的完整建模流水线。

| Skill | Directory | What it does |
|-------|-----------|--------------|
| 数据轮廓速览 | `financial-engineering-expert/data-profiling` | 快速扫描数据集的基本统计分布与质量 |
| 单变量分析 | `financial-engineering-expert/univariate-analysis` | 分析单个变量的分布、缺失与预测能力 |
| 特征深度分析 | `financial-engineering-expert/feature-analysis` | 深度分析特征工程效果与变量交互关系 |
| XGBoost建模 | `financial-engineering-expert/xgb-modeling` | 执行 XGBoost 模型训练与评估 |
| XGBoost超参数调优 | `financial-engineering-expert/xgb-tuning` | 对 XGBoost 模型进行超参数搜索与优化 |
| LR评分卡建模 | `financial-engineering-expert/lr-modeling` | 执行逻辑回归评分卡建模与评估 |
| LR超参数调优 | `financial-engineering-expert/lr-tuning` | 对逻辑回归模型进行正则化与特征选择优化 |
| DNN深度学习建模 | `financial-engineering-expert/dnn-modeling` | 执行深度神经网络模型训练与评估 |
| DNN超参数调优 | `financial-engineering-expert/dnn-tuning` | 对深度神经网络进行结构与超参数优化 |
| 模型解释(SHAP) | `financial-engineering-expert/model-explanation` | 使用 SHAP 等方法解释模型预测逻辑 |
| 分群自主探索 | `financial-engineering-expert/segment-modeling` | 自动探索最优客群分群策略并分别建模 |
| 深度集成建模 | `financial-engineering-expert/xgb-deepmodel` | 构建多模型深度集成的融合预测模型 |
| 多模型效果对比 | `financial-engineering-expert/model-comparison` | 对比多个候选模型的性能指标与业务效果 |
| 自主实验循环 | `financial-engineering-expert/auto-experiment` | 自主设计并执行建模实验的闭环迭代 |

---

## Skill Definition Format

每个技能的核心文件 `SKILL.md` 遵循统一规范，YAML Frontmatter 声明元数据，正文定义执行逻辑。

```yaml
---
name: skill-name                    # 技能唯一标识
description: >                      # 技能描述与触发词
  一句话核心描述...触发词包括："关键词1"、"关键词2"
target_role: 角色名称                # 目标使用角色
business_domain: 业务域路径           # 业务域 > 子域 > 场景
risk_level: low|medium|high         # 风险等级
version: 1.0.0                      # 版本号
status: draft|review|published      # 状态
data_sources:                       # 数据源依赖
  - 数据项1
upstream_skills:                    # 上游技能（本技能的前置依赖）
  - skill-a
downstream_skills:                  # 下游技能（本技能的输出消费者）
  - skill-b
---
```

正文包含以下标准化章节：

| Section | Purpose |
|---------|---------|
| 约束条件 (Constraints) | 执行约束与金融合规红线 |
| 审计追踪 (Audit Trail) | 操作审计日志格式 |
| 执行流程 (Workflow) | 分步骤执行逻辑，每步标注 executor（ai/human）与 confirmation 机制 |
| 交互模式 (Interaction Pattern) | 对话交互阶段定义 |
| 输出格式 (Output Format) | 结构化输出规范与可解析字段 |
| 踩坑记录 (Gotchas) | 常见错误与解决方案 |
| 示例 (Examples) | 输入输出示例 |

---

## How It Fits Together

| Component | Role | Notes |
|-----------|------|-------|
| `SKILL.md` Frontmatter | 元数据与编排入口 | 触发词、依赖声明、风险等级，供 Agent 框架解析 |
| Workflow steps | 执行逻辑 | 每步标注 `executor: ai/human`，高风险步骤强制人工确认 |
| `upstream_skills` / `downstream_skills` | 技能编排 DAG | 支持链式调用、分支与并行编排 |
| `risk_level` | 合规守卫 | `high` 级别触发强制人工审核节点；`low` 级别可全自动执行 |
| `assets/` | 输出模板 | 结构化输出格式，确保下游系统可解析 |
| `tests/` | 质量验证 | 回归测试用例，防止技能定义退化 |
| `references/` | 知识注入 | 行业规范、监管指引、话术库，作为 RAG 或上下文注入来源 |

---

## Getting Started

```bash
# 查看某角色的技能列表
ls investment-researcher/

# 阅读某技能的完整定义
cat corporate-banker/visit-memo/SKILL.md

# 解析所有技能的 Frontmatter 元数据
grep -r "risk_level:" ./*/*/SKILL.md

# 查找所有高风险技能
grep -rl "risk_level: high" ./ --include="SKILL.md"

# 查看技能依赖关系
grep -A5 "upstream_skills:" corporate-banker/credit-due-diligence/SKILL.md
```

对接 Agent 框架的集成步骤：

1. 解析 Frontmatter —— 读取 `SKILL.md` 的 YAML 头部获取元数据（触发词、依赖关系、风险等级）
2. 加载执行流程 —— 解析 Workflow 章节，映射到 Agent 的工具调用链
3. 构建编排图 —— 利用 `upstream_skills` / `downstream_skills` 构建技能调用 DAG
4. 注入合规守卫 —— 根据 `risk_level` 与 Constraints 章节插入人工确认节点

创建新技能：

```bash
# 复制模板目录作为起点
cp -r wealth-copilot/_template/ wealth-copilot/new-skill-name/

# 编辑技能定义
$EDITOR wealth-copilot/new-skill-name/SKILL.md
```

