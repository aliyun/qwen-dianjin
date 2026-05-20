# 使用示例

## 示例1: Demo 模式 - 查看单个客户画像

**用户输入**: "帮我分析一下张伟这个客户"

**操作步骤**:
1. 技能自动进入 Demo 演示模式
2. 从 `demo-data/demo_customers.json` 加载数据
3. 定位客户 C001 (张伟)
4. 生成六维标签 + 保障缺口矩阵 + 展业策略

**执行命令**:
```bash
python scripts/analyze_profile.py --input demo-data/demo_customers.json --mode demo --customer C001
```

**预期输出**: 张伟的完整360度画像报告，包含：
- 35岁IT工程师，已婚育儿期
- 已有重疾30万+百万医疗
- 寿险、意外险、养老储备缺口较大
- 建议优先配置定期寿险和意外险

---

## 示例2: Demo 模式 - 批量分析客户池

**用户输入**: "帮我分析一下所有Demo客户的整体情况"

**操作步骤**:
1. 技能进入批量分析模式
2. 加载全部20条Demo数据
3. 生成汇总报告 + 每位客户独立画像

**执行命令**:
```bash
python scripts/analyze_profile.py --input demo-data/demo_customers.json --mode batch --output reports/
```

**预期输出**: reports/ 目录下生成：
- `batch_summary.md` - 客户池汇总报告
- `profile_C001_张伟.md` - 各客户独立画像

---

## 示例3: 用户数据模式 - 分析自有客户

**用户输入**: "我有一个客户数据文件 my_clients.json，帮我分析一下"

**操作步骤**:
1. 确认用户文件路径和格式
2. 校验数据字段完整性
3. 运行分析生成画像

**执行命令**:
```bash
python scripts/analyze_profile.py --input my_clients.json --mode user
```

---

## 示例4: 保险小白客户画像 (C005 陈磊)

**场景**: 代理人面访前3分钟准备

**输出亮点**:
- 30岁新婚期，零保单状态
- 五维全面缺口 -> 建议从"百万医疗+意外险+定期寿险"基础三件套入手
- 话术重点：以新婚责任为切入点，用小保费撬动大保障

---

## 示例5: 高净值客户画像 (C003 刘建国)

**场景**: 团队长分析VIP客户

**输出亮点**:
- 52岁企业主，年入120万
- 已有寿险500万+重疾50万+高端医疗+养老年金
- 核心需求：资产传承、企业与家庭资产隔离
- 策略：推保险金信托方案，维护关系促转介绍

---

## 数据格式要求

用户提供的客户数据应为 JSON 数组格式，每条记录包含以下字段：

### 必需字段
| 字段 | 类型 | 说明 |
|:---|:---|:---|
| name | string | 客户姓名 |
| age | number | 年龄 |
| gender | string | 性别 |
| occupation | string | 职业 |

### 可选字段（越完整分析越准确）
| 字段 | 类型 | 说明 |
|:---|:---|:---|
| id | string | 客户编号 |
| annual_income | number | 家庭年收入(元) |
| monthly_expense | number | 月均支出(元) |
| marital_status | string | 婚姻状况 |
| family_members | array | 家庭成员列表 |
| existing_policies | array | 已有保单列表 |
| claims_history | array | 理赔记录 |
| interaction_records | array | 交互记录 |
| risk_preference | string | 风险偏好 |
| notes | string | 备注 |
