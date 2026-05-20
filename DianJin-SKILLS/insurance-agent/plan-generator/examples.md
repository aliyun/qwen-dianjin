# 使用示例与数据格式

## 示例1: 技能联动模式

先使用 `insurance-customer-profiling` 生成客户画像，再使用 `product-combo-pitch` 生成产品推荐，最后生成计划书：

```bash
# 1. 客户画像分析（得到画像 JSON）
python ../insurance-customer-profiling/scripts/analyze_profile.py --input clients.json --mode user --customer "张伟"

# 2. 产品推荐（得到推荐 JSON）
python ../product-combo-pitch/scripts/recommender.py --profile profile_output.json --products ../product-combo-pitch/templates/sample_product_library.json --top-k 3

# 3. 生成计划书
python scripts/generate_plan.py --profile profile_output.json --recommendation recommendations.json --agent agent.json --output 张伟_保障规划方案.docx
```

## 示例2: 独立输入模式

直接使用完整的 JSON 文件生成计划书：

```bash
python scripts/generate_plan.py --input demo-data/demo_plan_input.json --output demo_计划书.docx
```

## 示例3: 指定章节

只生成封面、产品方案和利益演示：

```bash
python scripts/generate_plan.py --input demo-data/demo_plan_input.json --sections cover,products,benefits --output 精简版计划书.docx
```

## 数据格式要求

### 统一输入 JSON 结构

```json
{
  "plan_config": {
    "plan_title": "可选，默认为 '{客户姓名} 家庭保障规划方案'",
    "plan_date": "2026-04-18",
    "plans_count": 2,
    "include_sections": ["cover","needs","gap","products","benefits","budget","compliance","agent"]
  },
  "customer": {
    "name": "张伟(必填)",
    "gender": "男(必填)",
    "age": 35,
    "occupation": "IT工程师",
    "annual_income": 350000,
    "monthly_expense": 15000,
    "city": "上海",
    "marital_status": "已婚",
    "life_stage": "育儿期",
    "risk_preference": "稳健型",
    "family_members": [
      {"relation": "配偶", "name": "李娜", "age": 33, "occupation": "教师"}
    ],
    "existing_policies": [
      {"type": "重疾险", "product_name": "XX重疾保", "sum_assured": 300000, "annual_premium": 8500, "payment_years": 20, "status": "缴费中"}
    ]
  },
  "gap_analysis": "可选，未提供时自动计算",
  "product_plans": {
    "plan_a": {
      "plan_label": "基础保障方案",
      "products": [
        {
          "product_name": "安心定期寿险(必填)",
          "product_type": "term_life(必填，见下表)",
          "insured_person": "张伟",
          "sum_assured": 2000000,
          "annual_premium": 2400,
          "payment_period": "20年",
          "coverage_period": "至60周岁",
          "benefit_details": {}
        }
      ]
    },
    "plan_b": "可选，格式同 plan_a"
  },
  "narratives": {
    "needs_narrative": "可选，LLM 生成的客户需求分析叙述",
    "gap_narrative": "可选，LLM 生成的缺口分析解读",
    "recommendation_narrative": "可选，LLM 生成的推荐说明"
  },
  "agent": {
    "name": "王明(必填)",
    "company": "XX保险(必填)",
    "phone": "138-0000-1234(必填)",
    "title": "资深理财规划师",
    "license_number": "02001234567",
    "wechat": "wangming",
    "email": "wangming@example.com"
  }
}
```

### 产品类型代码 (product_type)

| 代码 | 险种 | benefit_details 关键字段 |
|:---|:---|:---|
| term_life | 定期寿险 | death_benefit |
| whole_life | 终身寿险 | death_benefit |
| critical_illness | 重疾险 | ci_benefit, mild_ci_ratio |
| medical | 医疗险 | annual_limit, deductible, reimbursement_ratio, example_claim |
| accident | 意外险 | death_benefit, disability_benefit_max, medical_per_event, daily_hospital |
| annuity | 年金险 | start_receive_age, annual_receive, receive_years |
| education_fund | 教育金 | start_receive_age, annual_receive, receive_years |
| universal_life | 万能险 | guaranteed_rate, mid_rate, high_rate, projection_years |

### benefit_details 示例

**重疾险**:
```json
{"ci_benefit": 500000, "mild_ci_ratio": 0.3}
```

**医疗险**:
```json
{"annual_limit": 4000000, "deductible": 10000, "reimbursement_ratio": 1.0, "example_claim": {"total_cost": 300000, "social_insurance_paid": 90000}}
```

**年金险**:
```json
{"start_receive_age": 60, "annual_receive": 48000, "receive_years": 0}
```

**万能险**:
```json
{"guaranteed_rate": 0.02, "mid_rate": 0.035, "high_rate": 0.045, "projection_years": 30}
```
