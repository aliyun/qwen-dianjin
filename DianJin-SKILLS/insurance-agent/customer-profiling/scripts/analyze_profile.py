#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能客户画像分析 - 核心分析脚本
用于加载客户数据、生成标签、计算保障缺口并输出画像报告。

使用方式:
  python analyze_profile.py --input <数据文件> --mode <demo|user|batch>
  python analyze_profile.py --input demo-data/demo_customers.json --mode demo
  python analyze_profile.py --input demo-data/demo_customers.json --mode demo --customer C001
  python analyze_profile.py --input demo-data/demo_customers.json --mode batch --output reports/
"""

import json
import argparse
import io
import os
import sys
from datetime import datetime

# 修复 Windows 终端 UTF-8 输出问题
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ============================================================
# 数据加载
# ============================================================

def load_customers(file_path: str) -> list:
    """加载客户数据文件(JSON格式)"""
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    print(f"[信息] 成功加载 {len(data)} 条客户数据")
    return data


# ============================================================
# 标签生成引擎
# ============================================================

def tag_age_group(age: int) -> str:
    if age <= 25:
        return "青年期"
    elif age <= 35:
        return "成长期"
    elif age <= 45:
        return "黄金期"
    elif age <= 55:
        return "稳定期"
    elif age <= 65:
        return "预退休期"
    else:
        return "退休期"


def tag_family_structure(customer: dict) -> str:
    marital = customer.get("marital_status", "")
    members = customer.get("family_members", [])
    has_children = any(m.get("relation") == "子女" for m in members)
    has_parents = any(m.get("relation") == "父母" for m in members)

    if marital in ("离异", "丧偶") and has_children:
        return "单亲家庭"
    if marital == "未婚" and not has_children:
        return "单身"
    if marital == "已婚" and not has_children:
        return "二人世界"
    if has_parents:
        return "三代同堂"
    if has_children:
        return "核心家庭"
    return "其他"


def tag_life_stage(customer: dict) -> str:
    """优先使用数据中已有的 life_stage，否则自动推断"""
    if customer.get("life_stage"):
        return customer["life_stage"]
    age = customer.get("age", 0)
    marital = customer.get("marital_status", "")
    members = customer.get("family_members", [])
    children = [m for m in members if m.get("relation") == "子女"]
    minor_children = [c for c in children if c.get("age", 0) < 18]

    if marital == "未婚":
        return "单身期"
    if marital == "已婚" and not children:
        return "新婚期"
    if minor_children:
        return "育儿期"
    if age >= 60:
        return "养老期"
    if age >= 45:
        return "成熟期"
    return "成熟期"


def tag_risk_preference(customer: dict) -> str:
    return customer.get("risk_preference", "稳健型")


def tag_activity_level(customer: dict) -> str:
    return customer.get("activity_level", "中")


def tag_intent_level(customer: dict) -> str:
    """根据交互记录和响应率判断意向等级"""
    records = customer.get("interaction_records", [])
    response_rate = customer.get("response_rate", 0)
    has_consult = any("咨询" in r.get("type", "") or "咨询" in r.get("summary", "") for r in records)
    has_demand = any(kw in r.get("summary", "") for r in records for kw in ["了解", "考虑", "想", "需求", "规划"])

    if has_consult and has_demand and response_rate >= 0.7:
        return "高"
    elif has_consult or response_rate >= 0.5:
        return "中"
    else:
        return "低"


def generate_tags(customer: dict) -> dict:
    """生成客户六维标签"""
    return {
        "基础属性": {
            "年龄段": tag_age_group(customer.get("age", 0)),
            "职业类别": f"{customer.get('occupation_class', '未知')}职业",
            "家庭结构": tag_family_structure(customer),
            "地域": f"{customer.get('city', '')}({customer.get('region', '')})"
        },
        "保障现状": analyze_coverage_status(customer),
        "风险偏好": tag_risk_preference(customer),
        "生命周期": tag_life_stage(customer),
        "交互行为": {
            "活跃度": tag_activity_level(customer),
            "偏好渠道": customer.get("preferred_channel", "未知"),
            "响应率": f"{customer.get('response_rate', 0) * 100:.0f}%"
        },
        "意向等级": tag_intent_level(customer)
    }


# 六维标签解释说明映射表
TAG_EXPLANATIONS = {
    "年龄段": {
        "青年期": "18-25岁，职业起步阶段，保障意识待培养，预算有限",
        "成长期": "26-35岁，事业上升期，收入增长较快，保障需求开始显现",
        "黄金期": "36-45岁，事业与收入高峰期，家庭责任最重，保障需求最强",
        "稳定期": "46-55岁，收入趋于稳定，养老规划紧迫度上升",
        "预退休期": "56-65岁，临近退休，重点关注养老与医疗保障",
        "退休期": "65岁以上，已退休，以医疗保障和财富传承为主"
    },
    "家庭结构": {
        "单身": "无配偶无子女，保障责任集中在自身",
        "二人世界": "已婚无子女，双方互为经济依赖，需考虑对方保障",
        "核心家庭": "有配偶有子女的标准家庭，家庭支柱保障需求突出",
        "三代同堂": "上有老下有小，赡养+抚养双重压力，保障需求全面",
        "单亲家庭": "独自抚养子女，经济压力与保障需求更集中"
    },
    "风险偏好": {
        "保守型": "偏好确定性强的保障型产品，注重保本和稳定，决策谨慎",
        "稳健型": "兼顾保障与储蓄，关注性价比，接受适度的长期规划",
        "进取型": "对投资收益有期待，愿意尝试万能/投连/分红型产品"
    },
    "生命周期": {
        "单身期": "个人保障起步阶段，优先配置重疾+医疗+意外基础保障",
        "新婚期": "家庭初建，需考虑双方互保，可开始规划储蓄",
        "育儿期": "家庭责任最重时期，家庭支柱的寿险和重疾是刚需，教育金需求显现",
        "成熟期": "子女渐独立，养老规划进入关键窗口，资产保全需求上升",
        "养老期": "退休阶段，医疗和护理保障是核心，关注财富传承安排",
        "传承期": "高净值阶段，重点在资产隔离、税务筹划和保险金信托"
    },
    "活跃度": {
        "高": "近6个月互动>=3次且响应率>=70%，沟通意愿强，可加大触达频次",
        "中": "近6个月互动1-2次或响应率50-69%，需持续培育关系",
        "低": "近6个月互动少或响应率<50%，需重新激活，建议从关怀切入"
    },
    "意向等级": {
        "高": "主动咨询+明确需求+高响应率，属于近期可促成客户",
        "中": "有过产品咨询或正常响应，需进一步唤起需求",
        "低": "仅被动接触，无明确需求表达，需长期培育"
    }
}


# ============================================================
# 保障缺口分析引擎
# ============================================================

def analyze_coverage_status(customer: dict) -> dict:
    policies = customer.get("existing_policies", [])
    types = [p["type"] for p in policies]
    total_assured = sum(p.get("sum_assured", 0) for p in policies) / 10000
    return {
        "已购险种": types if types else ["无"],
        "已购险种数": len(types),
        "总保额": f"{total_assured:.0f}万",
        "缺口率": "待计算"
    }


def calculate_gap(customer: dict) -> dict:
    """计算五维保障缺口"""
    income = customer.get("annual_income", 0)
    policies = customer.get("existing_policies", [])
    age = customer.get("age", 0)

    def get_sum_by_type(t):
        return sum(p.get("sum_assured", 0) for p in policies if t in p.get("type", ""))

    # 寿险缺口
    life_suggested = income * 10
    life_existing = get_sum_by_type("寿险")
    life_gap = max(0, life_suggested - life_existing)

    # 重疾缺口
    ci_suggested = income * 5 + 300000
    ci_existing = get_sum_by_type("重疾")
    ci_gap = max(0, ci_suggested - ci_existing)

    # 医疗险缺口
    med_existing = get_sum_by_type("医疗")
    med_suggested = 1000000
    med_gap = max(0, med_suggested - med_existing)
    med_covered = med_existing >= med_suggested

    # 意外险缺口
    acc_suggested = income * 10
    acc_existing = get_sum_by_type("意外")
    acc_gap = max(0, acc_suggested - acc_existing)

    # 养老缺口 (简化: 月支出x12x20年 - 已有年金)
    monthly_expense = customer.get("monthly_expense", 0)
    retire_years = max(0, 80 - max(age, 60))
    pension_suggested = monthly_expense * 12 * retire_years
    pension_existing = get_sum_by_type("年金")
    pension_gap = max(0, pension_suggested - pension_existing)

    def gap_rate(suggested, existing):
        if suggested == 0:
            return 0
        return round((max(0, suggested - existing) / suggested) * 100, 1)

    def urgency(rate):
        if rate >= 80:
            return "高"
        elif rate >= 50:
            return "中"
        else:
            return "低"

    result = {
        "寿险": {
            "建议保额": round(life_suggested / 10000, 1),
            "已有保额": round(life_existing / 10000, 1),
            "缺口": round(life_gap / 10000, 1),
            "缺口率": gap_rate(life_suggested, life_existing),
            "紧迫度": urgency(gap_rate(life_suggested, life_existing))
        },
        "重疾险": {
            "建议保额": round(ci_suggested / 10000, 1),
            "已有保额": round(ci_existing / 10000, 1),
            "缺口": round(ci_gap / 10000, 1),
            "缺口率": gap_rate(ci_suggested, ci_existing),
            "紧迫度": urgency(gap_rate(ci_suggested, ci_existing))
        },
        "医疗险": {
            "建议保额": round(med_suggested / 10000, 1),
            "已有保额": round(med_existing / 10000, 1),
            "缺口": round(med_gap / 10000, 1),
            "缺口率": gap_rate(med_suggested, med_existing),
            "紧迫度": "低" if med_covered else urgency(gap_rate(med_suggested, med_existing)),
            "已覆盖": med_covered
        },
        "意外险": {
            "建议保额": round(acc_suggested / 10000, 1),
            "已有保额": round(acc_existing / 10000, 1),
            "缺口": round(acc_gap / 10000, 1),
            "缺口率": gap_rate(acc_suggested, acc_existing),
            "紧迫度": urgency(gap_rate(acc_suggested, acc_existing))
        },
        "养老储备": {
            "建议保额": round(pension_suggested / 10000, 1),
            "已有保额": round(pension_existing / 10000, 1),
            "缺口": round(pension_gap / 10000, 1),
            "缺口率": gap_rate(pension_suggested, pension_existing),
            "紧迫度": urgency(gap_rate(pension_suggested, pension_existing))
        }
    }

    # 计算综合缺口率
    total_suggested = life_suggested + ci_suggested + med_suggested + acc_suggested + pension_suggested
    total_existing = life_existing + ci_existing + med_existing + acc_existing + pension_existing
    result["综合缺口率"] = gap_rate(total_suggested, total_existing)

    return result


# ============================================================
# 画像报告生成
# ============================================================

def render_bar(value, max_val=10, width=20):
    filled = int((value / max_val) * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def generate_profile_report(customer: dict) -> str:
    """生成单个客户的360度画像报告"""
    tags = generate_tags(customer)
    gaps = calculate_gap(customer)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 基础信息
    members_str = ", ".join(
        f"{m['relation']}{m['name']}({m.get('age', '?')}岁/{m.get('occupation', '未知')})"
        for m in customer.get("family_members", [])
    ) or "无"

    # 保障完善度评分
    policies = customer.get("existing_policies", [])
    types = set(p.get("type", "") for p in policies)
    score_map = {"寿险": 2, "重疾": 2, "医疗": 2, "意外": 2, "年金": 2}
    coverage_score = 0
    for key in score_map:
        if any(key in t for t in types):
            coverage_score += score_map[key]

    # 六维评分
    intent = tags["意向等级"]
    intent_score = {"高": 9, "中": 6, "低": 3}.get(intent, 5)
    activity = tags["交互行为"]["活跃度"]
    activity_score = {"高": 9, "中": 6, "低": 3}.get(activity, 5)

    # 标签解释查找辅助函数
    def get_explanation(dimension, value):
        if dimension in TAG_EXPLANATIONS and value in TAG_EXPLANATIONS[dimension]:
            return TAG_EXPLANATIONS[dimension][value]
        return ""

    # 各维度解释
    age_tag = tags['基础属性']['年龄段']
    family_tag = tags['基础属性']['家庭结构']
    risk_tag = tags['风险偏好']
    stage_tag = tags['生命周期']
    activity_tag = tags['交互行为']['活跃度']
    intent_tag = tags['意向等级']

    report = f"""# 客户360度画像报告 -- {customer['name']}

> 生成时间: {now} | 客户编号: {customer.get('id', 'N/A')}

---

## 一、基础信息

| 项目 | 内容 |
|:---|:---|
| 姓名 | {customer['name']} |
| 性别 | {customer.get('gender', '未知')} |
| 年龄 | {customer.get('age', '未知')}岁 |
| 职业 | {customer.get('occupation', '未知')}（{customer.get('occupation_class', '?')}职业） |
| 家庭年收入 | {customer.get('annual_income', 0) / 10000:.1f}万元 |
| 月均支出 | {customer.get('monthly_expense', 0)}元 |
| 所在城市 | {customer.get('city', '')}（{customer.get('region', '')}） |
| 婚姻状况 | {customer.get('marital_status', '未知')} |"""

    # 家庭成员详细表格
    family = customer.get("family_members", [])
    if family:
        report += """

### 家庭成员

| 关系 | 姓名 | 年龄 | 职业 |
|:---|:---|:---|:---|"""
        for m in family:
            m_income = f"，年收入{m['annual_income']/10000:.0f}万" if m.get("annual_income") else ""
            report += f"\n| {m.get('relation', '')} | {m.get('name', '')} | {m.get('age', '?')}岁 | {m.get('occupation', '未知')}{m_income} |"
    else:
        report += "\n| 家庭成员 | 无 |"

    report += f"""

---

## 二、六维标签体系

### 标签雷达图

```
基础属性: {render_bar(8)}  8/10
保障现状: {render_bar(coverage_score)}  {coverage_score}/10
风险偏好: {render_bar(7)}  7/10
生命周期: {render_bar(8)}  8/10
交互行为: {render_bar(activity_score)}  {activity_score}/10
意向等级: {render_bar(intent_score)}  {intent_score}/10
```

### 标签明细与解读

| 维度 | 标签值 | 解读说明 |
|:---|:---|:---|
| 基础属性-年龄段 | {age_tag} | {get_explanation('年龄段', age_tag)} |
| 基础属性-职业 | {tags['基础属性']['职业类别']} | {customer.get('occupation', '')}，职业风险{customer.get('occupation_class', '?')} |
| 基础属性-家庭结构 | {family_tag} | {get_explanation('家庭结构', family_tag)} |
| 基础属性-地域 | {tags['基础属性']['地域']} | 所在区域经济与保险市场特征 |
| 保障现状 | 已购{tags['保障现状']['已购险种数']}险种，总保额{tags['保障现状']['总保额']} | 综合缺口率{gaps['综合缺口率']}%，{'保障较完善' if gaps['综合缺口率'] < 30 else '存在明显缺口需关注' if gaps['综合缺口率'] < 70 else '保障严重不足，需尽快规划'} |
| 风险偏好 | {risk_tag} | {get_explanation('风险偏好', risk_tag)} |
| 生命周期 | {stage_tag} | {get_explanation('生命周期', stage_tag)} |
| 交互行为-活跃度 | {activity_tag} | {get_explanation('活跃度', activity_tag)} |
| 交互行为-渠道 | 偏好{tags['交互行为']['偏好渠道']} | 建议优先通过{tags['交互行为']['偏好渠道']}触达，响应率{tags['交互行为']['响应率']} |
| 意向等级 | {intent_tag} | {get_explanation('意向等级', intent_tag)} |"""

    # 已有保单明细
    policies = customer.get("existing_policies", [])
    if policies:
        report += """

### 已有保单明细

| 险种 | 产品名称 | 保额(万) | 年缴保费(元) | 缴费年限 | 起保日期 | 状态 |
|:---|:---|:---|:---|:---|:---|:---|"""
        total_premium = 0
        for p in policies:
            sa = p.get('sum_assured', 0) / 10000
            premium = p.get('annual_premium', 0)
            total_premium += premium
            report += f"\n| {p.get('type', '')} | {p.get('product_name', '')} | {sa:.0f} | {premium:,} | {p.get('payment_years', '')}年 | {p.get('start_date', '')} | {p.get('status', '')} |"
        report += f"\n| **合计** | | | **{total_premium:,}** | | | |"
    else:
        report += "\n\n### 已有保单\n\n> 暂无已购保单记录（零保单客户）"

    report += f"""

---

## 三、保障缺口分析

### 五维缺口矩阵

| 保障维度 | 建议保额(万) | 已有保额(万) | 缺口(万) | 缺口率 | 紧迫度 |
|:---|:---|:---|:---|:---|:---|"""

    for dim in ["寿险", "重疾险", "医疗险", "意外险", "养老储备"]:
        g = gaps[dim]
        report += f"\n| {dim} | {g['建议保额']} | {g['已有保额']} | {g['缺口']} | {g['缺口率']}% | {g['紧迫度']} |"

    # 缺口可视化
    report += "\n\n### 缺口可视化\n\n```"
    for dim in ["寿险", "重疾险", "医疗险", "意外险", "养老储备"]:
        g = gaps[dim]
        covered_pct = 100 - g["缺口率"]
        bar_filled = int(covered_pct / 5)
        bar_empty = 20 - bar_filled
        label = f"{dim}".ljust(6, "\u3000")
        if g["缺口率"] == 0:
            report += f"\n{label} |{'=' * 20}| 已覆盖"
        else:
            report += f"\n{label} |{'=' * bar_filled}{' ' * bar_empty}| 缺口 {g['缺口']}万"
    report += "\n```"

    # 备注
    notes = customer.get("notes", "")
    if notes:
        report += f"\n\n### 代理人备注\n\n> {notes}"

    # 理赔历史
    claims = customer.get("claims_history", [])
    if claims:
        report += "\n\n### 理赔记录\n\n| 日期 | 类型 | 金额 | 原因 | 状态 |\n|:---|:---|:---|:---|:---|"
        for c in claims:
            report += f"\n| {c.get('date', '')} | {c.get('type', '')} | {c.get('amount', 0)}元 | {c.get('reason', '')} | {c.get('status', '')} |"

    report += f"""

---

## 四、客户旅程时间轴

"""
    # 收集所有时间事件
    timeline_events = []

    # 保单投保事件
    for p in customer.get("existing_policies", []):
        if p.get("start_date"):
            timeline_events.append({
                "date": p["start_date"],
                "icon": "[投保]",
                "category": "保单",
                "detail": f"投保{p.get('type', '')}「{p.get('product_name', '')}」，保额{p.get('sum_assured', 0)/10000:.0f}万，年缴{p.get('annual_premium', 0):,}元"
            })

    # 理赔事件
    for c in customer.get("claims_history", []):
        if c.get("date"):
            timeline_events.append({
                "date": c["date"],
                "icon": "[理赔]",
                "category": "理赔",
                "detail": f"{c.get('type', '')}理赔{c.get('amount', 0):,}元，原因：{c.get('reason', '')}（{c.get('status', '')}）"
            })

    # 交互记录事件
    for r in customer.get("interaction_records", []):
        if r.get("date"):
            timeline_events.append({
                "date": r["date"],
                "icon": "[互动]",
                "category": "互动",
                "detail": f"通过{r.get('channel', '')}进行{r.get('type', '')}：{r.get('summary', '')}"
            })

    # 生命周期推测事件（基于家庭成员年龄反推关键节点）
    current_year = datetime.now().year
    age = customer.get("age", 0)
    birth_year = current_year - age
    # 推测的里程碑
    marital = customer.get("marital_status", "")
    family = customer.get("family_members", [])
    children = [m for m in family if m.get("relation") == "子女"]

    if marital in ("已婚", "离异", "丧偶") and children:
        # 用最大子女年龄反推结婚大约时间（假设婚后1-2年生育）
        oldest_child_age = max(c.get("age", 0) for c in children)
        marry_year = current_year - oldest_child_age - 2
        if marry_year > birth_year + 18:
            timeline_events.append({
                "date": f"{marry_year}-01-01",
                "icon": "[里程碑]",
                "category": "生活",
                "detail": f"推测约{marry_year}年结婚（基于家庭信息推算）"
            })
        for child in children:
            child_age = child.get("age", 0)
            child_birth_year = current_year - child_age
            timeline_events.append({
                "date": f"{child_birth_year}-01-01",
                "icon": "[里程碑]",
                "category": "生活",
                "detail": f"{child.get('relation', '')}{child.get('name', '')}出生（现{child_age}岁，{child.get('occupation', '')}）"
            })
    elif marital == "已婚" and not children:
        # 新婚无子女，不做过多推测
        pass

    # 按日期排序
    timeline_events.sort(key=lambda x: x["date"])

    if timeline_events:
        report += "```\n"
        prev_year = ""
        for i, evt in enumerate(timeline_events):
            year = evt["date"][:4]
            date_short = evt["date"][:10]

            # 年份分隔
            if year != prev_year:
                if prev_year:
                    report += "  |\n"
                report += f"  {'=' * 6} {year}年 {'=' * 6}\n"
                prev_year = year

            # 连接线
            is_last = (i == len(timeline_events) - 1)
            connector = "  *" if is_last else "  |"

            report += f"  |\n"
            report += f"  o-- {date_short}  {evt['icon']} {evt['detail']}\n"

        report += "  |\n"
        report += f"  v  (当前: {datetime.now().strftime('%Y-%m')})\n"
        report += "```\n"

        # 旅程摘要
        policy_count = sum(1 for e in timeline_events if e["category"] == "保单")
        claim_count = sum(1 for e in timeline_events if e["category"] == "理赔")
        interact_count = sum(1 for e in timeline_events if e["category"] == "互动")
        life_count = sum(1 for e in timeline_events if e["category"] == "生活")

        report += f"\n> 旅程统计：共{len(timeline_events)}个关键节点"
        parts = []
        if life_count:
            parts.append(f"生活里程碑{life_count}个")
        if policy_count:
            parts.append(f"投保{policy_count}次")
        if claim_count:
            parts.append(f"理赔{claim_count}次")
        if interact_count:
            parts.append(f"互动{interact_count}次")
        if parts:
            report += f"（{'、'.join(parts)}）"
    else:
        report += "> 暂无可追溯的客户旅程事件\n"

    report += f"""

---

## 五、近期交互记录

| 日期 | 渠道 | 类型 | 摘要 |
|:---|:---|:---|:---|"""
    for r in customer.get("interaction_records", [])[-5:]:
        report += f"\n| {r.get('date', '')} | {r.get('channel', '')} | {r.get('type', '')} | {r.get('summary', '')} |"

    report += """

---

> 免责声明: 本报告由AI辅助生成，仅供参考。最终产品推荐与服务方案需由持牌代理人结合客户实际情况审核确认。
"""
    return report


def generate_batch_summary(customers: list) -> str:
    """生成批量客户的汇总分析报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(customers)

    # 统计维度
    stage_count = {}
    risk_count = {}
    intent_count = {}
    gap_sum = {"寿险": 0, "重疾险": 0, "医疗险": 0, "意外险": 0, "养老储备": 0}
    high_intent = []

    for c in customers:
        tags = generate_tags(c)
        gaps = calculate_gap(c)

        stage = tags["生命周期"]
        stage_count[stage] = stage_count.get(stage, 0) + 1

        risk = tags["风险偏好"]
        risk_count[risk] = risk_count.get(risk, 0) + 1

        intent = tags["意向等级"]
        intent_count[intent] = intent_count.get(intent, 0) + 1

        if intent == "高":
            high_intent.append(c["name"])

        for dim in gap_sum:
            gap_sum[dim] += gaps[dim]["缺口"]

    report = f"""# 客户池批量分析汇总报告

> 生成时间: {now} | 客户总数: {total}

---

## 一、客户池概览

### 生命周期分布
| 阶段 | 数量 | 占比 |
|:---|:---|:---|"""
    for stage, count in sorted(stage_count.items(), key=lambda x: -x[1]):
        report += f"\n| {stage} | {count} | {count/total*100:.1f}% |"

    report += """

### 风险偏好分布
| 类型 | 数量 | 占比 |
|:---|:---|:---|"""
    for risk, count in sorted(risk_count.items(), key=lambda x: -x[1]):
        report += f"\n| {risk} | {count} | {count/total*100:.1f}% |"

    report += """

### 意向等级分布
| 等级 | 数量 | 占比 |
|:---|:---|:---|"""
    for level, count in sorted(intent_count.items(), key=lambda x: -x[1]):
        report += f"\n| {level} | {count} | {count/total*100:.1f}% |"

    report += f"""

---

## 二、保障缺口汇总

### 客户池平均缺口

| 维度 | 平均缺口(万) |
|:---|:---|"""
    for dim, total_gap in gap_sum.items():
        report += f"\n| {dim} | {total_gap/total:.1f} |"

    report += f"""

---

## 三、高意向客户清单

共 {len(high_intent)} 位高意向客户: {', '.join(high_intent) if high_intent else '无'}

---

## 四、团队行动建议

1. **重点跟进**: 优先对接高意向客户，制定个性化沟通方案
2. **共性缺口**: 根据平均缺口数据，组织对应产品的专题培训
3. **分群经营**: 按生命周期阶段分群，制定差异化经营策略
4. **数据完善**: 对标签数据缺失的客户，安排补充信息采集

---

> 免责声明: 本报告由AI辅助生成，仅供参考。
"""
    return report


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="智能客户画像分析工具")
    parser.add_argument("--input", required=True, help="客户数据文件路径(JSON)")
    parser.add_argument("--mode", choices=["demo", "user", "batch"], default="demo", help="运行模式")
    parser.add_argument("--customer", help="指定客户ID(单客户分析)")
    parser.add_argument("--output", help="输出目录(batch模式)")

    args = parser.parse_args()
    customers = load_customers(args.input)

    if args.customer:
        # 单客户分析
        target = [c for c in customers if c.get("id") == args.customer]
        if not target:
            print(f"[错误] 未找到客户ID: {args.customer}")
            print(f"[提示] 可用的客户ID: {', '.join(c.get('id', '') for c in customers)}")
            sys.exit(1)
        report = generate_profile_report(target[0])
        print(report)

    elif args.mode == "batch":
        # 批量分析
        output_dir = args.output or "reports"
        os.makedirs(output_dir, exist_ok=True)

        # 汇总报告
        summary = generate_batch_summary(customers)
        summary_path = os.path.join(output_dir, "batch_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[信息] 汇总报告已生成: {summary_path}")

        # 每个客户的独立报告
        for c in customers:
            report = generate_profile_report(c)
            cid = c.get("id", "unknown")
            file_path = os.path.join(output_dir, f"profile_{cid}_{c['name']}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"[信息] 客户画像已生成: {file_path}")

        print(f"\n[完成] 共生成 {len(customers)} 份客户画像 + 1 份汇总报告")

    else:
        # 默认: 分析第一个客户（演示）
        report = generate_profile_report(customers[0])
        print(report)


if __name__ == "__main__":
    main()
