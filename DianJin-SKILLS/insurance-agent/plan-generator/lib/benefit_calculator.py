"""保险利益演示计算器 — 纯计算模块，不含任何 DOCX 逻辑。

每个函数返回统一格式:
    {"headers": [...], "rows": [[...]], "notes": [...]}

支持 CLI 独立测试:
    python benefit_calculator.py --type universal_life --params '{"annual_premium":10000,...}'
"""

import json
import argparse
import os
import sys

# 加载默认参数
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_PATH = os.path.join(_SCRIPT_DIR, "..", "templates", "benefit_models.json")

def _load_models():
    with open(_MODELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 工具函数
# ============================================================

def _fmt(v):
    """格式化金额为千分位字符串"""
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _milestone_years(payment_years, max_year, models=None):
    """生成里程碑年份列表"""
    if models is None:
        models = _load_models()
    base = models["projection_defaults"]["milestone_years"]
    years = sorted(set(base + [payment_years]))
    years = [y for y in years if 1 <= y <= max_year]
    if max_year not in years:
        years.append(max_year)
    return sorted(set(years))


# ============================================================
# 1. 寿险保障利益表 (定期寿/终身寿)
# ============================================================

def calc_life_table(annual_premium, sum_assured, payment_years, coverage_years,
                    insured_age, product_name="寿险", models=None):
    """计算寿险保障利益演示表。

    Args:
        annual_premium: 年缴保费(元)
        sum_assured: 身故保额(元)
        payment_years: 缴费年限
        coverage_years: 保障年限(如 30 或 999 表终身)
        insured_age: 被保人当前年龄
        product_name: 产品名称
    """
    if models is None:
        models = _load_models()

    if coverage_years >= 900:
        max_year = models["projection_defaults"]["life_expectancy"] - insured_age
    else:
        max_year = coverage_years
    max_year = max(max_year, payment_years + 5)

    milestones = _milestone_years(payment_years, max_year, models)
    cv_cfg = models["cash_value_curve"]

    headers = ["保单年度", "年龄", "累计已交保费", "身故保险金", "现金价值(估)"]
    rows = []

    for y in milestones:
        age = insured_age + y
        cum_premium = min(y, payment_years) * annual_premium
        death_benefit = max(sum_assured, cum_premium)

        # 估算现金价值
        ratio = _cv_ratio(y, payment_years, cv_cfg)
        cash_value = int(cum_premium * ratio)

        rows.append([
            f"第{y}年", age, _fmt(cum_premium), _fmt(death_benefit), _fmt(cash_value)
        ])

    notes = [
        f"产品: {product_name}",
        "现金价值为行业平均估算值，实际以保单年度末数值为准。",
        "身故保险金给付须符合合同约定的保险责任范围。",
    ]
    return {"headers": headers, "rows": rows, "notes": notes}


def _cv_ratio(year, payment_years, cv_cfg):
    """根据保单年度和参数配置返回现金价值/累计保费比率"""
    if year <= 0:
        return 0
    if year == 1:
        return cv_cfg.get("year_1", 0.15)
    if year == 2:
        return cv_cfg.get("year_2", 0.25)
    if year == 3:
        return cv_cfg.get("year_3", 0.40)
    if year <= 10:
        start = cv_cfg.get("year_4", 0.50)
        end = cv_cfg.get("year_10", 0.75)
        t = (year - 4) / 6
        return start + t * (end - start)
    if year <= 20:
        start = cv_cfg.get("year_11_20_start", 0.78)
        end = cv_cfg.get("year_11_20_end", 1.05)
        t = (year - 11) / 9
        return start + t * (end - start)
    # 20年以后
    base = cv_cfg.get("year_11_20_end", 1.05)
    growth = cv_cfg.get("year_20_plus_growth", 0.02)
    return base + (year - 20) * growth


# ============================================================
# 2. 重疾保障利益表
# ============================================================

def calc_ci_table(annual_premium, ci_benefit, payment_years, insured_age,
                  mild_ratio=None, product_name="重疾险", models=None):
    """计算重疾险保障利益演示表。

    Args:
        annual_premium: 年缴保费(元)
        ci_benefit: 重疾保额(元)
        payment_years: 缴费年限
        insured_age: 被保人当前年龄
        mild_ratio: 轻症赔付比例(如 0.3)，默认从配置读取
        product_name: 产品名称
    """
    if models is None:
        models = _load_models()
    if mild_ratio is None:
        mild_ratio = models.get("ci_mild_default_ratio", 0.3)

    max_year = models["projection_defaults"]["life_expectancy"] - insured_age
    max_year = max(max_year, payment_years + 5)
    milestones = _milestone_years(payment_years, max_year, models)
    cv_cfg = models["cash_value_curve"]

    mild_benefit = int(ci_benefit * mild_ratio)

    headers = ["保单年度", "年龄", "累计已交保费", "重疾给付", "轻症给付", "身故保险金"]
    rows = []

    breakeven_noted = False
    for y in milestones:
        age = insured_age + y
        cum_premium = min(y, payment_years) * annual_premium
        death_benefit = max(ci_benefit, cum_premium)

        year_label = f"第{y}年"
        if not breakeven_noted and cum_premium >= ci_benefit:
            year_label += " *"
            breakeven_noted = True

        rows.append([
            year_label, age, _fmt(cum_premium),
            _fmt(ci_benefit), _fmt(mild_benefit), _fmt(death_benefit)
        ])

    notes = [
        f"产品: {product_name}",
        f"轻症赔付按重疾保额的{int(mild_ratio * 100)}%计算（演示用，以合同为准）。",
        "标 * 年度表示累计已交保费达到重疾保额，此前保障杠杆更高。",
        "重大疾病保障范围以保险合同所列疾病定义为准。",
    ]
    return {"headers": headers, "rows": rows, "notes": notes}


# ============================================================
# 3. 年金/教育金领取表
# ============================================================

def calc_annuity_table(annual_premium, payment_years, start_receive_age,
                       annual_receive, receive_years, insured_age,
                       product_name="年金险", models=None):
    """计算年金/教育金领取演示表。

    Args:
        annual_premium: 年缴保费(元)
        payment_years: 缴费年限
        start_receive_age: 起领年龄
        annual_receive: 年领金额(元)
        receive_years: 领取年限(0=至终身)
        insured_age: 被保人当前年龄
        product_name: 产品名称
    """
    if models is None:
        models = _load_models()

    total_premium = annual_premium * payment_years
    life_exp = models["projection_defaults"]["life_expectancy"]

    if receive_years <= 0:
        end_age = life_exp
    else:
        end_age = start_receive_age + receive_years

    headers = ["年龄", "保单年度", "年度领取", "累计领取", "累计已交保费", "回本率"]
    rows = []

    # 积累期关键节点
    accum_years = start_receive_age - insured_age
    accum_milestones = [y for y in [1, 3, 5, 10, 15, 20] if y < accum_years]
    if accum_years > 0 and accum_years not in accum_milestones:
        accum_milestones.append(accum_years)

    for y in accum_milestones:
        age = insured_age + y
        cum_premium = min(y, payment_years) * annual_premium
        rows.append([age, f"第{y}年", "—", "—", _fmt(cum_premium), "—"])

    # 领取期
    cum_received = 0
    receive_milestones = list(range(start_receive_age, end_age + 1))
    # 精简: 只显示关键年份
    if len(receive_milestones) > 15:
        key_ages = set()
        key_ages.add(start_receive_age)
        key_ages.add(start_receive_age + 1)
        for step in [5, 10, 15, 20, 25]:
            a = start_receive_age + step
            if a <= end_age:
                key_ages.add(a)
        key_ages.add(end_age)
        # 教育金里程碑
        for ma in models.get("education_milestone_ages", []):
            if start_receive_age <= ma <= end_age:
                key_ages.add(ma)
        receive_milestones = sorted(key_ages)

    prev_age = start_receive_age - 1
    for age in receive_milestones:
        if age < start_receive_age:
            continue
        years_receiving = age - start_receive_age + 1
        cum_received = annual_receive * years_receiving
        y = age - insured_age
        cum_premium = min(y, payment_years) * annual_premium if y <= payment_years else total_premium
        payback = cum_received / total_premium * 100 if total_premium > 0 else 0

        rows.append([
            age, f"第{y}年", _fmt(annual_receive), _fmt(cum_received),
            _fmt(cum_premium if y <= payment_years else total_premium),
            f"{payback:.1f}%"
        ])

    notes = [
        f"产品: {product_name}",
        f"缴费期: {payment_years}年，年缴保费: {_fmt(annual_premium)}元",
        f"起领年龄: {start_receive_age}岁，年领: {_fmt(annual_receive)}元",
        "年金领取金额以保险合同约定为准。",
    ]
    return {"headers": headers, "rows": rows, "notes": notes}


# ============================================================
# 4. 万能账户三档演示
# ============================================================

def calc_universal_table(annual_premium, years, insured_age,
                         guaranteed_rate=None, mid_rate=None, high_rate=None,
                         product_name="万能险", models=None):
    """计算万能账户三档演示表。

    Args:
        annual_premium: 年缴保费(元)
        years: 演示年数
        insured_age: 被保人当前年龄
        guaranteed_rate: 最低保证利率
        mid_rate: 中档演示利率
        high_rate: 高档演示利率
        product_name: 产品名称
    """
    if models is None:
        models = _load_models()

    rates = models["universal_rates"]
    if guaranteed_rate is None:
        guaranteed_rate = rates["guaranteed"]
    if mid_rate is None:
        mid_rate = rates["mid"]
    if high_rate is None:
        high_rate = rates["high"]

    headers = [
        "保单年度", "年龄", "累计投入",
        f"低档({guaranteed_rate*100:.1f}%)", f"中档({mid_rate*100:.1f}%)",
        f"高档({high_rate*100:.1f}%)"
    ]
    rows = []

    low_val = mid_val = high_val = 0.0
    milestones = _milestone_years(years, years, models)
    if years not in milestones:
        milestones.append(years)
    milestones = sorted(set(milestones))

    for y in range(1, years + 1):
        cum_input = annual_premium * y
        low_val = (low_val + annual_premium) * (1 + guaranteed_rate)
        mid_val = (mid_val + annual_premium) * (1 + mid_rate)
        high_val = (high_val + annual_premium) * (1 + high_rate)

        if y in milestones:
            age = insured_age + y
            rows.append([
                f"第{y}年", age, _fmt(cum_input),
                _fmt(int(low_val)), _fmt(int(mid_val)), _fmt(int(high_val))
            ])

    notes = [
        f"产品: {product_name}",
        f"最低保证利率: {guaranteed_rate*100:.1f}%，中档演示: {mid_rate*100:.1f}%，高档演示: {high_rate*100:.1f}%",
        "中档、高档演示利率仅供参考，不代表对未来收益的承诺。",
        "实际结算利率由保险公司根据投资收益情况确定并公布。",
        "万能账户可能收取初始费用、保障成本等费用，本表为简化演示。",
    ]
    return {"headers": headers, "rows": rows, "notes": notes}


# ============================================================
# 5. 医疗险理赔场景演示
# ============================================================

def calc_medical_example(annual_limit, deductible, reimbursement_ratio,
                         example_claim=None, product_name="百万医疗险", **kwargs):
    """计算医疗险单次住院理赔场景演示。

    Args:
        annual_limit: 年度报销限额(元)
        deductible: 免赔额(元)
        reimbursement_ratio: 报销比例(如 1.0 = 100%)
        example_claim: 可选的理赔场景 dict:
            {"total_cost": 费用, "social_insurance_paid": 社保报销}
        product_name: 产品名称
    """
    if example_claim is None:
        example_claim = {"total_cost": 300000, "social_insurance_paid": 90000}

    total = example_claim["total_cost"]
    social = example_claim.get("social_insurance_paid", 0)
    remaining = total - social
    after_deductible = max(0, remaining - deductible)
    reimbursed = min(int(after_deductible * reimbursement_ratio), annual_limit)
    out_of_pocket = total - social - reimbursed

    headers = ["费用项目", "金额(元)", "说明"]
    rows = [
        ["住院总费用", _fmt(total), "含手术、药品、检查、床位等"],
        ["社保报销", _fmt(social), "基本医疗保险报销部分"],
        ["社保后自付", _fmt(remaining), "总费用 - 社保报销"],
        ["减去免赔额", _fmt(deductible), f"年度免赔额{_fmt(deductible)}元"],
        ["可报销金额", _fmt(after_deductible), "自付部分 - 免赔额"],
        [f"商业险报销({int(reimbursement_ratio*100)}%)", _fmt(reimbursed), "按报销比例计算"],
        ["最终自付", _fmt(out_of_pocket), "个人实际承担费用"],
    ]

    notes = [
        f"产品: {product_name}",
        f"年度报销限额: {_fmt(annual_limit)}元 | 免赔额: {_fmt(deductible)}元 | 报销比例: {int(reimbursement_ratio*100)}%",
        "以上为模拟理赔场景，实际理赔以医疗费用和合同条款为准。",
        f"本次场景可为客户节省 {_fmt(reimbursed)} 元医疗支出。",
    ]
    return {"headers": headers, "rows": rows, "notes": notes}


# ============================================================
# 6. 意外险保障一览
# ============================================================

def calc_accident_summary(death_benefit, disability_benefit_max,
                          medical_per_event=0, daily_hospital=0,
                          product_name="意外险", **kwargs):
    """计算意外险保障概览表。

    Args:
        death_benefit: 意外身故保额(元)
        disability_benefit_max: 意外伤残最高保额(元)
        medical_per_event: 意外医疗每次限额(元)
        daily_hospital: 住院津贴(元/天)
        product_name: 产品名称
    """
    headers = ["保障项目", "保障金额", "说明"]
    rows = [
        ["意外身故", _fmt(death_benefit) + "元", "因意外事故导致身故"],
        ["意外伤残", _fmt(disability_benefit_max) + "元", "按伤残等级比例给付(1-10级)"],
    ]
    if medical_per_event > 0:
        rows.append(["意外医疗", _fmt(medical_per_event) + "元/次", "每次意外事故医疗费用限额"])
    if daily_hospital > 0:
        rows.append(["住院津贴", _fmt(daily_hospital) + "元/天", "意外住院期间每日补贴"])

    notes = [
        f"产品: {product_name}",
        "意外事故定义及伤残等级标准以保险合同约定为准。",
        "高风险运动等除外责任详见合同免责条款。",
    ]
    return {"headers": headers, "rows": rows, "notes": notes}


# ============================================================
# 分发函数
# ============================================================

def calculate_benefit(product_type, benefit_details, product_name="",
                      annual_premium=0, insured_age=30, payment_years=20,
                      coverage_years=999, sum_assured=0, models=None):
    """根据产品类型分发到对应的计算函数。

    Args:
        product_type: 产品类型代码
        benefit_details: 产品利益详情 dict
        product_name: 产品名称
        annual_premium: 年缴保费
        insured_age: 被保人年龄
        payment_years: 缴费年限
        coverage_years: 保障年限
        sum_assured: 保额
    """
    if models is None:
        models = _load_models()

    bd = benefit_details or {}

    if product_type in ("term_life", "whole_life"):
        cov = 999 if product_type == "whole_life" else coverage_years
        return calc_life_table(
            annual_premium=annual_premium,
            sum_assured=bd.get("death_benefit", sum_assured),
            payment_years=payment_years,
            coverage_years=cov,
            insured_age=insured_age,
            product_name=product_name,
            models=models,
        )

    if product_type == "critical_illness":
        return calc_ci_table(
            annual_premium=annual_premium,
            ci_benefit=bd.get("ci_benefit", sum_assured),
            payment_years=payment_years,
            insured_age=insured_age,
            mild_ratio=bd.get("mild_ci_ratio"),
            product_name=product_name,
            models=models,
        )

    if product_type in ("annuity", "education_fund"):
        recv_years = bd.get("receive_years", 0)
        if product_type == "education_fund":
            recv_years = bd.get("receive_years", 7)
        return calc_annuity_table(
            annual_premium=annual_premium,
            payment_years=payment_years,
            start_receive_age=bd.get("start_receive_age", 60),
            annual_receive=bd.get("annual_receive", 0),
            receive_years=recv_years,
            insured_age=insured_age,
            product_name=product_name,
            models=models,
        )

    if product_type == "universal_life":
        return calc_universal_table(
            annual_premium=annual_premium,
            years=bd.get("projection_years", 30),
            insured_age=insured_age,
            guaranteed_rate=bd.get("guaranteed_rate"),
            mid_rate=bd.get("mid_rate"),
            high_rate=bd.get("high_rate"),
            product_name=product_name,
            models=models,
        )

    if product_type == "medical":
        return calc_medical_example(
            annual_limit=bd.get("annual_limit", 4000000),
            deductible=bd.get("deductible", 10000),
            reimbursement_ratio=bd.get("reimbursement_ratio", 1.0),
            example_claim=bd.get("example_claim"),
            product_name=product_name,
        )

    if product_type == "accident":
        return calc_accident_summary(
            death_benefit=bd.get("death_benefit", sum_assured),
            disability_benefit_max=bd.get("disability_benefit_max", sum_assured),
            medical_per_event=bd.get("medical_per_event", 0),
            daily_hospital=bd.get("daily_hospital", 0),
            product_name=product_name,
        )

    # 未知类型 — 返回简单摘要
    headers = ["项目", "内容"]
    rows = [[k, str(v)] for k, v in bd.items()]
    return {"headers": headers, "rows": rows, "notes": [f"产品: {product_name}"]}


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="保险利益演示计算器")
    parser.add_argument("--type", required=True,
                        help="产品类型: term_life/whole_life/critical_illness/annuity/education_fund/universal_life/medical/accident")
    parser.add_argument("--params", required=True, help="JSON 参数字符串")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = calculate_benefit(
        product_type=args.type,
        benefit_details=params.get("benefit_details", {}),
        product_name=params.get("product_name", args.type),
        annual_premium=params.get("annual_premium", 10000),
        insured_age=params.get("insured_age", 30),
        payment_years=params.get("payment_years", 20),
        coverage_years=params.get("coverage_years", 999),
        sum_assured=params.get("sum_assured", 0),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
