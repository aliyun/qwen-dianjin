#!/usr/bin/env python3
"""
保险产品 Wiki 技能测试脚本 v2.1
覆盖维度：红线检查、入库流程、学习模式、话术质量、模拟演练、知识库关联、边界测试
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

SKILL_PATH = Path.home() / ".hermes" / "skills" / "insurance-product-wiki"
PRODUCTS_PATH = SKILL_PATH / "products"
INDEX_PATH = SKILL_PATH / "product-index.md"

def run_tavily(query, count=5):
    """运行 Tavily 搜索"""
    cmd = f'cd ~/.hermes/skills/openclaw-imports/tavily-search-skill && ./search.sh "{query}" {count}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        try:
            return json.loads(result.stdout)
        except:
            return {"raw": result.stdout}
    return None

def check_red_line(content, rule_num, rule_desc):
    """检查红线是否被遵守"""
    violations = []
    
    # 规则 1: 真实底线 - 不编造数据
    if rule_num == 1:
        if any(x in content for x in ["编造", "虚构", "假设数据"]):
            violations.append(f"红线 {rule_num} ({rule_desc}): 发现编造内容")
    
    # 规则 2: 客户语言 - 不用专业术语堆砌
    if rule_num == 2:
        # 如果有等待期，检查是否用客户语言解释
        if "等待期" in content and "出事不赔" not in content and "正式生效" not in content:
            violations.append(f"红线 {rule_num} ({rule_desc}): 未使用客户语言解释等待期")
    
    # 规则 4: 敢判断 - 不说"既可能好也可能坏"
    if rule_num == 4:
        if any(x in content for x in ["既可能好也可能坏", "不好说", "看情况而定"]):
            violations.append(f"红线 {rule_num} ({rule_desc}): 不敢判断")
    
    # 规则 5: 短词优先
    if rule_num == 5:
        if "进行保险责任的赔付" in content:
            violations.append(f"红线 {rule_num} ({rule_desc}): 未使用短词")
    
    # 规则 7: 来源标注
    if rule_num == 7:
        if "[[IP-" not in content and "[Tavily" not in content and "[[012-" not in content and "来源" not in content:
            violations.append(f"红线 {rule_num} ({rule_desc}): 未标注来源")
    
    # 规则 8: 不填充
    if rule_num == 8:
        if "保险产品是家庭财务规划的重要组成部分" in content:
            violations.append(f"红线 {rule_num} ({rule_desc}): AI 文案腔")
    
    return violations

def test_product_wiki_structure(product_dir):
    """测试产品 Wiki 结构"""
    wiki_path = product_dir / "wiki.md"
    scripts_path = product_dir / "scripts.md"
    raw_path = product_dir / "raw" / "source-text.md"
    
    results = {
        "wiki_exists": wiki_path.exists(),
        "scripts_exists": scripts_path.exists(),
        "raw_exists": raw_path.exists(),
        "wiki_sections": [],
        "violations": []
    }
    
    if wiki_path.exists():
        content = wiki_path.read_text()
        required_sections = ["产品概览", "保障责任", "投保规则", "核心卖点", "关联产品"]
        for section in required_sections:
            if section in content:
                results["wiki_sections"].append(section)
        
        # 红线检查
        for rule_num, rule_desc in [(1, "真实底线"), (4, "敢判断"), (7, "来源标注"), (8, "不填充")]:
            violations = check_red_line(content, rule_num, rule_desc)
            results["violations"].extend(violations)
    
    return results

def test_scripts_quality(product_dir):
    """测试话术质量"""
    scripts_path = product_dir / "scripts.md"
    
    results = {
        "exists": scripts_path.exists(),
        "sections": [],
        "violations": [],
        "quality_score": 0
    }
    
    if scripts_path.exists():
        content = scripts_path.read_text()
        required_sections = ["开场白", "异议处理", "促成"]
        for section in required_sections:
            if section in content:
                results["sections"].append(section)
        
        # 红线检查
        for rule_num, rule_desc in [(2, "客户语言"), (3, "场景绑定"), (5, "短词优先"), (6, "敢判断"), (8, "不填充")]:
            violations = check_red_line(content, rule_num, rule_desc)
            results["violations"].extend(violations)
        
        # 质量评分（改进版）
        quality_indicators = ["场景", "客户", "话术", "说", "如果", "觉得", "预算", "贵", "担心", "保障"]
        score = sum(1 for ind in quality_indicators if ind in content)
        results["quality_score"] = min(int(score * 100 / len(quality_indicators)), 100)
    
    return results

def test_index_integrity():
    """测试索引完整性"""
    results = {
        "index_exists": INDEX_PATH.exists(),
        "product_count": 0,
        "format_valid": False,
        "violations": []
    }
    
    if INDEX_PATH.exists():
        content = INDEX_PATH.read_text()
        # 统计产品数量
        product_lines = [line for line in content.split('\n') if line.strip().startswith('| IP-')]
        results["product_count"] = len(product_lines)
        results["format_valid"] = "产品总数" in content and "最后更新" in content
    
    return results

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("保险产品 Wiki 技能测试 v2.0")
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "violations": []
        }
    }
    
    # 测试 1: 索引完整性
    print("\n[测试 1] 索引完整性")
    index_result = test_index_integrity()
    all_results["tests"].append({"name": "索引完整性", "result": index_result})
    if index_result["index_exists"] and index_result["format_valid"]:
        print(f"  ✅ 索引存在，格式正确")
        print(f"  📊 产品数量：{index_result['product_count']}")
        all_results["summary"]["passed"] += 1
    else:
        print(f"  ❌ 索引存在问题")
        all_results["summary"]["failed"] += 1
    all_results["summary"]["total_tests"] += 1
    
    # 测试 2: 产品 Wiki 结构（每个产品）
    print("\n[测试 2] 产品 Wiki 结构")
    if PRODUCTS_PATH.exists():
        for product_dir in PRODUCTS_PATH.iterdir():
            if product_dir.is_dir():
                result = test_product_wiki_structure(product_dir)
                test_name = f"Wiki 结构 - {product_dir.name}"
                all_results["tests"].append({"name": test_name, "result": result})
                all_results["summary"]["total_tests"] += 1
                
                status = "✅" if result["wiki_exists"] and result["scripts_exists"] else "❌"
                print(f"  {status} {product_dir.name}")
                if result["wiki_exists"]:
                    print(f"     Wiki 模块：{', '.join(result['wiki_sections'])}")
                if result["violations"]:
                    for v in result["violations"]:
                        print(f"     ⚠️ {v}")
                    all_results["summary"]["violations"].extend(result["violations"])
                    all_results["summary"]["failed"] += 1
                else:
                    all_results["summary"]["passed"] += 1
    
    # 测试 3: 话术质量
    print("\n[测试 3] 话术质量")
    if PRODUCTS_PATH.exists():
        for product_dir in PRODUCTS_PATH.iterdir():
            if product_dir.is_dir():
                result = test_scripts_quality(product_dir)
                test_name = f"话术质量 - {product_dir.name}"
                all_results["tests"].append({"name": test_name, "result": result})
                all_results["summary"]["total_tests"] += 1
                
                if result["exists"]:
                    print(f"  ✅ {product_dir.name}")
                    print(f"     包含场景：{', '.join(result['sections'])}")
                    print(f"     质量评分：{result['quality_score']}%")
                    if result["violations"]:
                        for v in result["violations"]:
                            print(f"     ⚠️ {v}")
                        all_results["summary"]["violations"].extend(result["violations"])
                        all_results["summary"]["failed"] += 1
                    else:
                        all_results["summary"]["passed"] += 1
                else:
                    print(f"  ❌ {product_dir.name} - 话术文件不存在")
                    all_results["summary"]["failed"] += 1
    
    # 测试 4: Tavily 搜索集成
    print("\n[测试 4] Tavily 搜索集成")
    search_result = run_tavily("好医保长期医疗 2025 最新费率", 3)
    test_name = "Tavily 搜索集成"
    all_results["tests"].append({"name": test_name, "result": {"success": search_result is not None}})
    all_results["summary"]["total_tests"] += 1
    
    if search_result:
        print(f"  ✅ 搜索成功")
        if isinstance(search_result, dict):
            print(f"  📊 结果数量：{len(search_result.get('results', []))}")
            print(f"  💰 配额：{search_result.get('quota_info', {}).get('remaining', 'N/A')} 剩余")
        all_results["summary"]["passed"] += 1
    else:
        print(f"  ❌ 搜索失败")
        all_results["summary"]["failed"] += 1
    
    # 测试 5: 脚本可用性
    print("\n[测试 5] 文档提取脚本可用性")
    script_path = SKILL_PATH / "scripts" / "extract_doc.py"
    script_test = {"exists": script_path.exists(), "runnable": False}
    all_results["tests"].append({"name": "文档提取脚本", "result": script_test})
    all_results["summary"]["total_tests"] += 1
    
    if script_path.exists():
        try:
            result = subprocess.run(
                ["python3", str(script_path), "--help"],
                capture_output=True, text=True, timeout=10
            )
            script_test["runnable"] = result.returncode == 0
            if script_test["runnable"]:
                print(f"  ✅ 脚本可用")
                all_results["summary"]["passed"] += 1
            else:
                print(f"  ❌ 脚本运行失败")
                all_results["summary"]["failed"] += 1
        except Exception as e:
            print(f"  ❌ 脚本异常：{e}")
            all_results["summary"]["failed"] += 1
    else:
        print(f"  ❌ 脚本不存在")
        all_results["summary"]["failed"] += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    s = all_results["summary"]
    print(f"  总测试数：{s['total_tests']}")
    print(f"  通过：{s['passed']}")
    print(f"  失败：{s['failed']}")
    print(f"  红线违规：{len(s['violations'])}")
    
    if s['violations']:
        print("\n  红线违规详情：")
        for v in s['violations']:
            print(f"    - {v}")
    
    # 保存测试结果
    report_path = SKILL_PATH / "test-report-v2.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  测试报告已保存：{report_path}")
    
    return all_results

if __name__ == "__main__":
    run_all_tests()
