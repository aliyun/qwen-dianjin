#!/usr/bin/env bash
# ============================================================
# 保险产品 Wiki - 依赖安装脚本
# 检测并安装所有依赖，支持优雅降级提示
# 用法: bash install.sh
# ============================================================

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
WARN=0
FAIL=0

print_header() {
    echo ""
    echo "========================================"
    echo "  保险产品 Wiki - 依赖检查 & 安装"
    echo "========================================"
    echo ""
}

print_pass() {
    echo "  ✅ $1"
    PASS=$((PASS + 1))
}

print_warn() {
    echo "  ⚠️  $1"
    WARN=$((WARN + 1))
}

print_fail() {
    echo "  ❌ $1"
    FAIL=$((FAIL + 1))
}

# -------------------- 1. Python 依赖 --------------------
check_python_deps() {
    echo "📦 检查 Python 依赖..."

    if ! command -v python3 &>/dev/null; then
        print_fail "python3 未安装，请先安装 Python 3"
        return
    fi
    print_pass "python3 已安装: $(python3 --version 2>&1)"

    local pip_ok=false
    if command -v pip3 &>/dev/null; then
        pip_ok=true
    elif command -v pip &>/dev/null; then
        pip_ok=true
    fi

    if [ "$pip_ok" = false ]; then
        print_warn "pip 未检测到，无法自动安装 Python 包"
        print_warn "请手动运行: pip install pdfplumber python-docx python-pptx Pillow"
        return
    fi

    # 逐个检查 Python 包
    local packages=("pdfplumber" "docx" "pptx")
    local package_names=("pdfplumber" "python-docx" "python-pptx")
    local missing=()

    for i in "${!packages[@]}"; do
        if python3 -c "import ${packages[$i]}" 2>/dev/null; then
            print_pass "${package_names[$i]} 已安装"
        else
            missing+=("${package_names[$i]}")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "  📋 缺失的包: ${missing[*]}"
        echo "  正在安装..."
        if pip3 install "${missing[@]}" 2>/dev/null || pip install "${missing[@]}" 2>/dev/null; then
            print_pass "安装完成"
        else
            print_fail "安装失败，请手动运行: pip install ${missing[*]}"
        fi
    fi
}

# -------------------- 2. Tavily 搜索技能 --------------------
check_tavily() {
    echo ""
    echo "🔍 检查 Tavily 搜索技能..."

    local tavily_path="$HOME/.hermes/skills/openclaw-imports/tavily-search-skill/search.sh"

    if [ -f "$tavily_path" ]; then
        print_pass "Tavily 搜索技能已就绪"
    else
        print_warn "Tavily 搜索技能未找到: $tavily_path"
        echo ""
        echo "  降级说明: 无 Tavily 时仅使用本地产品 Wiki + 知识库"
        echo "  如需安装，请将 tavily-search-skill 放入:"
        echo "    ~/.hermes/skills/openclaw-imports/tavily-search-skill/"
    fi
}

# -------------------- 3. 本地知识库 --------------------
check_knowledge_base() {
    echo ""
    echo "📚 检查本地知识库..."

    local kb_path="$HOME/.openclaw/workspace/knowledge-base"
    local notes_path="$kb_path/读书笔记"

    if [ -d "$kb_path" ]; then
        print_pass "本地知识库已就绪: $kb_path"
    else
        print_warn "本地知识库未找到: $kb_path"
        echo ""
        echo "  降级说明: 无知识库时仅基于产品 Wiki 回答"
        echo "  如需使用，请将知识库放置到上述路径"
    fi

    if [ -d "$notes_path" ]; then
        local count
        count=$(find "$notes_path" -name "*.md" 2>/dev/null | wc -l)
        print_pass "读书笔记目录: $count 个文件"

        # 检查保险相关笔记
        if [ -f "$notes_path/012-保险基础知识核心思想.md" ]; then
            print_pass "保险基础笔记已就绪"
        else
            print_warn "未找到 012-保险基础知识核心思想.md"
        fi

        if [ -f "$notes_path/019-这就是保险代理人核心思想.md" ]; then
            print_pass "保险代理人笔记已就绪"
        else
            print_warn "未找到 019-这就是保险代理人核心思想.md"
        fi
    fi
}

# -------------------- 4. 技能自身完整性 --------------------
check_skill_integrity() {
    echo ""
    echo "🔧 检查技能完整性..."

    local critical_files=(
        "SKILL.md"
        "schema.md"
        "product-index.md"
        "scripts/extract_doc.py"
        "templates/product_wiki.md"
        "templates/script_template.md"
    )

    for f in "${critical_files[@]}"; do
        if [ -f "$SKILL_DIR/$f" ]; then
            print_pass "$f"
        else
            print_fail "$f 缺失"
        fi
    done

    # 检查 extract_doc.py 可执行
    if [ -f "$SKILL_DIR/scripts/extract_doc.py" ]; then
        chmod +x "$SKILL_DIR/scripts/extract_doc.py" 2>/dev/null || true
    fi
}

# -------------------- 主流程 --------------------
print_header

check_python_deps
check_tavily
check_knowledge_base
check_skill_integrity

echo ""
echo "========================================"
echo "  检查完成"
echo "========================================"
echo "  ✅ 通过: $PASS"
echo "  ⚠️  警告: $WARN (可降级使用)"
echo "  ❌ 失败: $FAIL"
echo "========================================"
echo ""

if [ $WARN -gt 0 ]; then
    echo "💡 提示: 警告项不影响核心功能使用，技能将自动降级运行"
fi

if [ $FAIL -gt 0 ]; then
    echo "⚡ 注意: 失败项可能影响部分功能，建议修复后重试"
fi

echo ""
echo "使用技能: 提及 '保险产品' 或 '产品入库' 即可触发"
echo ""
