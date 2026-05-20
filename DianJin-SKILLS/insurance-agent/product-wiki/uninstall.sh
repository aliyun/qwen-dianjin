#!/usr/bin/env bash
# ============================================================
# 保险产品 Wiki - 卸载脚本
# 仅清理 Python 依赖，不删除任何产品数据
# 用法: bash uninstall.sh
# ============================================================

set -euo pipefail

echo ""
echo "========================================"
echo "  保险产品 Wiki - 卸载"
echo "========================================"
echo ""
echo "注意: 此脚本仅卸载 Python 依赖包"
echo "      产品数据 (products/, product-index.md 等) 不会被删除"
echo ""

read -p "确认卸载 Python 依赖? (y/N): " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "正在卸载..."

pip3 uninstall -y pdfplumber python-docx python-pptx 2>/dev/null \
    || pip uninstall -y pdfplumber python-docx python-pptx 2>/dev/null \
    || echo "  部分包已不存在，跳过"

echo ""
echo "========================================"
echo "  卸载完成"
echo "========================================"
echo ""
echo "已卸载: pdfplumber, python-docx, python-pptx"
echo "保留:   产品数据、Wiki 页面、话术文件"
echo ""
echo "重新安装: bash ~/.hermes/skills/insurance-product-wiki/install.sh"
echo ""
