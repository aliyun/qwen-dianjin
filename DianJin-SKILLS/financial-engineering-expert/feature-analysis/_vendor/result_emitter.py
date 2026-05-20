# -*- coding: utf-8 -*-
"""
ResultEmitter (portable version) — 将脚本产物清单落盘为 outputs/result.json。

与内置版 (backend/skills/shared/result_emitter.py) 的区别：
- 不再 print [RESULT:{json}] 供 python_executor 正则捕获
- 改为 write(path) 将 manifest 写入本地文件，平台侧读取文件即可
- 新增 metrics / summary 字段，便于外部 LLM 直接摘要

Manifest 结构：
    {
        "skill": "xgb-modeling",
        "status": "success | failed",
        "output_dir": "./outputs/20260512_143000",
        "files": [{"path": "...", "role": "model|meta|report|asset|other", "meta": {...}}],
        "metrics": {"auc": 0.82, "ks": 0.45, ...},
        "summary": "3 行人类可读摘要",
        "state": {...},
        "artifacts": [{"type": "...", "title": "...", "data": {...}}]
    }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class ResultEmitter:
    """累积产物并在脚本结束时写入 result.json。"""

    def __init__(self, skill: str = "", output_dir: Optional[Union[str, Path]] = None) -> None:
        self._skill: str = skill
        self._output_dir: Optional[str] = str(output_dir) if output_dir else None
        self._status: str = "success"
        self._files: List[Dict[str, Any]] = []
        self._state: Dict[str, Any] = {}
        self._artifacts: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = {}
        self._summary: str = ""
        self._report_path: Optional[str] = None
        self._written: bool = False

    # ── 累积 API ─────────────────────────────────────────────────────────

    def add_file(
        self,
        path: Union[str, Path],
        role: str = "other",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """登记一个产物文件。role: model / meta / report / asset / other"""
        self._files.append({
            "path": str(path),
            "role": role,
            "meta": meta or {},
        })

    def add_model(
        self,
        model_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None,
        **meta: Any,
    ) -> None:
        """同时登记模型文件和对应 meta JSON。"""
        self.add_file(model_path, role="model", meta=meta)
        if meta_path is not None:
            self.add_file(meta_path, role="meta")

    def add_report(self, report_path: Union[str, Path]) -> None:
        """登记 markdown 报告路径。"""
        self._report_path = str(report_path)
        self.add_file(report_path, role="report")

    def add_asset(self, path: Union[str, Path], title: str = "") -> None:
        """登记图片/附件资源。"""
        self.add_file(path, role="asset", meta={"title": title} if title else None)

    def add_artifact(self, type_: str, title: str, data: Any) -> None:
        """登记一个可视化 artifact。"""
        self._artifacts.append({"type": type_, "title": title, "data": data})

    def update_state(self, data: Dict[str, Any]) -> None:
        """合并 state（用于跨 skill 传递轻量上下文）。"""
        if data:
            self._state.update(data)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """合并关键指标（auc/ks/brier 等）。"""
        if metrics:
            self._metrics.update(metrics)

    def set_summary(self, summary: str) -> None:
        """设置人类可读摘要。"""
        self._summary = summary.strip()

    def set_status(self, status: str) -> None:
        """success / failed"""
        self._status = status

    def set_output_dir(self, output_dir: Union[str, Path]) -> None:
        self._output_dir = str(output_dir)

    def set_report_path(self, path: Union[str, Path]) -> None:
        self._report_path = str(path)

    # ── 落盘 ─────────────────────────────────────────────────────────────

    def build_manifest(self) -> Dict[str, Any]:
        """生成 manifest dict（不落盘）。"""
        return {
            "skill": self._skill,
            "status": self._status,
            "output_dir": self._output_dir,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "files": self._files,
            "metrics": self._metrics,
            "summary": self._summary,
            "state": self._state,
            "artifacts": self._artifacts,
            "report_path": self._report_path,
        }

    def write(self, path: Union[str, Path]) -> str:
        """将 manifest 写入 result.json，返回绝对路径；幂等。"""
        if self._written:
            return str(Path(path).resolve())
        manifest = self.build_manifest()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._written = True
        return str(out_path.resolve())

    # ── introspection（测试用） ─────────────────────────────────────────

    @property
    def files(self) -> List[Dict[str, Any]]:
        return list(self._files)

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    @property
    def artifacts(self) -> List[Dict[str, Any]]:
        return list(self._artifacts)

# ── 便捷 API：一次性 write，适合简单脚本 ────────────────────────────────────

def write_result(
    path: Union[str, Path],
    *,
    skill: str = "",
    status: str = "success",
    output_dir: Optional[Union[str, Path]] = None,
    files: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    summary: str = "",
    state: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    report_path: Optional[Union[str, Path]] = None,
) -> str:
    """一把梭写入 result.json。"""
    manifest = {
        "skill": skill,
        "status": status,
        "output_dir": str(output_dir) if output_dir else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files": files or [],
        "metrics": metrics or {},
        "summary": summary.strip(),
        "state": state or {},
        "artifacts": artifacts or [],
        "report_path": str(report_path) if report_path else None,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(out_path.resolve())

# 向后兼容：保留 emit_result 名字但底层改为写文件（需要 path 参数）
def emit_result(path: Union[str, Path], **kwargs: Any) -> str:
    """[兼容] 等价于 write_result(path, **kwargs)。"""
    return write_result(path, **kwargs)

__all__ = ["ResultEmitter", "write_result", "emit_result"]
