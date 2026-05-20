# -*- coding: utf-8 -*-
"""兼容门面：实际实现已下沉至 shared.evaluation.significance。

现有的 `from significance import SignificanceTester` 调用不受影响。
新代码请直接 import shared.evaluation.significance。
"""
from evaluation.significance import *  # noqa: F401,F403
from evaluation.significance import SignificanceTester, CrossValidationTester  # noqa: F401

__all__ = ["SignificanceTester", "CrossValidationTester"]
