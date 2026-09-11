"""banqi/eval.py — 评估常量薄封装（对战评估已移交调度器 gatekeeper rating）。

价值漂移 / 策略命中率等评估核心工具在 `banqi_training/training/eval.py`。
"""

from __future__ import annotations

from banqi_training.training.eval import eval_policy_accuracy, eval_value_drift

__all__ = [
    "eval_policy_accuracy",
    "eval_value_drift",
]
