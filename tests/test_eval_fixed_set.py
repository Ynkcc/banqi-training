"""固定验证集「仪表」单元测试：退化拒绝、跨轮累积、AUC 与 nan 语义。

背景（2026-09-17 事故）：固定验证集由启动后最初几百个样本一次性构建，恰好整批同一
终局类别时，corr(终局)/胜负区分度在数学上不可计算，而旧实现对此**静默返回字面量
0.0**，与「真实的零相关」无法区分 —— 价值评估仪表因此失效 268 轮无人发现。
本测试锁定修复后的行为：退化必须被拒绝并可见，池子跨轮累积直到能建出可用验证集。
"""

from __future__ import annotations

import math
import types
from types import SimpleNamespace

import numpy as np
import pytest

from banqi_training.constants import build_constants
from banqi_training.training.eval import (
    auc_win_loss,
    build_fixed_eval,
    result_classes,
    select_balanced_fixed_samples,
)
from banqi_training.training.worker import TrainWorker
from banqi_training.variant import get_variant


def _sample(variant, result: float, health: float = 0.0, teacher: int = 0) -> dict:
    """构造一个 build_fixed_eval 可消费的样本（形状取自变体常量）。"""
    C = build_constants(variant)
    return {
        "board_state": np.zeros(
            C.TOTAL_INPUT_CHANNELS * C.BOARD_ROWS * C.BOARD_COLS, dtype=np.float32
        ),
        "scalar_state": np.zeros(C.SCALAR_FEATURE_COUNT, dtype=np.float32),
        "policy_probs": np.full(C.ACTION_SPACE_SIZE, 1.0 / C.ACTION_SPACE_SIZE, dtype=np.float32),
        "action_mask": np.ones(C.ACTION_SPACE_SIZE, dtype=np.float32),
        "game_result_value": result,
        "health_diff": health,
        "teacher_action": teacher,
    }


def test_result_classes_counts() -> None:
    assert result_classes(np.array([1, 1, 0, -1], dtype=np.float32)) == {
        "win": 2,
        "draw": 1,
        "loss": 1,
    }


@pytest.mark.parametrize("vid", ["4x8", "4x4", "4x2"])
def test_build_fixed_eval_accepts_balanced(vid: str) -> None:
    variant = get_variant(vid)
    C = build_constants(variant)
    samples = [_sample(variant, r) for r in (1.0, -1.0, 0.0, 1.0)]

    fixed = build_fixed_eval(samples, variant)

    assert fixed is not None
    assert fixed["classes"] == {"win": 2, "draw": 1, "loss": 1}
    assert fixed["results"].tolist() == [1.0, -1.0, 0.0, 1.0]
    assert fixed["boards"].shape == (4, C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS)
    assert fixed["scalars"].shape == (4, C.SCALAR_FEATURE_COUNT)
    assert fixed["masks"].shape == (4, C.ACTION_SPACE_SIZE)
    assert np.isfinite(fixed["scalars"]).all()


def test_build_fixed_eval_rejects_single_class() -> None:
    """整批同一终局类别 → 必须拒绝构建（否则 corr/sep 恒为 0 的假仪表）。"""
    variant = get_variant("4x8")
    # 全平局（历史事故现场）
    assert build_fixed_eval([_sample(variant, 0.0) for _ in range(8)], variant) is None
    # 单类但取值为胜（同样无方差）
    assert build_fixed_eval([_sample(variant, 1.0) for _ in range(8)], variant) is None
    # 空样本
    assert build_fixed_eval([], variant) is None


def test_select_balanced_fixed_samples_covers_all_classes() -> None:
    variant = get_variant("4x8")
    pool = (
        [_sample(variant, 1.0) for _ in range(20)]
        + [_sample(variant, -1.0) for _ in range(20)]
        + [_sample(variant, 0.0) for _ in range(20)]
    )
    picked = select_balanced_fixed_samples(pool, 9)
    classes = result_classes(np.array([p["game_result_value"] for p in picked], dtype=np.float32))
    assert len(picked) == 9
    assert classes["win"] == 3 and classes["loss"] == 3 and classes["draw"] == 3


def test_auc_win_loss_separation_and_ties() -> None:
    results = np.array([1, 1, -1, -1], dtype=np.float32)

    assert auc_win_loss(np.array([3.0, 2.0, -2.0, -3.0], dtype=np.float32), results) == 1.0
    assert auc_win_loss(np.array([-3.0, -2.0, 2.0, 3.0], dtype=np.float32), results) == 0.0
    # 全部预测相等（类间并列）：并列校正后应为 0.5（不校正会算出假优值）
    assert auc_win_loss(np.zeros(4, dtype=np.float32), results) == 0.5
    # 类内并列但类间可分（胜方 2,2 全高于负方 -2,-2）：仍是完美 1.0
    assert auc_win_loss(np.array([2.0, 2.0, -2.0, -2.0], dtype=np.float32), results) == 1.0


def test_auc_win_loss_ignores_draws_and_is_nan_when_uncomputable() -> None:
    results = np.array([1, 0, 0, -1], dtype=np.float32)
    pred = np.array([5.0, 100.0, -100.0, -5.0], dtype=np.float32)
    # 平局样本的预测极端值不得影响 AUC（只取决胜样本）
    assert auc_win_loss(pred, results) == 1.0

    # 单类 / 空集 → nan（不可计算 ≠ 0）
    assert math.isnan(auc_win_loss(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)))
    assert math.isnan(auc_win_loss(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)))


def _worker_stub(variant, n_fixed: int):
    """只暴露 `_ensure_fixed_eval_from_selfplay` 所需属性的轻量替身。

    直接绑定真实方法（不构造完整 TrainWorker —— 那需要加载 torch 模型），
    保证测的是生产代码路径本身。
    """
    stub = SimpleNamespace(
        variant=variant,
        _fixed_eval=None,
        _raw_sample_pool=[],
        cfg=SimpleNamespace(VALUE_DRIFT_NUM_POSITIONS=n_fixed),
    )
    stub.ensure = types.MethodType(TrainWorker._ensure_fixed_eval_from_selfplay, stub)
    return stub


def test_fixed_eval_pool_accumulates_until_usable() -> None:
    """退化时保留池子、跨轮累积，直到能建出含两类终局的验证集。"""
    variant = get_variant("4x8")
    stub = _worker_stub(variant, n_fixed=6)

    # 第 1 轮：6 个全是平局 → 走完构建流程但退化被拒，池子必须保留
    stub.ensure([_sample(variant, 0.0) for _ in range(6)])
    assert stub._fixed_eval is None, "退化样本不得被当成可用验证集"
    assert len(stub._raw_sample_pool) == 6, "退化时必须保留池子（否则每轮都用同一小批重建，永远修不好）"

    # 第 2 轮：补齐决胜样本 → 构建成功（两类齐备即可用），池子清空（仪表只需建一次）
    stub.ensure([_sample(variant, 1.0), _sample(variant, -1.0)])
    assert stub._fixed_eval is not None
    classes = stub._fixed_eval["classes"]
    assert classes["win"] >= 1 and classes["loss"] >= 1, f"应含胜与负两类: {classes}"
    assert sum(classes.values()) == 6
    assert stub._raw_sample_pool == []

    # 已就绪后不再重建（局面固定才可比）
    stub.ensure([_sample(variant, 1.0)])
    assert stub._raw_sample_pool == []


def test_fixed_eval_pool_waits_until_enough_samples() -> None:
    """样本不足时只是等待，不构建也不清空。"""
    variant = get_variant("4x8")
    stub = _worker_stub(variant, n_fixed=10)

    stub.ensure([_sample(variant, 1.0) for _ in range(9)])
    assert stub._fixed_eval is None
    assert len(stub._raw_sample_pool) == 9
