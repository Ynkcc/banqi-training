"""banqi_training/actions.py — 动作表 / 动作计数的兼容入口。

动作序与动作计数的**唯一真源**在 `banqi_training.symmetry`
（`build_action_lookup_tables` / `compute_action_counts`，与 banqi-core
`core::env::actions.rs` / `core::env::config.rs` 同序）。

本模块仅做转发，不再保留独立实现，避免两份「Rust 镜像」各自漂移。
"""

from __future__ import annotations

from typing import Tuple

from banqi_training.symmetry import build_action_lookup_tables, compute_action_counts

# 兼容旧名：原 build_action_tables 与 symmetry.build_action_lookup_tables 语义、返回值一致。
build_action_tables = build_action_lookup_tables


def count_actions(rows: int, cols: int) -> Tuple[int, int, int, int]:
    """返回 (n_reveal, n_move, n_cannon, n_total)。

    计数取自 `compute_action_counts`（Rust const fn 的镜像），总数取自动作表长度，
    两者不一致即说明 Python 侧动作表与计数推导已漂移，直接失败。
    """
    action_to_coords, _ = build_action_lookup_tables(rows, cols)
    reveal, regular, cannon = compute_action_counts(rows, cols)
    total = len(action_to_coords)
    if total != reveal + regular + cannon:
        raise AssertionError(
            f"{rows}x{cols}: 动作表长度 {total} != 计数之和 {reveal + regular + cannon}"
        )
    return reveal, regular, cannon, total


if __name__ == "__main__":
    # 与三套旧 constant 断言：4x8 / 4x4 / 4x2
    expected = {
        (4, 8): (32, 104, 216, 352),
        (4, 4): (16, 48, 48, 112),
        (4, 2): (8, 20, 12, 40),
    }
    for (rows, cols), exp in expected.items():
        got = count_actions(rows, cols)
        assert got == exp, f"{(rows, cols)}: 推导 {got} != 预期 {exp}"
        print(f"[banqi_training.actions] {rows}x{cols}: reveal={got[0]} move={got[1]} "
              f"cannon={got[2]} total={got[3]} OK")
    print("[banqi_training.actions] all OK")
