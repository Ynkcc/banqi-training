"""banqi/actions.py — 动作表构建（Rust build_action_lookup_tables 的 Python 镜像）

src/game_env/actions.rs::build_action_lookup_tables 的唯一镜像实现：
  1. 翻棋：action == 格子序号
  2. 常规移动：四方向（上/下/左/右）各 1 步
  3. 炮击：同行/同列隔子（|距离| > 1），已存在表中则跳过

动作顺序必须与 Rust 逐条一致 —— data_augmentation 的动作置换表、
constants 的动作空间计数都依赖它。任何修改需与 Rust 侧同步并跑自检。
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def build_action_tables(
    rows: int, cols: int
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    """返回 (action_to_coords, coords_to_action)，顺序与 Rust 完全一致。"""
    total_positions = rows * cols
    action_to_coords: List[Tuple[int, ...]] = []
    coords_to_action: Dict[Tuple[int, ...], int] = {}
    idx = 0

    # 1. 翻棋：action == sq
    for sq in range(total_positions):
        coords = (sq,)
        action_to_coords.append(coords)
        coords_to_action[coords] = idx
        idx += 1

    # 2. 常规移动：四方向各 1 步（顺序：上/下/左/右，与 Rust 一致）
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r1 in range(rows):
        for c1 in range(cols):
            from_sq = r1 * cols + c1
            for dr, dc in moves:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    coords = (from_sq, r2 * cols + c2)
                    action_to_coords.append(coords)
                    coords_to_action[coords] = idx
                    idx += 1

    # 3. 炮击：同行隔子（水平）+ 同列隔子（垂直），已在表中的对跳过
    for r1 in range(rows):
        for c1 in range(cols):
            from_sq = r1 * cols + c1
            # 水平
            for c2 in range(cols):
                if abs(c1 - c2) > 1:
                    coords = (from_sq, r1 * cols + c2)
                    if coords not in coords_to_action:
                        action_to_coords.append(coords)
                        coords_to_action[coords] = idx
                        idx += 1
            # 垂直
            for r2 in range(rows):
                if abs(r1 - r2) > 1:
                    coords = (from_sq, r2 * cols + c1)
                    if coords not in coords_to_action:
                        action_to_coords.append(coords)
                        coords_to_action[coords] = idx
                        idx += 1

    return action_to_coords, coords_to_action


def _is_adjacent(coords: Tuple[int, ...], cols: int) -> bool:
    """两格坐标（from_sq, to_sq）是否相邻（四方向各 1 步）。"""
    f, t = coords
    r1, c1 = divmod(f, cols)
    r2, c2 = divmod(t, cols)
    return abs(r1 - r2) + abs(c1 - c2) == 1


def count_actions(rows: int, cols: int) -> Tuple[int, int, int, int]:
    """返回 (n_reveal, n_move, n_cannon, n_total)，由动作表推导。"""
    action_to_coords, _ = build_action_tables(rows, cols)
    n_reveal = sum(1 for c in action_to_coords if len(c) == 1)
    n_move = sum(1 for c in action_to_coords if len(c) == 2 and _is_adjacent(c, cols))
    n_total = len(action_to_coords)
    n_cannon = n_total - n_reveal - n_move
    return n_reveal, n_move, n_cannon, n_total


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
