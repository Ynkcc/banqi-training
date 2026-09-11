"""banqi_training/symmetry.py — 空间对称增强（纯 Python 实现）。

与 banqi-core `core::env::symmetry.rs` / `actions.rs` 语义逐条一致：
  - build_action_lookup_tables(rows, cols)
    动作坐标表（唯一动作序来源）：翻棋 → 常规移动（上下左右）→ 炮击（水平/垂直，
    距离 > 1），与 Rust `action_lookup_tables` 同序。
  - sq_map / action_permutation / transform_board
    D4 空间对称：格子重排表 map[i] = 变换后位置 i 的原格子索引；
    动作置换表 perm 满足 new_policy = old_policy[perm]；
    扁平特征 (channels, rows, cols) 沿空间轴重排，通道序不变。
  - validate_symmetry
    置换合法性自检：排列 + 对合（perm[perm[i]]==i）或 4 次还原。

变换集：identity / rot90 / rot180 / rot270 / hflip / vflip / diag / anti_diag；
各变体可用集由 variant.non_identity_transforms 限定。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

SYMMETRY_TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "diag",
    "anti_diag",
)

# 对合变换（两次作用恒等）；rot90/rot270 需 4 次还原
_INVOLUTIONS = {"identity", "rot180", "hflip", "vflip", "diag", "anti_diag"}


def compute_action_counts(rows: int, cols: int) -> Tuple[int, int, int]:
    """翻棋 / 常规移动 / 炮击 动作计数（与动作表构造逻辑一致）。"""
    reveal = rows * cols
    regular = 0
    cannon = 0
    for r1 in range(rows):
        for c1 in range(cols):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= r1 + dr < rows and 0 <= c1 + dc < cols:
                    regular += 1
            cannon += sum(1 for c2 in range(cols) if abs(c1 - c2) > 1)
            cannon += sum(1 for r2 in range(rows) if abs(r1 - r2) > 1)
    return reveal, regular, cannon


@lru_cache(maxsize=None)
def build_action_lookup_tables(rows: int, cols: int):
    """动作坐标表：(action -> coords 序列, coords -> action 映射)。

    动作序与 Rust `build_action_lookup_tables` 一致：
      1. 翻棋：sq = 0..rows*cols；
      2. 常规移动：按 (r1, c1, 方向[-1,0 / 1,0 / 0,-1 / 0,1]) 枚举界内相邻格；
      3. 炮击：按 (r1, c1) 枚举水平（同行，|dc|>1）再垂直（同列，|dr|>1），
         去重（与常规移动相邻格不重叠，跨方向亦不重叠）。
    """
    action_to_coords: List[Tuple[int, ...]] = []
    coords_to_action: Dict[Tuple[int, ...], int] = {}

    def _add(coords: Tuple[int, ...]) -> None:
        action_to_coords.append(coords)
        coords_to_action[coords] = len(action_to_coords) - 1

    # 1. 翻棋
    for sq in range(rows * cols):
        _add((sq,))

    # 2. 常规移动
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for r1 in range(rows):
        for c1 in range(cols):
            from_sq = r1 * cols + c1
            for dr, dc in moves:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    _add((from_sq, r2 * cols + c2))

    # 3. 炮击
    for r1 in range(rows):
        for c1 in range(cols):
            from_sq = r1 * cols + c1
            for c2 in range(cols):
                if abs(c1 - c2) > 1:
                    coords = (from_sq, r1 * cols + c2)
                    if coords not in coords_to_action:
                        _add(coords)
            for r2 in range(rows):
                if abs(r1 - r2) > 1:
                    coords = (from_sq, r2 * cols + c1)
                    if coords not in coords_to_action:
                        _add(coords)

    return action_to_coords, coords_to_action


def _map_sq(rows: int, cols: int, sym: str, rr: int, cc: int) -> Tuple[int, int]:
    """变换后位置 (rr, cc) 对应的原格子坐标 (pr, pc)。与 Rust `sq_map` 一致。"""
    if sym == "identity":
        return rr, cc
    if sym == "rot90":  # 顺时针 90°
        return cols - 1 - cc, rr
    if sym == "rot180":
        return rows - 1 - rr, cols - 1 - cc
    if sym == "rot270":  # 逆时针 90°
        return cc, rows - 1 - rr
    if sym == "hflip":
        return rr, cols - 1 - cc
    if sym == "vflip":
        return rows - 1 - rr, cc
    if sym == "diag":
        return cc, rr
    if sym == "anti_diag":
        return cols - 1 - cc, rows - 1 - rr
    raise ValueError(f"未知对称变换 {sym!r}")


@lru_cache(maxsize=None)
def sq_map(rows: int, cols: int, sym: str) -> Tuple[int, ...]:
    """格子重排表：map[i] = 变换后位置 i 对应的原格子索引。"""
    if sym not in SYMMETRY_TRANSFORMS:
        raise ValueError(f"未知对称变换 {sym!r}")
    out: List[int] = []
    for rr in range(rows):
        for cc in range(cols):
            pr, pc = _map_sq(rows, cols, sym, rr, cc)
            out.append(pr * cols + pc)
    return tuple(out)


@lru_cache(maxsize=None)
def get_action_symmetry_table(rows: int, cols: int, transform: str) -> Tuple[int, ...]:
    """动作置换表 perm（new_policy = old_policy[perm]）。"""
    if transform not in SYMMETRY_TRANSFORMS:
        raise ValueError(f"未知对称变换 {transform!r}")
    action_to_coords, coords_to_action = build_action_lookup_tables(rows, cols)
    mapping = sq_map(rows, cols, transform)
    perm = [0] * len(action_to_coords)
    for a, coords in enumerate(action_to_coords):
        mapped = tuple(mapping[sq] for sq in coords)
        perm[coords_to_action.get(mapped, a)] = a
    return tuple(perm)


def transform_board(
    board: Sequence[float], rows: int, cols: int, channels: int, transform: str
) -> List[float]:
    """扁平特征张量沿空间轴重排：out[c, i, j] = in[c, pr(i, j), pc(i, j)]。"""
    if len(board) != channels * rows * cols:
        raise ValueError(
            f"board 长度 {len(board)} != channels({channels})*rows({rows})*cols({cols})"
            f" = {channels * rows * cols}"
        )
    mapping = sq_map(rows, cols, transform)
    plane = rows * cols
    out: List[float] = []
    for ch in range(channels):
        base = ch * plane
        out.extend(board[base + sq] for sq in mapping)
    return out


def transform_policy(policy: Sequence[float], perm: Sequence[int]) -> List[float]:
    """按置换表 gather：out[a] = policy[perm[a]]。"""
    if len(policy) != len(perm):
        raise ValueError(f"policy 长度 {len(policy)} != 置换表长度 {len(perm)}")
    return [policy[a] for a in perm]


def transform_action(action: int, perm: Sequence[int]) -> int:
    """对单个 action 做置换：返回 perm[action]。"""
    if action >= len(perm):
        raise IndexError(f"action {action} 超出置换表长度 {len(perm)}")
    return perm[action]


def validate_symmetry(rows: int, cols: int, transforms: Sequence[str]) -> bool:
    """校验置换合法性：是排列，且对合变换满足 perm[perm[i]]==i，rot90/rot270 满足 4 次还原。"""
    action_space_size = len(build_action_lookup_tables(rows, cols)[0])
    for name in transforms:
        perm = get_action_symmetry_table(rows, cols, name)
        if len(set(perm)) != action_space_size or any(p >= action_space_size for p in perm):
            return False
        if name in _INVOLUTIONS:
            if any(perm[perm[i]] != i for i in range(action_space_size)):
                return False
        else:
            if any(perm[perm[perm[perm[i]]]] != i for i in range(action_space_size)):
                return False
    return True
