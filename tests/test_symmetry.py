"""banqi_training/symmetry.py 单元测试：与 Rust 语义一致性（动作表 / 置换 / 重排）。"""

from __future__ import annotations

import pytest

from banqi_training.constants import build_constants
from banqi_training.symmetry import (
    SYMMETRY_TRANSFORMS,
    build_action_lookup_tables,
    compute_action_counts,
    get_action_symmetry_table,
    sq_map,
    transform_action,
    transform_board,
    transform_policy,
    validate_symmetry,
)
from banqi_training.variant import VARIANTS

# 矩形盘合法变换集（与 variant.symmetries / Rust search_group 的 Klein-4 一致）
_RECT_TRANSFORMS = ("identity", "rot180", "hflip", "vflip")


@pytest.mark.parametrize("rows,cols", [(4, 2), (4, 4), (4, 8)])
def test_action_counts_match_tables(rows: int, cols: int) -> None:
    """动作表大小与计数函数一致（翻棋/移动/炮击三段同序）。"""
    action_to_coords, _ = build_action_lookup_tables(rows, cols)
    reveal, regular, cannon = compute_action_counts(rows, cols)
    assert len(action_to_coords) == reveal + regular + cannon
    assert all(len(c) == 1 for c in action_to_coords[:reveal])
    assert all(len(c) == 2 for c in action_to_coords[reveal:])


@pytest.mark.parametrize("vid", list(VARIANTS))
def test_action_space_size_matches_table(vid: str) -> None:
    """动作表大小与变体常量 ACTION_SPACE_SIZE 一致（Python 唯一声明源自洽）。"""
    v = VARIANTS[vid]
    C = build_constants(v)
    action_to_coords, _ = build_action_lookup_tables(C.BOARD_ROWS, C.BOARD_COLS)
    assert len(action_to_coords) == C.ACTION_SPACE_SIZE
    reveal, regular, cannon = v.action_counts
    reveal_n, regular_n, cannon_n = compute_action_counts(C.BOARD_ROWS, C.BOARD_COLS)
    assert (reveal, regular, cannon) == (reveal_n, regular_n, cannon_n)


def test_action_coords_semantics() -> None:
    """动作坐标语义抽查：翻棋=单坐标；移动=相邻；炮击=同行/列且距离>1。"""
    rows, cols = 4, 4
    action_to_coords, coords_to_action = build_action_lookup_tables(rows, cols)
    reveal = rows * cols
    # 翻棋段：(0,0) 翻棋 = action 0
    assert action_to_coords[0] == (0,)
    # 移动段（方向序 (-1,0),(1,0),(0,-1),(0,1)）：(0,0) 处 (1,0) 与 (0,1) 有效
    a_down = coords_to_action[(0, cols)]  # (0,0)->(1,0)
    a_right = coords_to_action[(0, 1)]  # (0,0)->(0,1)
    assert (a_down, a_right) == (reveal, reveal + 1)
    # 炮击段：同行距离 > 1，例如 (0,0)->(0,3)
    assert (0, 3) in coords_to_action
    assert coords_to_action[(0, 3)] >= reveal + 8  # 位于常规移动段之后


@pytest.mark.parametrize("rows,cols", [(4, 2), (4, 4), (4, 8)])
@pytest.mark.parametrize("transform", _RECT_TRANSFORMS)
def test_permutation_validity_klein4(rows: int, cols: int, transform: str) -> None:
    """Klein-4 变置换合法性（排列 + 对合），矩形盘与方盘均适用。"""
    assert validate_symmetry(rows, cols, [transform])


@pytest.mark.parametrize(
    "transform",
    ("rot90", "rot270", "diag", "anti_diag"),
)
def test_permutation_validity_square_only(transform: str) -> None:
    """方盘（4x4）专属 D4 变换的置换合法性（rot90/rot270 为 4 次还原）。"""
    assert validate_symmetry(4, 4, [transform])


def test_policy_gather_semantics() -> None:
    """new_policy = old_policy[perm]：单位置单热经置换后仍单热且位置正确。"""
    rows, cols = 4, 4
    _, coords_to_action = build_action_lookup_tables(rows, cols)
    perm = get_action_symmetry_table(rows, cols, "hflip")
    # (0,0)->(0,1) 的移动动作，hflip 后应为 (0,3)->(0,2)
    src = coords_to_action[(0, 1)]
    dst = coords_to_action[(0 * cols + 3, 0 * cols + 2)]
    onehot = [0.0] * len(perm)
    onehot[src] = 1.0
    out = transform_policy(onehot, perm)
    assert out[dst] == 1.0 and sum(out) == 1.0
    assert transform_action(src, perm) == dst


def test_transform_board_hflip() -> None:
    """hflip 沿列翻转：out[c,r,k] = in[c,r,cols-1-k]；其余轴不变。"""
    rows, cols, channels = 2, 3, 2
    board = [float(i) for i in range(channels * rows * cols)]
    out = transform_board(board, rows, cols, channels, "hflip")
    for ch in range(channels):
        for r in range(rows):
            for k in range(cols):
                assert out[ch * rows * cols + r * cols + k] == board[
                    ch * rows * cols + r * cols + (cols - 1 - k)
                ]


def test_transform_board_identity() -> None:
    board = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert transform_board(board, 2, 3, 1, "identity") == board


def test_sq_map_involution() -> None:
    """对合变换的格子重排表两次作用还原。"""
    rows, cols = 4, 8
    for sym in ("rot180", "hflip", "vflip"):
        m = sq_map(rows, cols, sym)
        assert all(m[m[i]] == i for i in range(rows * cols))


@pytest.mark.parametrize("vid", list(VARIANTS))
def test_augmenter_roundtrip(vid: str) -> None:
    """EpisodeAugmenter 对 episode dict 的增强结果与置换表 gather 一致。

    输入的 episode dict 走解码器输出的形状约定：boards (steps, channels, rows, cols)，
    policies / action_masks (steps, action_space)。
    """
    import numpy as np

    from banqi_training.training.augment import EpisodeAugmenter

    class _Cfg:
        DATA_AUGMENT_ENABLED = True
        DATA_AUGMENT_TRANSFORMS = ""
        DATA_AUGMENT_KEEP_ORIGINAL = True

    variant = VARIANTS[vid]
    C = build_constants(variant)
    aug = EpisodeAugmenter(variant, _Cfg())
    aspace = C.ACTION_SPACE_SIZE
    rows, cols = C.BOARD_ROWS, C.BOARD_COLS
    channels = C.TOTAL_INPUT_CHANNELS
    ep = {
        "boards": np.ones((1, channels, rows, cols), dtype=np.float32),
        "policies": np.zeros((1, aspace), dtype=np.float32),
        "action_masks": np.ones((1, aspace), dtype=np.int32),
        "actions": np.array([0], dtype=np.uint32),
    }
    for t in variant.non_identity_transforms:
        out = aug.transform_episode(ep, t)
        perm = aug.permutation(t)
        assert out["boards"].shape == (1, channels, rows, cols)
        assert out["boards"][0].reshape(-1).tolist() == transform_board(
            ep["boards"][0].reshape(-1).tolist(), rows, cols, channels, t
        )
        assert out["policies"].shape == (1, aspace)
        assert out["policies"][0].tolist() == transform_policy(ep["policies"][0].tolist(), perm)
        assert out["action_masks"][0].tolist() == transform_policy(
            ep["action_masks"][0].tolist(), perm
        )
        assert out["actions"].tolist() == [transform_action(0, perm)]
