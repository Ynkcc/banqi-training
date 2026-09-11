"""banqi/constants.py — 由 Variant 派生全部维度常量

所有「棋盘尺寸 / 通道数 / 标量维度 / 动作空间 / 子力配置」常量都由
variant 唯一声明源计算，不再手写。

对外暴露统一入口：
    consts = build_constants(variant)        # -> Constants（含全部命名常量）
    d = consts.module()                      # -> 模块级字典（供薄壳 globals().update）
"""

from __future__ import annotations

from typing import Any, Dict

from banqi_training.variant import Variant


class Constants:
    """一个变体的全部派生常量。全部字段只读。"""

    def __init__(self, v: Variant) -> None:
        self.variant = v
        # ---- 棋盘 / 特征 ----
        self.BOARD_ROWS = v.board_rows
        self.BOARD_COLS = v.board_cols
        self.TOTAL_POSITIONS = v.total_positions
        self.NUM_PIECE_TYPES = 7
        self.NUM_ACTIVE_PIECE_TYPES = v.num_active_piece_types
        self.BOARD_CHANNELS = v.board_channels
        self.TOTAL_INPUT_CHANNELS = v.board_channels
        self.TOTAL_PIECES_PER_PLAYER = v.total_pieces_per_player
        self.SURVIVAL_VECTOR_SIZE = v.total_pieces_per_player
        self.SCALAR_FEATURE_COUNT = v.scalar_feature_count
        # ---- 子力 ----
        self.PIECE_COUNTS = v.piece_counts
        self.PIECE_VALUES = v.piece_values
        self.INITIAL_HEALTH = v.initial_health
        # ---- 整型血量差（离散分类头） ----
        # 未归一化的血量差 d = 己方HP - 对方HP ∈ [-D, +D]，D = INITIAL_HEALTH。
        # 离散分类头输出 K = 2D+1 个桶；桶 i 的整数中心 = i - D，One-hot 目标 index = d + D。
        self.HEALTH_DIFF_BINS = 2 * v.initial_health + 1
        # 归一化血量差的除分母，与 Rust `terminal_health_diff_red` 完全一致：
        #   normalized = (红HP - 黑HP) / (initial_health + max(piece_values))
        self.HEALTH_DIFF_DENOM = v.initial_health + max(v.piece_values)
        self.SOLDIERS_COUNT, self.CANNONS_COUNT, self.HORSES_COUNT, \
            self.CHARIOTS_COUNT, self.ELEPHANTS_COUNT, self.ADVISORS_COUNT, \
            self.GENERALS_COUNT = v.piece_counts
        # ---- 动作空间 ----
        self.REVEAL_ACTIONS_COUNT, self.REGULAR_MOVE_ACTIONS_COUNT, \
            self.CANNON_ATTACK_ACTIONS_COUNT = v.action_counts
        self.ACTION_SPACE_SIZE = v.action_space_size
        # ---- 网络 ----
        self.HIDDEN_CHANNELS = v.hidden_channels
        self.NUM_RES_BLOCKS = v.num_res_blocks
        self.POLICY_HEAD_CHANNELS = v.policy_head_channels
        self.VALUE_HEAD_CHANNELS = v.value_head_channels
        self.POLICY_FC1_HIDDEN = v.policy_fc1_hidden
        self.VALUE_FC1_HIDDEN = v.value_fc1_hidden

    def health_diff_int(self, normalized: float) -> int:
        """由归一化血量差精确反推整型血量差（按己方视角）。

        Rust 侧 `terminal_health_diff_red` 返回 `diff / HEALTH_DIFF_DENOM`，其中
        `diff` 是整型血量差；因此 `round(normalized * DENOM)` 可精确恢复（无量化误差）。
        非有限输入（NaN/Inf）归 0（中位桶），由调用方按异常样本过滤处理。
        """
        n = float(normalized)
        if n != n or n in (float("inf"), float("-inf")):  # NaN / ±Inf
            return 0
        val = int(round(n * self.HEALTH_DIFF_DENOM))
        return max(-self.INITIAL_HEALTH, min(self.INITIAL_HEALTH, val))

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in dir(self)
                if k.isupper() and not k.startswith("_")}

    def module(self) -> Dict[str, Any]:
        """生成适合 `globals().update(...)` 的模块级字典（含 variant）。"""
        d = self.as_dict()
        d["variant"] = self.variant
        return d


_cache: Dict[str, Constants] = {}


def build_constants(variant: Variant) -> Constants:
    """构造（并缓存）一个变体的 Constants。"""
    if variant.id not in _cache:
        _cache[variant.id] = Constants(variant)
    return _cache[variant.id]


if __name__ == "__main__":
    from banqi_training.variant import VARIANTS
    for vid, v in VARIANTS.items():
        c = build_constants(v)
        print(f"[banqi_training.constants] {vid}: ch={c.TOTAL_INPUT_CHANNELS} "
              f"scalar={c.SCALAR_FEATURE_COUNT} action={c.ACTION_SPACE_SIZE} "
              f"pieces={c.TOTAL_PIECES_PER_PLAYER} health={c.INITIAL_HEALTH}")
    print("[banqi_training.constants] all OK")
