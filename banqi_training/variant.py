"""banqi/variant.py — 变体描述符：4x2 / 4x4 / 4x8 的唯一声明源

每个变体只在这里声明一次「棋盘形状 / 子力配置 / 网络尺寸 / 可用对称 /
Rust 绑定前缀 / 模型文件名」，其余所有派生量（通道数、标量维度、动作空间等）
由 cached_property 计算，保证单一声明源。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple

from banqi_training.actions import count_actions

# PieceType 索引：兵0 炮1 马2 车3 象4 士5 将6
NUM_PIECE_TYPES = 7

# 全部可用的空间对称变换（某棋盘是否可用由 variant.symmetries 限定）
ALL_SYMMETRY_TRANSFORMS: Tuple[str, ...] = (
    "identity", "rot90", "rot180", "rot270", "hflip", "vflip", "diag", "anti_diag",
)
# 对合变换（perm[perm] == identity）；rot90 / rot270 需 4 次还原
INVOLUTION_TRANSFORMS: Tuple[str, ...] = ("rot180", "hflip", "vflip", "diag", "anti_diag")


@dataclass(frozen=True)
class Variant:
    """一个棋盘变体的完整声明。"""

    id: str
    board_rows: int
    board_cols: int
    piece_counts: Tuple[int, ...]          # 7 元，按 PieceType 索引（允许 0）
    piece_values: Tuple[int, ...]          # 7 元，按 PieceType 索引
    initial_health: int                    # 血量/目标分上限
    hidden_channels: int
    num_res_blocks: int
    policy_head_channels: int
    value_head_channels: int
    policy_fc1_hidden: int
    value_fc1_hidden: int
    symmetries: Tuple[str, ...]            # 该棋盘合法的空间对称（含 identity）
    rust_prefix: str                       # "" | "mini" | "game4x4" → 绑定函数名前缀
    env_const_prefix: str                  # "" | "MINI_" | "GAME4X4_" → banqi_4x8 常量名前缀
    model_basename: str                    # banqi_model_latest / banqi4x4_model_latest / banqi_mini_model_latest
    archive_dir: str | None                # 冷存储本地目录；None=该变体不归档
    tb_dir: str                            # TensorBoard runs 目录

    # ---------------- 派生量（单一声明源，勿在别处重复） ----------------

    @cached_property
    def total_positions(self) -> int:
        return self.board_rows * self.board_cols

    @cached_property
    def num_active_piece_types(self) -> int:
        return sum(1 for c in self.piece_counts if c > 0)

    @cached_property
    def board_channels(self) -> int:
        """己方(active) + 敌方(active) + hidden + empty。"""
        return 2 * self.num_active_piece_types + 2

    @cached_property
    def total_pieces_per_player(self) -> int:
        return sum(self.piece_counts)

    @cached_property
    def scalar_feature_count(self) -> int:
        """3 全局（步数/我方HP/敌方HP）+ 存活向量×2 + 暗子向量×2。"""
        return 3 + 4 * self.total_pieces_per_player

    @cached_property
    def action_counts(self) -> Tuple[int, int, int]:
        """(reveal, move, cannon)，由动作表推导（与 Rust 一致）。"""
        n_reveal, n_move, n_cannon, _ = count_actions(self.board_rows, self.board_cols)
        return n_reveal, n_move, n_cannon

    @cached_property
    def action_space_size(self) -> int:
        _, _, _, total = count_actions(self.board_rows, self.board_cols)
        return total

    @cached_property
    def base_output_dir(self) -> str:
        """变体输出根目录：python/outputs/{id} 的绝对路径。"""
        import os
        py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(py_dir, "outputs", self.id)

    @cached_property
    def checkpoints_dir(self) -> str:
        """变体模型/检查点目录。"""
        return os.path.join(self.base_output_dir, "checkpoints")

    @cached_property
    def logs_dir(self) -> str:
        """变体运行日志目录。"""
        return os.path.join(self.base_output_dir, "logs")

    @cached_property
    def non_identity_transforms(self) -> Tuple[str, ...]:
        return tuple(t for t in self.symmetries if t != "identity")

    @cached_property
    def involutions(self) -> Tuple[str, ...]:
        return tuple(t for t in self.symmetries if t in INVOLUTION_TRANSFORMS)


def _sym(*names: str) -> Tuple[str, ...]:
    return tuple(names)


VARIANTS: Dict[str, Variant] = {
    "4x8": Variant(
        id="4x8",
        board_rows=4,
        board_cols=8,
        piece_counts=(5, 2, 2, 2, 2, 2, 1),
        piece_values=(2, 5, 5, 5, 5, 10, 30),
        initial_health=60,               # 4x8 目标分制
        hidden_channels=64,
        num_res_blocks=6,
        policy_head_channels=4,
        value_head_channels=4,
        policy_fc1_hidden=512,
        value_fc1_hidden=256,
        symmetries=_sym("identity", "hflip", "vflip", "rot180"),
        rust_prefix="",
        env_const_prefix="",
        model_basename="banqi_model_latest",
        archive_dir="outputs/4x8/archive",
        tb_dir="outputs/4x8/tensorboard",
    ),
    "4x4": Variant(
        id="4x4",
        board_rows=4,
        board_cols=4,
        piece_counts=(2, 1, 1, 1, 1, 1, 1),
        piece_values=(4, 10, 10, 10, 10, 20, 30),
        initial_health=60,               # 4x4 变体指定
        hidden_channels=24,
        num_res_blocks=2,
        policy_head_channels=2,
        value_head_channels=2,
        policy_fc1_hidden=128,
        value_fc1_hidden=64,
        symmetries=_sym(
            "identity", "rot90", "rot180", "rot270", "hflip", "vflip", "diag", "anti_diag",
        ),
        rust_prefix="game4x4",
        env_const_prefix="GAME4X4_",
        model_basename="banqi4x4_model_latest",
        archive_dir="outputs/4x4/archive",
        tb_dir="outputs/4x4/tensorboard",
    ),
    "4x2": Variant(
        id="4x2",
        board_rows=4,
        board_cols=2,
        piece_counts=(1, 1, 0, 0, 0, 1, 1),   # 仅 兵/炮/士/将 激活
        piece_values=(2, 5, 5, 5, 5, 10, 30),
        initial_health=47,               # 单方子力价值之和：2+5+10+30
        hidden_channels=32,
        num_res_blocks=2,
        policy_head_channels=4,
        value_head_channels=4,
        policy_fc1_hidden=128,
        value_fc1_hidden=64,
        symmetries=_sym("identity", "hflip", "vflip", "rot180"),
        rust_prefix="mini",
        env_const_prefix="MINI_",
        model_basename="banqi_mini_model_latest",
        archive_dir="outputs/4x2/archive",
        tb_dir="outputs/4x2/tensorboard",
    ),
}


def get_variant(variant_id: str) -> Variant:
    if variant_id not in VARIANTS:
        raise KeyError(f"未知变体 {variant_id!r}，可选: {sorted(VARIANTS)}")
    return VARIANTS[variant_id]


if __name__ == "__main__":
    for vid, v in VARIANTS.items():
        print(
            f"[banqi_training.variant] {vid}: board=({v.board_channels},{v.board_rows},{v.board_cols}) "
            f"scalar={v.scalar_feature_count} action={v.action_space_size} "
            f"(reveal={v.action_counts[0]}, move={v.action_counts[1]}, cannon={v.action_counts[2]}) "
            f"sym={v.symmetries} params(hid={v.hidden_channels}, res={v.num_res_blocks})"
        )
    print("[banqi_training.variant] all OK")
