"""banqi/training/augment.py — 空间对称数据增强（Rust 绑定执行）。

把 episode dict 的 board 特征重排、policy / action_mask 按动作置换表 gather、
动作索引置换，全部下沉到 Rust（banqi_4x8），Python 侧只做调度与缓存。
"""

from __future__ import annotations

import random
from typing import Dict, List

from banqi_training.constants import build_constants
from banqi_training.rust_bridge import (
    get_action_symmetry_table,
    transform_action,
    transform_board,
    transform_policy,
)
from banqi_training.variant import Variant


class EpisodeAugmenter:
    """按变体配置对 episode dict 做空间对称增强（动作置换表带缓存）。"""

    def __init__(self, variant: Variant, cfg) -> None:
        self.variant = variant
        self.cfg = cfg
        self.C = build_constants(variant)
        self._perm_cache: Dict[str, list] = {}

    def permutation(self, transform: str) -> list:
        """获取 Rust 导出的动作置换表（new_policy = old_policy[perm]），带缓存。"""
        perm = self._perm_cache.get(transform)
        if perm is None:
            perm = get_action_symmetry_table(
                self.C.BOARD_ROWS, self.C.BOARD_COLS, transform
            )
            self._perm_cache[transform] = perm
        return perm

    def transform_episode(self, episode_dict: Dict, transform: str) -> Dict:
        """对一个 episode dict 做空间对称增强（全部由 Rust 绑定执行）。"""
        out = dict(episode_dict)
        perm = self.permutation(transform)
        rows, cols = self.C.BOARD_ROWS, self.C.BOARD_COLS
        channels = self.C.TOTAL_INPUT_CHANNELS
        # board 特征空间重排（Rust）
        out["boards"] = [
            transform_board(
                list(b), rows, cols, channels, transform
            )
            for b in out["boards"]
        ]
        # policy / action_mask 按置换表 gather（Rust 提供 gather）
        def _gather(p):
            return transform_policy(list(p), perm)
        out["policies"] = [_gather(p) for p in out["policies"]]
        out["action_masks"] = [_gather(m) for m in out["action_masks"]]
        if out.get("actions"):
            out["actions"] = [
                int(transform_action(a, perm)) for a in out["actions"]
            ]
        return out

    def augment(self, episode_dict: Dict) -> List[Dict]:
        """按 config 对 episode 做空间对称增强。

        返回用于训练的 episode dict 列表：
          - DATA_AUGMENT_ENABLED=false：原样返回 [episode_dict]。
          - 开启时：对每局按 DATA_AUGMENT_TRANSFORMS 随机抽一个非恒等变换，
            生成增强副本；DATA_AUGMENT_KEEP_ORIGINAL=true 时保留原始局。
        """
        cfg = self.cfg
        if not cfg.DATA_AUGMENT_ENABLED:
            return [episode_dict]
        transforms = cfg.DATA_AUGMENT_TRANSFORMS or ""
        if transforms:
            transform_list = [
                t.strip() for t in transforms.split(",") if t.strip()
            ]
        else:
            transform_list = list(self.variant.non_identity_transforms)
        # 只保留该变体合法的非恒等变换
        valid = set(self.variant.non_identity_transforms)
        transform_list = [t for t in transform_list if t in valid]
        if not transform_list:
            return [episode_dict]
        keep = cfg.DATA_AUGMENT_KEEP_ORIGINAL
        # 每局随机抽 1 个变换（训练侧增强多样性），并保留原始局
        t = transform_list[random.randrange(len(transform_list))]
        out = [episode_dict] if keep else []
        out.append(self.transform_episode(episode_dict, t))
        return out
