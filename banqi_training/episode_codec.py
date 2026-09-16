"""banqi_training/episode_codec.py — 训练数据记录解码（scheduler.proto 契约）。

collector 把一批 episode 编码为 `EpisodeBatch` 二进制（Rust 侧唯一实现见
banqi-collector/src/pipeline/self_play/codec.rs）→ gzip → 直传 R2。本模块是该
格式在训练侧的唯一解码实现，取代原先三处各自为政的 JSON 解析。

数据类别（`kind`，服务端下发）两类互斥：`DATA_RESNET` 只带 `episodes`
（Gumbel MCTS 自对弈的稠密特征），`DATA_NNUE` 只带 `nnue_episodes`
（Expectimax 强自对弈的稀疏特征）。解码时强制该不变式：类别与实际记录不符、
或声明类别的那一侧为空，一律抛错（错位的训练数据比没有数据更危险）。

张量布局（与 proto 注释一致）：
- boards_bits      : [步][通道][位置字节] 位平面，字节内 MSB 优先，尾部补 0
- action_masks_bits: [步][动作字节] 位图，位序同上
- 其余 bytes 字段  : 稠密小端缓冲区（f32 / u32 / u8），np.frombuffer 零拷贝还原
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from banqi_training.proto import scheduler_pb2

# 本训练端能正确解码的 schema 版本（proto 中 SCHEMA_VERSION 的镜像）
SUPPORTED_SCHEMA_VERSION = 1

# 数据类别（与 proto DataKind 同值）：训练端消费哪类由调用方显式声明
DATA_RESNET = scheduler_pb2.DATA_RESNET
DATA_NNUE = scheduler_pb2.DATA_NNUE

_KIND_NAMES = {DATA_RESNET: "resnet", DATA_NNUE: "nnue"}


def kind_name(kind: int) -> str:
    return _KIND_NAMES.get(kind, f"unknown({kind})")


@dataclass
class DecodedBatch:
    """一批解码后的训练记录。records 只含声明类别的那一类（另一类恒为空）。"""

    variant: str
    kind: int
    records: List[Dict[str, Any]] = field(default_factory=list)


def decode_episode_batch(
    body: bytes,
    expect_variant: Optional[str] = None,
    expect_kind: Optional[int] = None,
) -> DecodedBatch:
    """解码一个 R2 对象（gzip 解压后的 EpisodeBatch 二进制）。

    - expect_variant 非空时校验变体一致，防止别的变体的 episode 混进训练；
    - expect_kind 非空时校验数据类别一致，防止把 NNUE 数据喂进 ResNet 训练循环。
    """
    batch = scheduler_pb2.EpisodeBatch.FromString(body)
    if batch.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"episode 记录版本不支持: {batch.schema_version}"
            f"（本训练端支持 {SUPPORTED_SCHEMA_VERSION}），请同步升级 banqi-training"
        )
    if not batch.variant:
        raise ValueError("episode 记录未标注变体（variant 为空）")
    if expect_variant and batch.variant != expect_variant:
        raise ValueError(
            f"episode 变体不符: 记录为 {batch.variant}，训练端为 {expect_variant}"
        )
    kind = batch.kind
    if expect_kind is not None and kind != expect_kind:
        raise ValueError(
            f"episode 数据类别不符: 记录为 {kind_name(kind)}，训练端消费 {kind_name(expect_kind)}"
        )

    # 类别与内容必须自洽（两类互斥）：声明哪类就只该有哪类，且不能为空
    if kind == DATA_RESNET:
        if batch.nnue_episodes:
            raise ValueError(f"类别为 resnet，却携带 {len(batch.nnue_episodes)} 局 NNUE 记录")
        if not batch.episodes:
            raise ValueError("类别为 resnet，但对象内没有任何 EpisodeRecord")
        return DecodedBatch(
            variant=batch.variant, kind=kind, records=[_decode_episode(r) for r in batch.episodes]
        )
    if kind == DATA_NNUE:
        if batch.episodes:
            raise ValueError(f"类别为 nnue，却携带 {len(batch.episodes)} 局 ResNet 记录")
        if not batch.nnue_episodes:
            raise ValueError("类别为 nnue，但对象内没有任何 NnueEpisodeRecord")
        return DecodedBatch(
            variant=batch.variant,
            kind=kind,
            records=[_decode_nnue_episode(r) for r in batch.nnue_episodes],
        )
    raise ValueError(f"未知数据类别 kind={kind}（本训练端仅认识 {sorted(_KIND_NAMES)}）")


# ============================================================================
# ResNet（Gumbel MCTS）episode
# ============================================================================

def _decode_episode(rec: "scheduler_pb2.EpisodeRecord") -> Dict[str, Any]:
    steps = int(rec.steps)
    if steps <= 0:
        raise ValueError("episode 记录 steps=0（空记录不应出现在载荷中）")
    channels, rows, cols = int(rec.board_channels), int(rec.board_rows), int(rec.board_cols)
    scalar_count, action_space = int(rec.scalar_count), int(rec.action_space)
    if min(channels, rows, cols, scalar_count, action_space) <= 0:
        raise ValueError(
            f"episode 维度非法: channels={channels} rows={rows} cols={cols} "
            f"scalar={scalar_count} action_space={action_space}"
        )

    positions = rows * cols
    boards = _decode_bits(rec.boards_bits, steps * channels, positions, "棋盘位平面")
    masks = _decode_bits(rec.action_masks_bits, steps, action_space, "动作掩码")

    # 局面快照（reanalysis 侧信道）：空 = 本批未收集（None）；非空必须与步数严格对齐，
    # 否则重搜会把目标写到错误的位置上，宁可丢一批。
    snapshots = [bytes(s) for s in rec.positions]
    if snapshots and len(snapshots) != steps:
        raise ValueError(f"局面快照数 {len(snapshots)} 与样本数 {steps} 不一致")

    return {
        "num_samples": steps,
        "game_length": int(rec.game_length),
        "winner": rec.winner if rec.HasField("winner") else None,
        "health_diff_red": rec.health_diff_red if rec.HasField("health_diff_red") else None,
        "boards": boards.reshape(steps, channels, rows, cols).astype(np.float32),
        "scalars": _f32(rec.scalars, steps * scalar_count, "标量特征").reshape(steps, scalar_count),
        "policies": _f32(rec.policies, steps * action_space, "策略概率").reshape(steps, action_space),
        "action_masks": masks.astype(np.int32),
        "mcts_values": _f32(rec.mcts_values, steps, "MCTS 根价值"),
        "completed_qs": _f32(rec.completed_qs, steps, "completed_Q"),
        "game_results": _f32(rec.game_results, steps, "终局回报"),
        "health_diffs": _f32(rec.health_diffs, steps, "血量差"),
        "root_visits": _u32(rec.root_visits, steps, "根访问次数"),
        "actions": _u32(rec.actions, steps, "动作索引"),
        "is_full_search": _u8(rec.is_full_search, steps, "Full Search 标记").astype(bool),
        "positions": snapshots or None,
    }


# ============================================================================
# NNUE（Expectimax 强自对弈）episode
# ============================================================================

def _decode_nnue_episode(rec: "scheduler_pb2.NnueEpisodeRecord") -> Dict[str, Any]:
    steps = int(rec.steps)
    if steps <= 0:
        raise ValueError("NNUE episode 记录 steps=0")
    if not rec.HasField("features"):
        raise ValueError("NNUE episode 记录缺少 features")
    return {
        "num_samples": steps,
        "game_length": int(rec.game_length),
        "winner": rec.winner if rec.HasField("winner") else None,
        **_decode_nnue_features(rec.features, steps),
        "search_values": _f32(rec.search_values, steps, "搜索值"),
        "mcts_values": _f32(rec.search_values, steps, "搜索值"),
        "completed_qs": _f32(rec.search_values, steps, "搜索值"),
        "players": _i32(rec.players, steps, "行棋方"),
        "actions": _u32(rec.actions, steps, "动作索引"),
        "is_full_search": np.ones(steps, dtype=bool),
    }


def _decode_nnue_features(feats: "scheduler_pb2.NnueFeatures", steps: int) -> Dict[str, Any]:
    """还原双视角稀疏特征：索引保持「每步一个整数列表」的契约（供 NNUE 蒸馏消费）。"""
    offsets = _u32(feats.offsets, 2 * steps + 1, "NNUE 特征偏移")
    indices = _u32(feats.indices, None, "NNUE 特征索引")
    if offsets[0] != 0 or offsets[-1] != indices.size or np.any(
        np.diff(offsets.astype(np.int64)) < 0
    ):
        raise ValueError(
            f"NNUE 特征偏移非单调前缀和: offsets[0]={offsets[0]} offsets[-1]={offsets[-1]} "
            f"indices={indices.size}"
        )
    meta = feats.meta
    if indices.size and indices.max() >= meta.feature_dim:
        raise ValueError(f"NNUE 特征索引 {indices.max()} 超出 feature_dim={meta.feature_dim}")
    movers, opponents = [], []
    for i in range(steps):
        movers.append(indices[offsets[2 * i]:offsets[2 * i + 1]].tolist())
        opponents.append(indices[offsets[2 * i + 1]:offsets[2 * i + 2]].tolist())
    return {
        "nnue_meta": {
            "feature_dim": int(meta.feature_dim),
            "states_per_square": int(meta.states_per_square),
            "bag_stride": int(meta.bag_stride),
            "num_active": int(meta.num_active),
            "total_positions": int(meta.total_positions),
        },
        "nnue_features": {"mover": movers, "opponent": opponents},
    }


# ============================================================================
# 缓冲区还原工具
# ============================================================================

def _f32(buf: bytes, count: Optional[int], what: str) -> np.ndarray:
    return _from_buffer(buf, "<f4", count, what)


def _u32(buf: bytes, count: Optional[int], what: str) -> np.ndarray:
    return _from_buffer(buf, "<u4", count, what)


def _i32(buf: bytes, count: Optional[int], what: str) -> np.ndarray:
    return _from_buffer(buf, "<i4", count, what)


def _u8(buf: bytes, count: Optional[int], what: str) -> np.ndarray:
    return _from_buffer(buf, "u1", count, what)


def _from_buffer(buf: bytes, dtype: str, count: Optional[int], what: str) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=dtype)
    if count is not None and arr.size != count:
        raise ValueError(
            f"{what} 长度不符: 实际 {arr.size} 个元素，契约要求 {count} "
            f"（{arr.size * arr.dtype.itemsize} vs {count * arr.dtype.itemsize} 字节）"
        )
    return arr


def _decode_bits(buf: bytes, frames: int, width: int, what: str) -> np.ndarray:
    """还原 (frames, width) 的 0/1 位平面（字节内 MSB 优先）。"""
    frame_bytes = -(-width // 8)
    raw = np.frombuffer(buf, dtype=np.uint8)
    if raw.size != frames * frame_bytes:
        raise ValueError(
            f"{what} 长度不符: 实际 {raw.size} 字节，契约要求 {frames}×{frame_bytes}="
            f"{frames * frame_bytes} 字节"
        )
    bits = np.unpackbits(raw, axis=0).reshape(frames, frame_bytes * 8)
    return bits[:, :width]
