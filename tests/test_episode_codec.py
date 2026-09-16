"""episode 编解码契约测试：重点锁定 reanalysis 侧信道字段 `positions`。

Rust 侧由 `banqi-collector/src/pipeline/self_play/codec.rs` 编码，本测试直接构造
`EpisodeRecord` 验证解码端行为（不需要 Rust 参与），因此能独立守住两侧契约。
"""

from __future__ import annotations

import numpy as np
import pytest

from banqi_training import episode_codec as ec
from banqi_training.proto import scheduler_pb2 as pb

CHANNELS, ROWS, COLS, SCALAR_COUNT, ACTION_SPACE = 2, 2, 2, 3, 4


def _record(steps: int, snapshots: list[bytes] | None = None) -> "pb.EpisodeRecord":
    """构造一条维度自洽的 EpisodeRecord（全 0 特征，仅用于解码契约验证）。"""
    rec = pb.EpisodeRecord(
        steps=steps,
        board_channels=CHANNELS,
        board_rows=ROWS,
        board_cols=COLS,
        scalar_count=SCALAR_COUNT,
        action_space=ACTION_SPACE,
        boards_bits=bytes(steps * CHANNELS * -(-(ROWS * COLS) // 8)),
        action_masks_bits=bytes(steps * -(-ACTION_SPACE // 8)),
        scalars=np.zeros(steps * SCALAR_COUNT, "<f4").tobytes(),
        policies=np.full(steps * ACTION_SPACE, 1.0 / ACTION_SPACE, "<f4").tobytes(),
        mcts_values=np.zeros(steps, "<f4").tobytes(),
        completed_qs=np.zeros(steps, "<f4").tobytes(),
        game_results=np.zeros(steps, "<f4").tobytes(),
        health_diffs=np.zeros(steps, "<f4").tobytes(),
        root_visits=np.zeros(steps, "<u4").tobytes(),
        actions=np.zeros(steps, "<u4").tobytes(),
        is_full_search=bytes(steps),
        game_length=steps,
        winner=1,
        health_diff_red=0.1,
    )
    if snapshots is not None:
        rec.positions.extend(snapshots)
    return rec


def _batch(record: "pb.EpisodeRecord") -> bytes:
    return pb.EpisodeBatch(
        schema_version=ec.SUPPORTED_SCHEMA_VERSION,
        variant="4x2",
        kind=pb.DATA_RESNET,
        episodes=[record],
    ).SerializeToString()


def test_positions_decoded_when_present() -> None:
    """携带局面快照时按步原样解码（reanalysis 的位置来源）。"""
    body = _batch(_record(2, [b"\x01\x02\x03", b"\x04\x05"]))
    decoded = ec.decode_episode_batch(body, expect_variant="4x2")
    assert decoded.records[0]["positions"] == [b"\x01\x02\x03", b"\x04\x05"]


def test_positions_none_when_absent() -> None:
    """未收集（旧数据 / collect_positions=false）时为 None，而非空列表。"""
    decoded = ec.decode_episode_batch(_batch(_record(1)))
    assert decoded.records[0]["positions"] is None


def test_misaligned_positions_rejected() -> None:
    """快照数与样本数不符必须报错：错位会让重搜目标写到错误的位置上。"""
    body = _batch(_record(2, [b"\x01"]))
    with pytest.raises(ValueError, match="局面快照数"):
        ec.decode_episode_batch(body)
