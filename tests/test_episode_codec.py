"""episode_codec 单元测试：位平面布局 / 张量还原 / 契约与类别校验。

用例的手写字节按「位打包：字节内 MSB 优先」逐位推导，因此能真正卡住
Rust 编码端与 Python 解码端的布局分歧（两边各自独立实现同一份 proto 约定）。
"""

from __future__ import annotations

import numpy as np
import pytest

from banqi_training.episode_codec import (
    DATA_NNUE,
    DATA_RESNET,
    SUPPORTED_SCHEMA_VERSION,
    decode_episode_batch,
)
from banqi_training.proto import scheduler_pb2


def _f32_bytes(values) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _u32_bytes(values) -> bytes:
    return np.asarray(values, dtype="<u4").tobytes()


def _episode_record(steps: int = 2) -> scheduler_pb2.EpisodeRecord:
    """构造一个 2 通道 2x2 棋盘、3 维标量、3 动作的最小 ResNet 记录。

    棋盘（每步 2 通道 × ceil(4/8)=1 字节）：
      第 0 步 ch0=[0,1,0,0] → 0b0100_0000=0x40，ch1 全 0 → 0x00
      第 1 步 ch0=[1,1,1,1] → 0b1111_0000=0xF0，ch1=[0,1,0,1] → 0b0101_0000=0x50
    掩码（每步 ceil(3/8)=1 字节）：
      第 0 步 [1,0,1] → 0b1010_0000=0xA0；第 1 步 [0,1,1] → 0b0110_0000=0x60
    """
    return scheduler_pb2.EpisodeRecord(
        steps=steps,
        board_channels=2,
        board_rows=2,
        board_cols=2,
        scalar_count=3,
        action_space=3,
        boards_bits=bytes([0x40, 0x00, 0xF0, 0x50]),
        scalars=_f32_bytes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        policies=_f32_bytes([0.5, 0.25, 0.25, 1.0, 0.0, 0.0]),
        action_masks_bits=bytes([0xA0, 0x60]),
        mcts_values=_f32_bytes([0.1, 0.2]),
        completed_qs=_f32_bytes([-0.1, 0.2]),
        game_results=_f32_bytes([1.0, -1.0]),
        health_diffs=_f32_bytes([0.3, -0.3]),
        root_visits=_u32_bytes([7, 9]),
        actions=_u32_bytes([2, 1]),
        is_full_search=bytes([1, 0]),
        game_length=steps,
        winner=1,
        health_diff_red=0.3,
    )


def _nnue_record() -> scheduler_pb2.NnueEpisodeRecord:
    """2 步 NNUE 记录：第 0 步 mover=[3,5]/opponent=[7]，第 1 步 mover=[11]/opponent=[]。"""
    rec = scheduler_pb2.NnueEpisodeRecord(
        steps=2,
        search_values=_f32_bytes([0.25, -0.5]),
        players=np.asarray([1, -1], dtype="<i4").tobytes(),
        actions=_u32_bytes([4, 9]),
        game_length=2,
        winner=-1,
    )
    rec.features.meta.feature_dim = 64
    rec.features.meta.total_positions = 8
    rec.features.indices = _u32_bytes([3, 5, 7, 11])
    rec.features.offsets = _u32_bytes([0, 2, 3, 4, 4])
    return rec


def _batch(kind: int = DATA_RESNET, **overrides) -> scheduler_pb2.EpisodeBatch:
    if kind == DATA_RESNET:
        body = dict(episodes=[_episode_record()])
    else:
        body = dict(nnue_episodes=[_nnue_record()])
    defaults = dict(schema_version=SUPPORTED_SCHEMA_VERSION, variant="4x4", kind=kind, **body)
    defaults.update(overrides)
    return scheduler_pb2.EpisodeBatch(**defaults)


def test_decode_bitplanes_and_arrays() -> None:
    batch = decode_episode_batch(_batch().SerializeToString(), expect_variant="4x4")
    assert batch.variant == "4x4"
    assert batch.kind == DATA_RESNET
    assert len(batch.records) == 1
    ep = batch.records[0]

    assert ep["num_samples"] == 2
    assert ep["game_length"] == 2
    assert ep["winner"] == 1
    assert ep["health_diff_red"] == pytest.approx(0.3)

    boards = ep["boards"]
    assert boards.shape == (2, 2, 2, 2)
    assert boards.dtype == np.float32
    assert boards[0, 0].tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert boards[0, 1].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert boards[1, 0].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert boards[1, 1].tolist() == [[0.0, 1.0], [0.0, 1.0]]

    assert ep["scalars"].shape == (2, 3)
    assert ep["policies"].shape == (2, 3)
    assert ep["policies"][1].tolist() == [1.0, 0.0, 0.0]
    assert ep["action_masks"][0].tolist() == [1, 0, 1]
    assert ep["action_masks"][1].tolist() == [0, 1, 1]
    assert ep["root_visits"].tolist() == [7, 9]
    assert ep["actions"].tolist() == [2, 1]
    assert ep["is_full_search"].tolist() == [True, False]
    assert ep["game_results"][0] == pytest.approx(1.0)


def test_decode_nnue_batch() -> None:
    batch = decode_episode_batch(
        _batch(kind=DATA_NNUE).SerializeToString(), expect_variant="4x4", expect_kind=DATA_NNUE
    )
    assert batch.kind == DATA_NNUE
    assert len(batch.records) == 1
    rec = batch.records[0]
    assert rec["num_samples"] == 2
    assert rec["game_length"] == 2
    assert rec["winner"] == -1
    assert rec["nnue_meta"]["feature_dim"] == 64
    assert rec["nnue_features"]["mover"] == [[3, 5], [11]]
    assert rec["nnue_features"]["opponent"] == [[7], []]
    assert rec["players"].tolist() == [1, -1]
    assert rec["actions"].tolist() == [4, 9]
    assert rec["search_values"].tolist() == [pytest.approx(0.25), pytest.approx(-0.5)]
    assert rec["is_full_search"].tolist() == [True, True]


def test_reject_unknown_schema_version() -> None:
    body = _batch(schema_version=SUPPORTED_SCHEMA_VERSION + 1).SerializeToString()
    with pytest.raises(ValueError, match="版本不支持"):
        decode_episode_batch(body)


def test_reject_variant_mismatch() -> None:
    with pytest.raises(ValueError, match="变体不符"):
        decode_episode_batch(_batch().SerializeToString(), expect_variant="4x2")


def test_reject_kind_mismatch() -> None:
    """训练端只消费 ResNet 时，NNUE 对象必须被拒绝（不能静默 yield 0 条）。"""
    with pytest.raises(ValueError, match="数据类别不符"):
        decode_episode_batch(
            _batch(kind=DATA_NNUE).SerializeToString(), expect_kind=DATA_RESNET
        )


def test_reject_kind_content_inconsistency() -> None:
    """类别与内容必须自洽：resnet 带 NNUE 记录、声明类别那侧为空、类别未知，都要报错。"""
    with pytest.raises(ValueError, match="却携带"):
        decode_episode_batch(
            _batch(
                kind=DATA_RESNET,
                episodes=[_episode_record()],
                nnue_episodes=[_nnue_record()],
            ).SerializeToString()
        )
    with pytest.raises(ValueError, match="没有任何 EpisodeRecord"):
        decode_episode_batch(_batch(kind=DATA_RESNET, episodes=[]).SerializeToString())
    unknown = scheduler_pb2.EpisodeBatch(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        variant="4x4",
        kind=7,
        episodes=[_episode_record()],
    )
    with pytest.raises(ValueError, match="未知数据类别"):
        decode_episode_batch(unknown.SerializeToString())


def test_reject_truncated_buffer() -> None:
    rec = _episode_record()
    rec.policies = rec.policies[:8]  # 少一个样本的策略
    with pytest.raises(ValueError, match="策略概率 长度不符"):
        decode_episode_batch(_batch(episodes=[rec]).SerializeToString())


def test_reject_bad_nnue_offsets() -> None:
    rec = _nnue_record()
    rec.features.indices = _u32_bytes([3, 5, 7])
    rec.features.offsets = _u32_bytes([0, 2, 2, 3, 4])  # offsets[-1] != indices.size
    with pytest.raises(ValueError, match="偏移"):
        decode_episode_batch(_batch(kind=DATA_NNUE, nnue_episodes=[rec]).SerializeToString())
