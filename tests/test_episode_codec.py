"""episode_codec 单元测试：位平面布局 / 张量还原 / 契约校验。

用例的手写字节按「位打包：字节内 MSB 优先」逐位推导，因此能真正卡住
Rust 编码端与 Python 解码端的布局分歧（两边各自独立实现同一份 proto 约定）。
"""

from __future__ import annotations

import numpy as np
import pytest

from banqi_training.episode_codec import SUPPORTED_SCHEMA_VERSION, decode_episode_batch
from banqi_training.proto import scheduler_pb2


def _f32_bytes(values) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _u32_bytes(values) -> bytes:
    return np.asarray(values, dtype="<u4").tobytes()


def _batch(steps: int = 2, **overrides) -> scheduler_pb2.EpisodeBatch:
    """构造一个 2 通道 2x2 棋盘、3 维标量、3 动作的最小批次。

    棋盘（每步 2 通道 × ceil(4/8)=1 字节）：
      第 0 步 ch0=[0,1,0,0] → 0b0100_0000=0x40，ch1 全 0 → 0x00
      第 1 步 ch0=[1,1,1,1] → 0b1111_0000=0xF0，ch1=[0,1,0,1] → 0b0101_0000=0x50
    掩码（每步 ceil(3/8)=1 字节）：
      第 0 步 [1,0,1] → 0b1010_0000=0xA0；第 1 步 [0,1,1] → 0b0110_0000=0x60
    """
    rec = scheduler_pb2.EpisodeRecord(
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
    defaults = dict(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        variant="4x4",
        episodes=[rec],
    )
    defaults.update(overrides)
    return scheduler_pb2.EpisodeBatch(**defaults)


def test_decode_bitplanes_and_arrays() -> None:
    batch = decode_episode_batch(_batch().SerializeToString(), expect_variant="4x4")
    assert batch.variant == "4x4"
    assert len(batch.episodes) == 1
    ep = batch.episodes[0]

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


def test_decode_nnue_features_offsets() -> None:
    # 第 0 步 mover=[3,5]、opponent=[7]；第 1 步 mover=[11]、opponent=[]
    rec = _batch().episodes[0]
    rec.nnue.meta.feature_dim = 64
    rec.nnue.indices = _u32_bytes([3, 5, 7, 11])
    rec.nnue.offsets = _u32_bytes([0, 2, 3, 4, 4])
    batch = decode_episode_batch(_batch(episodes=[rec]).SerializeToString())
    ep = batch.episodes[0]
    assert ep["nnue_meta"]["feature_dim"] == 64
    assert ep["nnue_features"]["mover"] == [[3, 5], [11]]
    assert ep["nnue_features"]["opponent"] == [[7], []]


def test_missing_nnue_features_absent() -> None:
    ep = decode_episode_batch(_batch().SerializeToString()).episodes[0]
    assert "nnue_meta" not in ep
    assert "nnue_features" not in ep


def test_reject_unknown_schema_version() -> None:
    body = _batch(schema_version=SUPPORTED_SCHEMA_VERSION + 1).SerializeToString()
    with pytest.raises(ValueError, match="版本不支持"):
        decode_episode_batch(body)


def test_reject_variant_mismatch() -> None:
    with pytest.raises(ValueError, match="变体不符"):
        decode_episode_batch(_batch().SerializeToString(), expect_variant="4x2")


def test_reject_truncated_buffer() -> None:
    rec = _batch().episodes[0]
    rec.policies = rec.policies[:8]  # 少一个样本的策略
    with pytest.raises(ValueError, match="策略概率 长度不符"):
        decode_episode_batch(_batch(episodes=[rec]).SerializeToString())


def test_reject_bad_nnue_offsets() -> None:
    rec = _batch().episodes[0]
    rec.nnue.meta.feature_dim = 64
    rec.nnue.indices = _u32_bytes([3, 5, 7])
    rec.nnue.offsets = _u32_bytes([0, 2, 2, 3, 4])  # offsets[-1] != indices.size
    with pytest.raises(ValueError, match="偏移"):
        decode_episode_batch(_batch(episodes=[rec]).SerializeToString())
