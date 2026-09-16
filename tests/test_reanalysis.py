"""局面重搜（reanalysis）trainer 侧测试：载荷字节布局 + 位置池语义。

载荷布局必须与 Rust 解码端（banqi-collector/src/pipeline/self_play/reanalysis.rs）
逐字节一致，否则跨进程重搜会在运行期整批报错。这里用固定的期望字节串把契约钉死。
"""

from __future__ import annotations

import struct

import pytest

from banqi_training.reanalysis import (
    PAYLOAD_VERSION,
    PositionPool,
    ReanalysisItem,
    encode_payload,
)


def _hex(payload: bytes) -> str:
    return payload.hex()


def test_payload_layout_matches_rust_decoder() -> None:
    """固定样例的期望字节串（与 Rust 侧同一常量，两侧测试互为锁）。"""
    items = [ReanalysisItem(snapshot=b"\x01\x02", winner=1, health_diff_red=0.5)]
    expected = (
        "01"          # version = 1
        "01000000"    # count = 1
        "02000000"    # snapshot_len = 2
        "0102"        # snapshot
        "03"          # flags: winner + health 均有效
        "01000000"    # winner = 1
        "0000003f"    # health_diff = 0.5 (f32 LE)
    )
    assert _hex(encode_payload(items)) == expected


def test_payload_flags_mark_absent_fields() -> None:
    """winner / health 缺失时按位不置，值域留 0（解码端据此还原 None）。"""
    payload = encode_payload([ReanalysisItem(snapshot=b"", winner=None, health_diff_red=None)])
    assert payload[:1] == bytes([PAYLOAD_VERSION])
    assert payload[1:5] == (1).to_bytes(4, "little")
    assert payload[5:9] == (0).to_bytes(4, "little")  # snapshot_len = 0
    assert payload[9] == 0  # flags 全 0
    assert payload[10:14] == (0).to_bytes(4, "little", signed=True)
    assert payload[14:18] == struct.pack("<f", 0.0)


def _episode(snapshots, winner=1, health=0.25):
    return {
        "positions": [bytes([i, i]) for i in snapshots] if snapshots is not None else None,
        "winner": winner,
        "health_diff_red": health,
    }


def test_pool_rejects_episodes_without_positions_or_winner() -> None:
    """无快照（collector 未开 collect_positions）与作废局（winner=None）都要被跳过并计数。"""
    pool = PositionPool(100)
    assert pool.add_episode(_episode(None)) == 0
    assert pool.add_episode(_episode([1, 2], winner=None)) == 0
    assert len(pool) == 0
    stats = pool.take_stats()
    assert stats.skipped_no_positions == 1
    assert stats.skipped_no_winner == 1
    # 统计读后清零
    assert pool.take_stats().skipped_no_positions == 0


def test_pool_fifo_eviction_and_peek_drop() -> None:
    """容量溢出丢最旧；peek 不移除、drop_front 才移除（提交失败可保留重试）。"""
    pool = PositionPool(4)
    pool.add_episode(_episode([1, 2, 3]))  # 3 条
    pool.add_episode(_episode([4, 5, 6]))  # 共 6 → 超容量丢最旧
    assert len(pool) == 4
    assert pool.take_stats().dropped_overflow == 2

    head = pool.peek(2)
    assert len(head) == 2
    assert head[0].snapshot == bytes([3, 3]), "队首应为最旧一条"
    assert len(pool) == 4, "peek 不应移除"
    pool.drop_front(2)
    assert len(pool) == 2
    assert pool.peek(10)[0].snapshot == bytes([5, 5])

    with pytest.raises(ValueError):
        PositionPool(0)
