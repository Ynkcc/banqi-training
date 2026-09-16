"""banqi_training/reanalysis.py — 局面重搜（reanalysis）的 trainer 侧：位置池 + 载荷编码。

数据流：collector 在自对弈时（`collect_positions=true`）把每步**局面快照**写进 episode；
trainer 把收到的局面攒进有界位置池，周期性打包成载荷提交给调度器，由任意 worker 用
**当前 best 网络**重跑 MCTS，产出新的策略/价值目标（一局面一条 1 样本 episode），
经常规 episode 通道回到训练。收益来自「旧局面 × 更强网络」——同一批自对弈数据被反复榨取。

载荷格式与 Rust 侧 `banqi-collector/src/pipeline/self_play/reanalysis.rs::decode_payload`
**严格镜像**（版本号不符即被拒绝）：

    u8  version(=1) | u32 count |
    每项: u32 snapshot_len | snapshot | u8 flags | i32 winner | f32 health_diff

flags: bit0 = winner 有效，bit1 = health_diff 有效。全部小端。
"""

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

PAYLOAD_VERSION = 1
FLAG_HAS_WINNER = 0b01
FLAG_HAS_HEALTH = 0b10

# 载荷固定布局（< 前缀关闭对齐填充，与 Rust 侧逐字节一致）
_HEADER = struct.Struct("<BI")   # version + count
_ITEM_LEN = struct.Struct("<I")  # snapshot_len
_ITEM_TAIL = struct.Struct("<Bif")  # flags + winner + health_diff


@dataclass(frozen=True)
class ReanalysisItem:
    """一个待重搜局面：快照字节串 + 该局面所属对局的终局信息（回填 game_result 用）。"""

    snapshot: bytes
    winner: Optional[int]
    health_diff_red: Optional[float]


def encode_payload(items: Iterable[ReanalysisItem]) -> bytes:
    """把待重搜项编码为载荷（与 Rust 解码端逐字节对应）。"""
    items = list(items)
    if len(items) > 0xFFFFFFFF:
        raise ValueError(f"位置条数 {len(items)} 超出 u32 上限")
    out = bytearray(_HEADER.pack(PAYLOAD_VERSION, len(items)))
    for item in items:
        flags = 0
        if item.winner is not None:
            flags |= FLAG_HAS_WINNER
        if item.health_diff_red is not None:
            flags |= FLAG_HAS_HEALTH
        out += _ITEM_LEN.pack(len(item.snapshot))
        out += item.snapshot
        out += _ITEM_TAIL.pack(flags, int(item.winner or 0), float(item.health_diff_red or 0.0))
    return bytes(out)


@dataclass
class PoolStats:
    """池子自建以来的累计计数（供日志；读后由 `reset_stats` 清零）。"""

    added: int = 0
    skipped_no_positions: int = 0
    skipped_no_winner: int = 0
    dropped_overflow: int = 0


class PositionPool:
    """有界 FIFO 位置池：先攒后提交，提交成功才移出。

    - 只收「带快照且已分出胜负」的局：`winner=None`（作废局）不能作为价值目标的来源，
      快照缺失说明 collector 未开 `collect_positions`；
    - 容量按「位置条数」计，超出丢**最旧**（最旧的位置在池子里待得最久，但容量本身
      就是「愿意为多老的数据付内存」的预算，故按 FIFO 淘汰而非丢弃最新）；
    - `peek` / `drop_front` 分离：提交被拒（调度器未启用 / 队列满 / RPC 失败）时
      位置保留在池中，下一轮重试，不丢数据。
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"REANALYSIS_POOL_SIZE 必须 > 0，实际 {capacity}")
        self.capacity = int(capacity)
        self._items: Deque[ReanalysisItem] = deque()
        self.stats = PoolStats()

    def add_episode(self, episode: Dict[str, Any]) -> int:
        """把一局 episode 的局面快照收进池子，返回实际入池条数。"""
        positions = episode.get("positions")
        if not positions:
            self.stats.skipped_no_positions += 1
            return 0
        if episode.get("winner") is None:
            self.stats.skipped_no_winner += 1
            return 0
        winner = int(episode["winner"])
        health = episode.get("health_diff_red")
        health_diff = None if health is None else float(health)
        n = 0
        for snapshot in positions:
            self._items.append(ReanalysisItem(bytes(snapshot), winner, health_diff))
            n += 1
        self.stats.added += n
        # 溢出按 FIFO 丢最旧
        if len(self._items) > self.capacity:
            dropped = len(self._items) - self.capacity
            for _ in range(dropped):
                self._items.popleft()
            self.stats.dropped_overflow += dropped
        return n

    def peek(self, n: int) -> List[ReanalysisItem]:
        """查看队首至多 n 条（不移除）。"""
        if n <= 0:
            return []
        return list(self._items)[:n]

    def drop_front(self, n: int) -> None:
        """移除队首 n 条（提交成功后调用）。"""
        for _ in range(min(n, len(self._items))):
            self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)

    def take_stats(self) -> PoolStats:
        """取走并清零累计统计（供每轮日志/诊断）。"""
        stats, self.stats = self.stats, PoolStats()
        return stats
