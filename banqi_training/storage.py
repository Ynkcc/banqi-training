"""banqi_training/storage.py — 冷存储（本地归档目录）读取。

冷存储目录存放从 R2 取回的 episode 对象（`*.epb.gz`，即 gzip 压缩的
EpisodeBatch 二进制，与在线链路同一格式与同一解码实现）——归档不再另立
一份 JSON 契约，避免两套格式各自漂移。

原 FileSaver / MongoSaver / JSONL 读写器已随「弃用 JSON episode 格式」移除：
本地采集与 Mongo 归档都不在当前闭环内（分布式形态由 collector 直传 R2）。
"""

from __future__ import annotations

import gzip
import os
from typing import Dict, Iterator, List, Optional

from banqi_training.episode_codec import decode_episode_batch

EPISODE_SUFFIX = ".epb.gz"


def list_episode_objects(archive_dir: str) -> List[str]:
    """列出归档目录下的 episode 对象文件（升序，保证登记顺序）。"""
    if not os.path.isdir(archive_dir):
        return []
    return sorted(
        os.path.join(archive_dir, f)
        for f in os.listdir(archive_dir)
        if f.endswith(EPISODE_SUFFIX)
    )


def iter_episodes_from_dir(archive_dir: str, variant: Optional[str] = None) -> Iterator[Dict]:
    """流式迭代归档目录中的 episode dict（逐个对象解码，不一次性物化全部）。

    单个对象损坏/版本不兼容时跳过并告警，不中断整批加载。
    """
    for path in list_episode_objects(archive_dir):
        try:
            with open(path, "rb") as f:
                batch = decode_episode_batch(gzip.decompress(f.read()), expect_variant=variant)
        except Exception as exc:  # noqa: BLE001
            print(f"[storage] ⚠️ 跳过无法解码的归档对象 {path}: {exc}")
            continue
        yield from batch.episodes


def load_episodes_from_dir(
    archive_dir: str, limit_games: Optional[int] = None, variant: Optional[str] = None
) -> List[Dict]:
    """加载归档目录中的 episode dict 列表；limit_games 限制局数（控制内存与数据分布）。"""
    episodes: List[Dict] = []
    for ep in iter_episodes_from_dir(archive_dir, variant=variant):
        episodes.append(ep)
        if limit_games is not None and len(episodes) >= limit_games:
            break
    return episodes
