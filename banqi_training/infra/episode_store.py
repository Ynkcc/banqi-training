"""banqi_training/infra/episode_store.py — EpisodeStore Protocol + Scheduler 实现。

分布式形态：worker（banqi-collector）把 episode 直传 R2，Trainer 经
Go scheduler 的 gRPC ListEpisodes 拿预签名 GET 列表拉取解析。
记录格式（EpisodeBatch 二进制，proto 契约）由 banqi_training.episode_codec 解码。
"""

from __future__ import annotations

import gzip
import os
import time
from collections import deque
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol

from banqi_training.episode_codec import decode_episode_batch


class EpisodeStore(Protocol):
    """episode 数据出口/入口统一接口（L4）。"""

    def put(self, episodes: Iterable[Dict[str, Any]]) -> int:
        """写入一批 episode，返回写入局数。"""
        ...

    def iter_new_episodes(self) -> Iterator[Dict[str, Any]]:
        """迭代尚未消费过的 episode（按批次文件序）。"""
        ...


class SchedulerEpisodeStore:
    """分布式实现：经 Go scheduler 拉取 worker 直传的 episode 批次。

    R2 凭据只在调度器持有：本类调 gRPC ListEpisodes 获取预签名 GET 列表
    （游标分页，服务端按 episode 登记顺序推进），再经 HTTP 下载解析。对象键布局
    `episodes/<network_sha>/<data_id>.epb.gz`。trainer 零存储配置，
    仅需 SCHEDULER_ENDPOINT。

    variant 非空时校验每条记录的变体标签，防止跨变体数据混入训练。
    """

    PAGE_LIMIT = 200

    def __init__(
        self,
        variant: Optional[str] = None,
        endpoint: Optional[str] = None,
        poll_interval: float = 5.0,
    ) -> None:
        import grpc

        from banqi_training.proto import scheduler_pb2, scheduler_pb2_grpc

        self.variant = variant
        self.endpoint = endpoint or os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
        self.poll_interval = poll_interval
        self._pb2 = scheduler_pb2
        # grpc.insecure_channel 只接受 host:port，不接受 URL scheme 前缀
        target = self.endpoint.split("://", 1)[-1]
        self._stub = scheduler_pb2_grpc.SchedulerServiceStub(grpc.insecure_channel(target))
        self._cursor: str = ""  # object_key 游标：服务端据此定位登记顺序，该键已消费
        self._pending: List[tuple] = []  # 已列出未下载的 (key, url)
        # 已下载对象的解析结果缓存：一个对象含多局 episode，调用方逐个消费
        # （get() 只取首项），必须缓存未消费的余项，否则对象键已出队、
        # 余下对局会被静默丢弃。
        self._buffered: deque = deque()

    # ---- 写入端（trainer 不产 episode，占位实现满足 Protocol） ----

    def put(self, episodes: Iterable[Dict[str, Any]]) -> int:
        raise NotImplementedError("SchedulerEpisodeStore 仅作消费端；写入经 collector 直传")

    # ---- 读取端 ----

    def _list_page(self) -> None:
        reply = self._stub.ListEpisodes(
            self._pb2.ListEpisodesRequest(after_key=self._cursor, limit=self.PAGE_LIMIT)
        )
        for obj in reply.objects:
            self._pending.append((obj.object_key, obj.download_url))
        # 服务端按登记顺序递增返回，始终推进游标防止重复取页
        if reply.objects:
            self._cursor = reply.objects[-1].object_key

    def _download(self, url: str) -> bytes:
        import urllib.request

        with urllib.request.urlopen(url, timeout=120) as resp:
            return resp.read()

    def iter_new_episodes(self) -> Iterator[Dict[str, Any]]:
        while True:
            while self._buffered:
                yield self._buffered.popleft()
            if not self._pending:
                self._list_page()
            if not self._pending:
                return
            key, url = self._pending.pop(0)
            try:
                batch = decode_episode_batch(
                    gzip.decompress(self._download(url)), expect_variant=self.variant
                )
                # 解码顺序即样本顺序：先缓冲完本对象，再逐条交给调用方
                self._buffered.extend(batch.episodes)
                if batch.nnue_episodes:
                    print(
                        f"[EpisodeStore] ⚠️ 对象 {key} 含 {len(batch.nnue_episodes)} 局 "
                        f"NNUE 专属 episode，本训练端暂未消费（NNUE 蒸馏未接通），已丢弃"
                    )
            except Exception as exc:
                print(f"[EpisodeStore] ⚠️ 跳过无法解码的对象 {key}: {exc}")

    def drain(self) -> List[Dict[str, Any]]:
        return list(self.iter_new_episodes())

    # ---- 队列语义（供 TrainWorker 直连） ----

    def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for ep in self.iter_new_episodes():
                return ep
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"SchedulerEpisodeStore.get 超时（{timeout}s）：无新 episode")
            time.sleep(self.poll_interval)

    def get_nowait(self) -> Dict[str, Any]:
        it = self.iter_new_episodes()
        try:
            return next(it)
        except StopIteration:
            raise TimeoutError("SchedulerEpisodeStore.get_nowait：无新 episode") from None

    def qsize(self) -> int:
        while True:
            before = len(self._pending)
            self._list_page()
            if len(self._pending) == before:  # 服务端已取尽
                break
        return len(self._pending) + len(self._buffered)
