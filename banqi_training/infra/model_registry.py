"""banqi_training/infra/model_registry.py — ModelRegistry Protocol + Scheduler 实现。

分布式形态：R2 凭据只在调度器持有，trainer 经 gRPC 签发预签名 URL
直传模型 + RegisterNetwork 登记（gatekeeper 判停晋级）。
"""

from __future__ import annotations

import os
from typing import Optional, Protocol


def scheduler_variant(endpoint: Optional[str] = None) -> str:
    """从调度器 GetInfo 获取变体 id（trainer 启动时无需命令行传入变体）。"""
    import grpc

    from banqi_training.proto import scheduler_pb2, scheduler_pb2_grpc

    endpoint = endpoint or os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
    target = endpoint.split("://", 1)[-1]
    stub = scheduler_pb2_grpc.SchedulerServiceStub(grpc.insecure_channel(target))
    variant = stub.GetInfo(scheduler_pb2.GetInfoRequest()).variant
    if not variant:
        raise ValueError(f"调度器未下发变体（GetInfo.variant 为空）: {endpoint}，请升级调度器并配置 SCHEDULER_VARIANT")
    return variant


def scheduler_should_stop(
    endpoint: Optional[str] = None, timeout: float = 5.0
) -> tuple[bool, str]:
    """轮询调度器的绝对强度停机信号（GetInfo.should_stop），返回 (是否停止, 原因)。

    调度器按「连续 N 次评测提升不足」置位该信号（见 banqi-scheduler/internal/scheduler/eval.go）。
    任何异常都按**不停止**处理：停机轮询失败绝不能让训练意外中断（网络抖动是常态）。
    每次调用新建 channel 并在 finally 关闭，避免长跑下 fd/subchannel 泄漏。
    """
    import grpc

    from banqi_training.proto import scheduler_pb2, scheduler_pb2_grpc

    endpoint = endpoint or os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
    target = endpoint.split("://", 1)[-1]
    channel = grpc.insecure_channel(target)
    try:
        stub = scheduler_pb2_grpc.SchedulerServiceStub(channel)
        reply = stub.GetInfo(scheduler_pb2.GetInfoRequest(), timeout=timeout)
        return bool(reply.should_stop), str(reply.stop_reason)
    except Exception as exc:  # noqa: BLE001 — 轮询失败按「不停止」处理
        return False, f"GetInfo 轮询失败: {exc}"
    finally:
        channel.close()


class ModelRegistry(Protocol):
    """模型版本与准入接口（L4）。"""

    def latest_model_path(self) -> Optional[str]:
        """当前 best 模型路径；尚无模型时返回 None。"""
        ...

    def publish(self, model_path: str) -> None:
        """把新导出的模型登记为 best。"""
        ...


class SchedulerModelRegistry:
    """分布式实现：经调度器签发预签名 URL 直传模型 + RegisterNetwork 登记。

    R2 凭据只在调度器持有，trainer 零存储配置：
    - publish：SignNetworkUpload（请求预签名 PUT）→ HTTP 直传 → RegisterNetwork
      （首个网络直接晋级，其后自动创建 gatekeeper 对打）；
    - parent_sha 记录上次成功登记的 sha，作为谱系信息上报。
    调度器地址：SCHEDULER_ENDPOINT（默认 http://127.0.0.1:50051）。
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        import grpc

        from banqi_training.proto import scheduler_pb2, scheduler_pb2_grpc

        self.endpoint = endpoint or os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
        self._pb2 = scheduler_pb2
        # grpc.insecure_channel 只接受 host:port，不接受 URL scheme 前缀
        target = self.endpoint.split("://", 1)[-1]
        self._channel = grpc.insecure_channel(target)
        self._stub = scheduler_pb2_grpc.SchedulerServiceStub(self._channel)
        self.last_sha: Optional[str] = None

    @staticmethod
    def sha256_of(path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def latest_model_path(self) -> Optional[str]:
        """分布式形态下 trainer 不需要从 registry 取模型，返回 None。"""
        return None

    def publish(self, model_path: str) -> None:
        import urllib.request

        sha = self.sha256_of(model_path)
        # 权重格式取文件扩展名（与对象键扩展名同源，worker 据此分派加载器）
        fmt = os.path.splitext(model_path)[1].lstrip(".").lower()
        if not fmt:
            print(f"[registry] ⚠️ 模型路径缺少扩展名，无法判定权重格式: {model_path}")
            return
        with open(model_path, "rb") as f:
            body = f.read()

        # 1. 请求调度器签发预签名 PUT（对象键 networks/<sha>.<format>）
        sign = self._stub.SignNetworkUpload(
            self._pb2.SignNetworkUploadRequest(
                trainer_id=f"trainer-{os.getpid()}",
                sha=sha,
                content_length=len(body),
                content_sha256=sha,
                format=fmt,
            )
        )
        if not sign.accepted:
            print(f"[registry] ⚠️ SignNetworkUpload 被拒绝: {sign.message}")
            return

        # 2. HTTP 直传 R2
        req = urllib.request.Request(sign.upload_url, data=body, method="PUT")
        with urllib.request.urlopen(req, timeout=600) as resp:
            if resp.status != 200:
                raise RuntimeError(f"模型直传失败: HTTP {resp.status} {sign.object_key}")
        print(f"[registry] ✅ 模型已直传: {sign.object_key} <- {model_path}")

        # 3. 登记网络（触发 gatekeeper 对打 / 首个网络晋级）
        ack = self._stub.RegisterNetwork(
            self._pb2.RegisterNetworkRequest(
                sha=sha,
                parent_sha=self.last_sha or "",
                notes=f"trainer publish {os.path.basename(model_path)}",
                format=fmt,
            )
        )
        if not ack.accepted:
            print(f"[registry] ⚠️ RegisterNetwork 被拒绝: {ack.message}")
            return
        self.last_sha = sha
        print(f"[registry] ✅ 已登记网络 sha={sha}: {ack.message} {ack.match_task_hint}".rstrip())
