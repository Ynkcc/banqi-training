"""banqi_training/proto — scheduler gRPC 契约（pb2 由 proto/scheduler.proto 生成）。

生成方式：
    python -m grpc_tools.protoc -I proto \
        --python_out=banqi_training/proto \
        --grpc_python_out=banqi_training/proto \
        proto/scheduler.proto
"""

from . import scheduler_pb2, scheduler_pb2_grpc  # noqa: F401

__all__ = ["scheduler_pb2", "scheduler_pb2_grpc"]
