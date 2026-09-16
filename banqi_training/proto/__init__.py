"""banqi_training/proto — scheduler gRPC 契约（pb2 由 proto/scheduler.proto 生成）。

唯一契约源是 banqi-scheduler/proto/scheduler.proto（本仓库的 proto/ 是同步副本）。

生成方式：proto 的「相对 include 根」的路径必须与输出包路径一致
（即 banqi_training/proto/scheduler.proto），否则 grpc 插件会生成
`import scheduler_pb2` 这种无法导入的绝对导入。故先用临时 include 根：

    REPO=/path/to/banqi-training
    T=$(mktemp -d); mkdir -p "$T/banqi_training/proto"
    cp "$REPO/proto/scheduler.proto" "$T/banqi_training/proto/"
    (cd "$T" && python -m grpc_tools.protoc -I . \\
        --python_out="$REPO" --grpc_python_out="$REPO" \\
        banqi_training/proto/scheduler.proto)
    rm -rf "$T"
"""

from . import scheduler_pb2, scheduler_pb2_grpc  # noqa: F401

__all__ = ["scheduler_pb2", "scheduler_pb2_grpc"]
