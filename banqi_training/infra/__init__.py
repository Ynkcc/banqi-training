"""banqi_training/infra — L4 协作接口（EpisodeStore / ModelRegistry）。

仅保留分布式形态实现：R2 凭据只在调度器持有，trainer/worker 经
gRPC 签发的预签名 URL 上下行，零存储配置，仅需 SCHEDULER_ENDPOINT。
"""

from .episode_store import EpisodeStore, SchedulerEpisodeStore
from .model_registry import (
    ModelRegistry,
    SchedulerModelRegistry,
    scheduler_heartbeat,
    scheduler_should_stop,
    scheduler_train_config,
    scheduler_variant,
)

__all__ = [
    "EpisodeStore",
    "SchedulerEpisodeStore",
    "ModelRegistry",
    "SchedulerModelRegistry",
    "scheduler_heartbeat",
    "scheduler_should_stop",
    "scheduler_train_config",
    "scheduler_variant",
]
