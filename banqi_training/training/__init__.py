"""banqi/training — 训练服务子包。

拆分自原单文件 training_service.py，按职责分层：
  buffer.py  : DataBuffer + episode_to_samples（replay buffer 与数据转换）
  losses.py  : train_step + run_training_epochs（单 batch 训练与整轮调度）
  worker.py  : TrainWorker（队列消费、训练量控制、checkpoint 保存）
training_service.py 保留为向后兼容的 re-export 入口。
"""

from .buffer import DataBuffer, episode_to_samples
from .worker import TrainWorker
from .losses import train_step, run_training_epochs

__all__ = [
    "DataBuffer",
    "episode_to_samples",
    "TrainWorker",
    "train_step",
    "run_training_epochs",
]
