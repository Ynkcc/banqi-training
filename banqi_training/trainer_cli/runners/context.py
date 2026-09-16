"""banqi/trainer_cli/runners/context.py — 各运行路径共享的基础设施。

包含：可选依赖探测（torch / system_monitor）、队列计数包装、stdout 落盘、
TensorBoard 运行元信息、Rust 变体维度查询缓存。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

# 系统资源监控（psutil + pynvml），缺失依赖时静默禁用
try:
    from banqi_training.system_monitor import SystemMonitor
    HAS_MONITOR = True
except ImportError:  # pragma: no cover
    HAS_MONITOR = False
    SystemMonitor = None  # type: ignore[assignment,misc]

from banqi_training.config import Config
from banqi_training.constants import build_constants
from banqi_training.tb_logger import add_hparams, add_text
from banqi_training.variant import Variant


class CountingQueue:
    """队列消费计数包装（观测自对弈吞吐）。"""

    def __init__(self, q) -> None:
        self._q = q
        self.consumed_games = 0
        self.consumed_samples = 0

    def get(self, *args, **kwargs):
        item = self._q.get(*args, **kwargs)
        if item is not None:
            self.consumed_games += 1
            self.consumed_samples += int(item.get("num_samples", 0))
        return item

    def get_nowait(self, *args, **kwargs):
        item = self._q.get_nowait(*args, **kwargs)
        if item is not None:
            self.consumed_games += 1
            self.consumed_samples += int(item.get("num_samples", 0))
        return item

    def put(self, *args, **kwargs):
        self._q.put(*args, **kwargs)


class TeeQueue:
    """数据旁路分流队列：put 同时转发主队列与旁路队列，get 仅消费主队列。

    主闭环自对弈生产者（rust 线程 / python spawn 子进程）把 episode put 进
    本对象：主队列照常供 TrainWorker 消费，旁路队列供 NnueDistillWorker
    蒸馏消费。旁路队列满时非阻塞丢弃并计数（蒸馏为旁路，不反压主闭环）。
    """

    def __init__(self, main_q, side_q) -> None:
        self.main_q = main_q
        self.side_q = side_q
        self.dropped = 0
        self.forwarded = 0

    def put(self, item, *args, **kwargs):
        try:
            self.side_q.put_nowait(item)
            self.forwarded += 1
        except Exception:
            self.dropped += 1
        return self.main_q.put(item, *args, **kwargs)

    def put_nowait(self, item, *args, **kwargs):
        try:
            self.side_q.put_nowait(item)
            self.forwarded += 1
        except Exception:
            self.dropped += 1
        return self.main_q.put_nowait(item, *args, **kwargs)

    def get(self, *args, **kwargs):
        return self.main_q.get(*args, **kwargs)

    def get_nowait(self, *args, **kwargs):
        return self.main_q.get_nowait(*args, **kwargs)

    def qsize(self):
        return self.main_q.qsize()

    def empty(self):
        return self.main_q.empty()

    def close(self):
        for q in (self.main_q, self.side_q):
            close = getattr(q, "close", None)
            if callable(close):
                close()

    def cancel_join_thread(self):
        for q in (self.main_q, self.side_q):
            fn = getattr(q, "cancel_join_thread", None)
            if callable(fn):
                fn()


class TeeStream:
    """将 stdout/stderr 同时输出到控制台与日志文件。"""

    def __init__(self, original_stream, file_obj) -> None:
        self.original_stream = original_stream
        self.file_obj = file_obj
        self._is_tee = True

    def write(self, data: str) -> None:
        self.original_stream.write(data)
        self.file_obj.write(data)
        self.file_obj.flush()

    def flush(self) -> None:
        self.original_stream.flush()
        self.file_obj.flush()

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()


def setup_variant_logging(variant: Variant) -> str:
    """初始化变体运行日志：在 variant.logs_dir 中创建日志文件并将 print/logging 同步落盘。"""
    import logging
    os.makedirs(variant.logs_dir, exist_ok=True)
    log_file = os.path.join(variant.logs_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    if not getattr(sys.stdout, "_is_tee", False):
        f = open(log_file, "a", encoding="utf-8")
        sys.stdout = TeeStream(sys.stdout, f)
        sys.stderr = TeeStream(sys.stderr, f)

    root_logger = logging.getLogger()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return log_file


def log_meta_tb(config: Config, variant_id: str, tb_log_dir: str) -> None:
    """TensorBoard 运行元信息：HParams + 文本标注（启动时调用一次）。

    add_hparams 在 log_dir/hparams 子目录写入超参数表，跨多次运行可在
    TensorBoard 的 HParams 面板中对比。metric_dict 传空（训练指标另行记录）。
    """
    add_text("meta/variant", variant_id, 0)
    add_text("meta/run_id", os.path.basename(tb_log_dir) if tb_log_dir else "", 0)
    add_text("meta/train_mode", config.TRAIN_MODE or "selfplay", 0)
    add_text("meta/value_target", config.VALUE_TARGET_MODE, 0)
    add_text(
        "meta/device",
        f"train={config.TRAIN_DEVICE} infer={config.INFER_DEVICE} "
        f"cpu_aux={config.INFER_CPU_AUX_WORKERS}",
        0,
    )
    add_text(
        "meta/augment",
        f"enabled={config.DATA_AUGMENT_ENABLED} k={config.DATA_AUGMENT_K} "
        f"transforms={config.DATA_AUGMENT_TRANSFORMS}",
        0,
    )
    hparams = {
        "variant": variant_id,
        "mcts_sims": config.MCTS_SIMS,
        "max_considered_actions": config.MAX_CONSIDERED_ACTIONS,
        "games_per_iter": config.GAMES_PER_ITER,
        "train_batch": config.TRAIN_BATCH,
        "learning_rate": config.LEARNING_RATE,
        "min_lr": config.MIN_LR,
        "lr_decay_steps": config.LR_DECAY_STEPS,
        "lr_decay_rounds": config.LR_DECAY_ROUNDS,
        "batches_per_round": config.batches_per_round(),
        "train_epochs_per_round": config.TRAIN_EPOCHS_PER_ROUND,
        "weight_decay": config.WEIGHT_DECAY,
        "max_buffer": config.MAX_SAMPLE_BUFFER_SIZE,
        "min_samples_to_start": config.MIN_SAMPLES_TO_START,
        "value_target": config.VALUE_TARGET_MODE,
        "data_augment": config.DATA_AUGMENT_ENABLED,
        "data_augment_k": config.DATA_AUGMENT_K,
        "min_new_samples_to_train": config.MIN_NEW_SAMPLES_TO_TRAIN,
        "fast_sample_loss_weight": config.FAST_SAMPLE_LOSS_WEIGHT,
        "reanalysis_enabled": config.REANALYSIS_ENABLED,
        "reanalysis_pool_size": config.REANALYSIS_POOL_SIZE,
        "reanalysis_batch_positions": config.REANALYSIS_BATCH_POSITIONS,
        "eval_match_rounds": config.EVAL_MATCH_ROUNDS,
        "eval_match_games": config.EVAL_MATCH_GAMES,
        "eval_match_opponents": config.EVAL_MATCH_OPPONENTS,
        "eval_match_vs_prev": config.EVAL_MATCH_VS_PREV,
    }
    add_hparams({k: str(v) for k, v in hparams.items()}, {})


def build_const(variant, name: str) -> int:
    """从 Python 侧 build_constants(variant) 取变体维度（variant 唯一声明源）。"""
    return int(getattr(build_constants(variant), name))
