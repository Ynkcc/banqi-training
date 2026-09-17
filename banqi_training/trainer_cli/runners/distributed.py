"""banqi_training/trainer_cli/runners/distributed.py — 分布式形态 Trainer（统一训练架构 L5）。

编排：SchedulerEpisodeStore（拉取 worker 直传的 episode 批次）
     + TrainWorker（经队列语义消费训练）
     + RegistryPublisher 线程（新导出 onnx → 预签名上传 + RegisterNetwork 登记）。

本进程不含 Collector，自对弈 worker（banqi-collector --backend scheduler）独立
部署启动。TRAIN_MODE=distributed 时启用。
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Optional

from banqi_training.config import Config, apply_overrides, make_config, snapshot_overridable
from banqi_training.episode_codec import DATA_RESNET, kind_name
from banqi_training.infra import (
    ModelRegistry,
    SchedulerEpisodeStore,
    SchedulerModelRegistry,
    scheduler_should_stop,
    scheduler_train_config,
)
from banqi_training.memory_guard import start_memory_guard
from banqi_training.tb_logger import close_summary_writer, init_summary_writer
from banqi_training.training import TrainWorker
from banqi_training.variant import get_variant

from .context import CountingQueue, log_meta_tb, setup_variant_logging


class RegistryPublisher(threading.Thread):
    """watch Trainer 导出的 onnx 文件，出现或变更即 publish 到 Registry。

    首次观测到的文件同样发布：冷启动时 TrainWorker 已导出初始模型，若不发布
    则调度器永远没有 best 网络，整条闭环无法启动。
    """

    def __init__(self, registry: ModelRegistry, model_path: str, stop: threading.Event,
                 tag: str) -> None:
        super().__init__(name="RegistryPublisher", daemon=True)
        self.registry = registry
        self.model_path = model_path
        self.stop = stop
        self.tag = tag
        self.published = 0

    def run(self) -> None:
        last_mtime: Optional[float] = None
        while not self.stop.is_set():
            try:
                mtime = os.path.getmtime(self.model_path)
                if mtime != last_mtime and os.path.getsize(self.model_path) > 0:
                    self.registry.publish(self.model_path)
                    self.published += 1
                    last_mtime = mtime
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"{self.tag} ⚠️ RegistryPublisher 异常: {exc}")
            self.stop.wait(2.0)


def run_distributed(variant_id: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    config._variant = variant
    tag = f"[{variant.id}][distributed]"
    trainer_id = os.environ.get("TRAINER_ID", "")

    # 调度层训练配置 bootstrap：必须在 TrainWorker 构造之前套用——结构校验、
    # DataBuffer 的值目标校验、lr_decay_batches 都是构造期一次性快照，晚了不生效。
    # 同时留下本地基线，供调度层撤销某项覆盖时回退。
    cfg_baseline = snapshot_overridable(config)
    accepted, train_overrides, message = scheduler_train_config(variant_id, trainer_id)
    if not accepted:
        print(f"{tag} ⚠️ 未取到调度层训练配置：{message}（使用本地配置）")
    elif train_overrides:
        print(f"{tag} ⚙️ 已套用调度层训练配置: {apply_overrides(config, train_overrides)}")
    applied_overrides: dict = dict(train_overrides) if accepted else {}

    log_file = setup_variant_logging(variant)
    print(f"{tag} 📝 运行日志记录至: {log_file}")

    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            log_meta_tb(config, variant_id, tb_log_dir)

    start_memory_guard()

    # 主闭环只消费 ResNet（Gumbel MCTS）数据；NNUE 数据走独立的蒸馏消费方
    store = SchedulerEpisodeStore(variant=variant_id, kind=DATA_RESNET)
    registry = SchedulerModelRegistry()
    counting_q = CountingQueue(store)

    thread_stop = threading.Event()
    # 先构造 TrainWorker（构造期导出冷启动初始模型），再让 publisher 监听它实际
    # 写出的 onnx 路径：启用血量头时是 last_health.onnx，与 config.ONNX_PATH 不同。
    # ckpt_dir 必须跟 config.OUTPUT_DIR 走（variant.checkpoints_dir 对 OUTPUT_DIR 无感）：
    # 否则同变体的两个实验臂会共用同一 ckpt 路径，后启动的臂会 resume 前一臂的权重
    # （A/B 方案 §2.2 标记为「没做这一步，实验无效」的陷阱）。
    train_worker = TrainWorker(
        variant, config, counting_q, thread_stop,
        ckpt_dir=os.path.join(config.OUTPUT_DIR, "checkpoints"),
        reanalysis_submitter=store.submit_reanalysis,
        cfg_baseline=cfg_baseline,
    )
    onnx_path = train_worker.onnx_path()
    sep = "=" * 56
    print(sep)
    print(f"  🚀 分布式 Trainer 启动（变体 {variant_id}，无 Collector）")
    print(
        f"  EPISODE_SOURCE = scheduler ListEpisodes（预签名 GET 拉取，"
        f"类别 {kind_name(DATA_RESNET)}）"
    )
    print(f"  SCHEDULER      = {registry.endpoint}（SignNetworkUpload + RegisterNetwork）")
    print(f"  WATCH ONNX     = {onnx_path}")
    print(sep)

    def _handler(signum, frame):
        if thread_stop.is_set():
            sys.exit(1)
        thread_stop.set()
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    publisher = RegistryPublisher(registry, onnx_path, thread_stop, tag)
    publisher.start()

    train_worker.start()

    start_t = time.time()
    # 绝对强度停机轮询：调度器按「连续 N 次评测无提升」置位 GetInfo.should_stop，
    # 命中即走**既有**优雅停止路径（set 事件 → TrainWorker 训完当前轮并落 checkpoint）。
    stop_poll = int(config.SHOULD_STOP_POLL_SECONDS)
    next_stop_poll = time.time() + stop_poll if stop_poll > 0 else float("inf")
    if stop_poll > 0:
        print(
            f"{tag} 停机 / 训练配置热更轮询：每 {stop_poll}s 一次 "
            f"GetInfo.should_stop + GetTrainConfig"
            f"（SHOULD_STOP_POLL_SECONDS=0 可关闭；轮询失败按「不停止、不改配置」处理）"
        )
    else:
        print(f"{tag} ⚠️ SHOULD_STOP_POLL_SECONDS=0：停机与训练配置热更轮询均已关闭")
    try:
        while not thread_stop.is_set():
            if config.MAX_RUNTIME_SECONDS > 0 and \
                    time.time() - start_t >= config.MAX_RUNTIME_SECONDS:
                print(f"{tag} 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                thread_stop.set()
                break
            if time.time() >= next_stop_poll:
                next_stop_poll = time.time() + stop_poll
                should_stop, reason = scheduler_should_stop(registry.endpoint)
                if should_stop:
                    print(f"{tag} 🏁 调度器下发停机信号，优雅停止：{reason}")
                    thread_stop.set()
                    break
                # 训练配置热更：与停机轮询共用节流周期，只在内容变化时登记；实际生效
                # 由 TrainWorker 在下一个轮边界执行（不打断进行中的训练）。
                accepted, latest, message = scheduler_train_config(variant_id, trainer_id)
                if not accepted:
                    print(f"{tag} ⚠️ 训练配置拉取失败：{message}")
                elif latest != applied_overrides:
                    train_worker.apply_config(latest)
                    applied_overrides = dict(latest)
            if not train_worker.is_alive():
                print(f"{tag} ⚠️ TrainWorker 已退出，停止闭环")
                thread_stop.set()
                break
            thread_stop.wait(2.0)
    finally:
        thread_stop.set()
        train_worker.join(timeout=60)
        if train_worker.is_alive():
            train_worker.join(timeout=10)
        train_worker.finalize()
        publisher.join(timeout=3)
        close_summary_writer()

    tr_stats = train_worker.stats()
    print(f"\n{sep}")
    print(f"  {variant_id} distributed trainer 结束")
    print(f"  累计训练批次: {tr_stats['total_batches']}, 轮次: {tr_stats['round_num']}, "
          f"平均 Loss: {tr_stats['avg_loss']:.4f}")
    print(f"  模型发布次数: {publisher.published}")
    print(sep)
