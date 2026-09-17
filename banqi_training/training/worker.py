"""banqi/training/worker.py — 训练 worker 线程。

TrainWorker 在独立线程消费 self_play 队列，把 episode 转换的 sample 写入
DataBuffer，按 (new_samples/batch)×epochs 限制训练量（避免旧数据反复训练），
并在 checkpoint 时保存 model/optimizer/scheduler/global_step + 训练监控。
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from banqi_training.checkpoint import export_model_isolated
from banqi_training.config import Config, apply_overrides, make_config
from banqi_training.tb_logger import add_scalar
from banqi_training.variant import Variant, get_variant

from .augment import EpisodeAugmenter
from .buffer import DataBuffer, episode_to_samples
from .losses import run_training_epochs, _resolve_device
from .lr_schedule import is_stopped, make_cosine_clamp_scheduler, resolve_lr_decay_batches
from banqi_training.reanalysis import PositionPool, encode_payload
from .eval import (
    build_fixed_eval,
    eval_policy_accuracy,
    eval_value_drift,
    prefill_from_archive,
    select_balanced_fixed_samples,
)


class TrainWorker(threading.Thread):
    def __init__(
        self,
        variant: Variant,
        cfg: Config,
        data_queue,
        stop_event,
        ckpt_dir: Optional[str] = None,
        device=None,
        run_dir: Optional[str] = None,
        ckpt_events: Optional[List[threading.Event]] = None,
        reanalysis_submitter: Optional[Callable[[str, int, bytes, int], Tuple[bool, str]]] = None,
        cfg_baseline: Optional[Dict[str, Any]] = None,
    ):
        """标准签名：(variant, cfg, data_queue, stop_event, ...)。

        cfg 必须是 make_config 构造的完整 Config（不设源码兜底）。旧式
        (data_q, stop_event, variant) 调用顺序请使用 from_legacy 类方法。
        ckpt_events: 可选的旁路事件列表（NnueDistillWorker / ExpectimaxSidecar），
        每次 checkpoint 实际落盘后逐个 set。
        reanalysis_submitter: 局面重搜提交回调 `(variant, mcts_sims, payload, positions)
        -> (accepted, message)`；REANALYSIS_ENABLED=true 时必须提供（缺失即报错，
        不静默跳过整个特性）。
        cfg_baseline: 可远程调节字段的本地基线（config.snapshot_overridable 采集）。
        调度层撤销某项覆盖时靠它回退本地值；本地形态不传，则覆盖只增不减。
        """
        super().__init__(name=f"TrainWorker-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = cfg
        self._cfg_baseline = cfg_baseline
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.ckpt_dir = ckpt_dir or variant.checkpoints_dir
        self.run_dir = run_dir
        self.ckpt_events = ckpt_events or []
        self.device = device or _resolve_device(cfg.TRAIN_DEVICE)
        # 血量差异价值头开关：开启时使用独立 _health 模型文件，与标准模型物理隔离。
        self.health_enabled = bool(cfg.HEALTH_VALUE_HEAD_ENABLED)
        # 分布化价值头开关：开启时 value 头输出改为 K 桶分布（损失换成 HL-Gauss 交叉熵），
        # 同样使用独立 _vdist 模型文件，避免与标准/血量头臂互相 resume。
        self.value_dist_enabled = bool(cfg.VALUE_DIST_ENABLED)
        self.value_dist_bins = int(cfg.VALUE_DIST_BINS)
        # 策略分支解耦（B3 方案 1）：开启时策略分支走独立 trunk
        self.policy_trunk_independent = bool(cfg.POLICY_TRUNK_INDEPENDENT)
        # 空间对称增强（纯 Python，动作置换表带缓存）
        self.augmenter = EpisodeAugmenter(variant, cfg)
        # 局面重搜位置池（reanalysis）：位置来自 episode 的快照侧信道（collector 采集）。
        # 启用但没给提交回调属于配置错误（跑起来才发现提交不了最费时间），直接失败。
        self.reanalysis_submitter = reanalysis_submitter
        self.reanalysis_pool: Optional[PositionPool] = None
        if cfg.REANALYSIS_ENABLED:
            if reanalysis_submitter is None:
                raise ValueError(
                    "[TR] REANALYSIS_ENABLED=true 但未提供 reanalysis_submitter："
                    "该特性需要连调度器的分布式 runner，请改用分布式形态或关闭 REANALYSIS_ENABLED"
                )
            self.reanalysis_pool = PositionPool(int(cfg.REANALYSIS_POOL_SIZE))
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 监控：每轮训练时长、最近 ckpt 路径、最近一次 epoch loss 分解
        self.metrics = {
            "train_duration": 0.0,
            "last_ckpt_path": None,
            "last_epoch_losses": None,
            "last_lr": 0.0,
            "global_step": 0,
        }

        # S3 默认 CPU（避免与主训练 GPU 争抢）；GPU 推理时显式 device
        self.desired_sp_device = "cuda" if cfg.SELF_PLAY_DEVICE.startswith("cuda") else "cpu"

        self._last_ckpt_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.round_num = 0
        self.total_loss_sum = 0.0
        self.total_policy_loss_sum = 0.0
        self.total_value_loss_sum = 0.0
        self.total_health_loss_sum = 0.0
        self.round_history: deque = deque(maxlen=1000)

        self._warmup_done = False
        self._raw_sample_pool: List[Dict] = []
        self._fixed_eval: Optional[Dict] = None
        # 调度层下发的运行时配置覆盖（字段名 → 值字符串）：apply_config 由 RPC 线程
        # 写入、_apply_pending_config 由训练线程在轮边界消费，故单独加锁保护。
        self._pending_overrides: Optional[Dict[str, str]] = None
        self._pending_overrides_lock = threading.Lock()
        self._init_model_and_checkpoint()

    @classmethod
    def from_legacy(cls, data_queue, stop_event, variant_or_id):
        """旧式调用顺序兼容入口：(data_queue, stop_event, variant)。

        variant_or_id 可以是 Variant 对象或 variant_id 字符串。内部用标准签名
        构造，避免在 __init__ 内做类型分派破坏静态类型检查。
        """
        v = variant_or_id if isinstance(variant_or_id, Variant) else get_variant(str(variant_or_id))
        return cls(v, make_config(v.id), data_queue, stop_event)

    def apply_config(self, overrides: Dict[str, str]) -> None:
        """登记调度层下发的训练配置覆盖（由 RPC 线程调用，立即返回）。

        只登记不落地：真正写入 cfg 与重算派生量由训练线程在轮边界执行
        （见 _apply_pending_config），避免与 scheduler.step / optimizer.step 争用。
        连发多次只有最后一次生效——配置是全量覆盖语义，不存在次序问题。
        """
        with self._pending_overrides_lock:
            self._pending_overrides = dict(overrides)

    def _apply_pending_config(self) -> bool:
        """在训练线程的轮边界应用待生效覆盖；返回是否发生了变更。

        「构造期快照」必须在这里重算，否则改了 cfg 也不生效：
        - lr_decay_batches + scheduler：LR 计划变了要重建余弦调度器并续接进度；
        - optimizer 的 weight_decay / initial_lr：AdamW 构造期固定，需逐 param_group 改；
        - anneal_rounds / ema_decay：_anneal_value_weight 与 EMA 更新按实例属性读；
        - buffer / augmenter 持同一份 cfg 引用，改 cfg 即刻生效，无需额外处理。

        空覆盖（{}）不是「无操作」：它表示调度层已清空全部覆盖，此时按基线回落本地值。
        """
        with self._pending_overrides_lock:
            overrides = self._pending_overrides
            self._pending_overrides = None
        if overrides is None:
            return False
        applied = apply_overrides(self.cfg, overrides, self._cfg_baseline)
        changed = set(applied)
        if not changed:
            return False
        if changed & {"LEARNING_RATE", "MIN_LR", "LR_DECAY_STEPS", "LR_DECAY_ROUNDS"}:
            self._rebuild_lr_schedule()
        if "WEIGHT_DECAY" in changed:
            for group in self.optimizer.param_groups:
                group["weight_decay"] = float(self.cfg.WEIGHT_DECAY)
        if "EMA_DECAY" in changed:
            self.ema_decay = float(self.cfg.EMA_DECAY)
        if "VALUE_TARGET_ANNEAL_ROUNDS" in changed:
            self.anneal_rounds = self.cfg.VALUE_TARGET_ANNEAL_ROUNDS
        print(f"[TR-{self.variant.id}] 应用调度层训练配置 {applied}: "
              f"LR={self.cfg.LEARNING_RATE:g}/MIN_LR={self.cfg.MIN_LR:g}, "
              f"TRAIN_BATCH={self.cfg.TRAIN_BATCH}, "
              f"TRAIN_EPOCHS_PER_ROUND={self.cfg.TRAIN_EPOCHS_PER_ROUND}, "
              f"价值目标={self.cfg.VALUE_TARGET_MODE}")
        return True

    def _rebuild_lr_schedule(self) -> None:
        """按当前 cfg 重建余弦调度器并续接进度（LR 计划字段热更后调用）。

        LambdaLR 的 base_lr 取自 param_group['initial_lr']，LEARNING_RATE 变更时必须
        同步刷新该值，否则新计划仍以旧 LR 为基准。
        """
        self.lr_decay_batches = resolve_lr_decay_batches(self.cfg)
        progress = self.scheduler.last_epoch
        for group in self.optimizer.param_groups:
            group["initial_lr"] = float(self.cfg.LEARNING_RATE)
        self.scheduler = make_cosine_clamp_scheduler(
            self.optimizer, self.cfg, self.lr_decay_batches
        )
        # 显式步进到原进度：让新计划的 LR 立刻生效，而不是等下次 step() 才切
        self.scheduler.step(progress)
        print(f"[TR-{self.variant.id}] LR 计划已重建: t_max={self.lr_decay_batches} batch, "
              f"进度={progress} step, 当前 LR={self.optimizer.param_groups[0]['lr']:.3g}")

    def _new_model(self):
        """按当前开关构造同结构的 BanqiNet（训练模型 / EMA 影子模型共用）。

        结构开关（血量头 / 分布化价值头）必须与 checkpoint、导出子进程完全一致，
        否则 load_state_dict 会因参数形状不符直接失败。
        """
        from banqi_training.nn_model import BanqiNet

        return BanqiNet(
            self.variant,
            enable_health=self.health_enabled,
            enable_value_dist=self.value_dist_enabled,
            value_dist_bins=self.value_dist_bins,
            independent_policy_trunk=self.policy_trunk_independent,
        )

    def _init_model_and_checkpoint(self):
        cfg = self.cfg

        ema_enabled = cfg.EMA_ENABLED
        ema_decay = float(cfg.EMA_DECAY)
        self.ema_enabled = ema_enabled
        self.ema_decay = ema_decay
        self.ema_model = None
        # LR 余弦的时间跨度（batch）：LR_DECAY_ROUNDS>0 时按每轮训练量折算，否则用
        # LR_DECAY_STEPS。两处 scheduler 构造共用同一值，resume 时由 scheduler_state 续接。
        self.lr_decay_batches = resolve_lr_decay_batches(cfg)

        if os.path.exists(self.last_ckpt_path()):  # resume
            print(f"[TR-{self.variant.id}] 从 checkpoint 恢复: {self.last_ckpt_path()}")
            ckpt = torch.load(self.last_ckpt_path(), map_location=self.device, weights_only=False)
            model = self._new_model()
            model.load_state_dict(ckpt["model_state"])
            self.model = model.to(self.device)
            if ema_enabled:
                self.ema_model = self._new_model().to(self.device)
                if "ema_model_state" in ckpt and ckpt["ema_model_state"] is not None:
                    self.ema_model.load_state_dict(ckpt["ema_model_state"])
                else:
                    self.ema_model.load_state_dict(self.model.state_dict())
            # 这里的 lr 只是构造期占位：紧随其后的 optimizer_state / scheduler_state
            # 会把实际 LR 与余弦进度（base_lrs + last_epoch）恢复到中断时的状态。
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY,
            )
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler = make_cosine_clamp_scheduler(
                self.optimizer, cfg, self.lr_decay_batches
            )
            if "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.global_step = ckpt.get("global_step", 0)
            self.metrics["global_step"] = self.global_step
            self.start_global_step = self.global_step
            self.start_total_samples = ckpt.get("total_samples", 0)
            self.version = ckpt.get("version", 0) + 1
            print(f"[TR-{self.variant.id}] 恢复 global_step={self.global_step}, "
                  f"version={self.version}" + (" (EMA 已启用)" if ema_enabled else ""))
        else:
            self.model = self._new_model().to(self.device)
            if ema_enabled:
                self.ema_model = self._new_model().to(self.device)
                self.ema_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY
            )
            self.scheduler = make_cosine_clamp_scheduler(
                self.optimizer, cfg, self.lr_decay_batches
            )
            self.global_step = 0
            self.start_global_step = 0
            self.start_total_samples = 0
            self.version = 0

            # 冷启动：立即导出初始模型，供 Rust 自对弈加载，避免自对弈等待
            # last.pt、训练 worker 又等待自对弈数据，两者互相等待而死锁。
            self._export_initial_model()

        buffer_capacity = cfg.MAX_SAMPLE_BUFFER_SIZE
        self.buffer = DataBuffer(buffer_capacity, self.variant, cfg)

        # 冷存储预填充 & 固定验证集生成
        fixed_archive = prefill_from_archive(self.buffer, self.variant, cfg)
        if fixed_archive is not None:
            self._fixed_eval = fixed_archive

        # 显式记录 value 目标模式（终端日志，便于复现/调试）
        print(f"[TR-{self.variant.id}] 价值目标模式={cfg.VALUE_TARGET_MODE}，"
              f"buffer 容量={buffer_capacity}，TRAIN_DEVICE={self.device}")

        # LR 计划可读化：把 batch 跨度换算成「约多少训练轮」，替代用 LR_DECAY_STEPS 猜
        per_round = cfg.batches_per_round()
        source = (f"LR_DECAY_ROUNDS={cfg.LR_DECAY_ROUNDS}" if int(cfg.LR_DECAY_ROUNDS) > 0
                  else f"LR_DECAY_STEPS={cfg.LR_DECAY_STEPS}")
        print(f"[TR-{self.variant.id}] LR 计划: {self.lr_decay_batches} batch"
              f"（每轮 ~{per_round} batch → 约 {self.lr_decay_batches / per_round:.0f} 轮退到"
              f" MIN_LR={cfg.MIN_LR:g}），来源 {source}")

        # ---- value 目标退火（VALUE_TARGET_MODE='anneal' 时）----
        # 退火权重 w：前 N 轮用 mcts 平滑评估，后段切到 game_result 真值。
        # 每轮训练前按 (round_idx / anneal_rounds) 更新 buffer.value_result_weight。
        self.anneal_rounds = cfg.VALUE_TARGET_ANNEAL_ROUNDS

        init_ckpt = cfg.INIT_FROM_CHECKPOINT
        if init_ckpt:
            self._load_pretrained(init_ckpt)

    def _load_pretrained(self, ckpt_path: str):
        """从指定 checkpoint 导入权重（仅 model + optimizer），重置 global_step。"""
        print(f"[TR-{self.variant.id}] 加载预训练权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            print(f"[TR-{self.variant.id}] ⚠️ state_dict 不完全匹配（跨变体/结构），"
                  f"忽略不匹配键: {e}")
            sd = ckpt["model_state"]
            own = self.model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in own and v.shape == own[k].shape}
            own.update(filtered)
            self.model.load_state_dict(own)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = 0
        self.metrics["global_step"] = 0

    def _model_stem(self) -> str:
        """模型文件主干名：结构开关各自加后缀，实现 A/B 两臂的物理隔离。

        隔离是必需的而非美化：两臂共用同一 ckpt 路径时，后启动的一臂会 resume
        前一臂的权重（且因参数形状不同直接报错），实验结论随之失效。
        """
        stem = "last"
        if self.health_enabled:
            stem += "_health"
        if self.value_dist_enabled:
            stem += "_vdist"
        return stem

    def _ckpt_path(self) -> str:
        """checkpoint 路径（worker 内部，用于断点续训）。"""
        return os.path.join(self.ckpt_dir, f"{self._model_stem()}.ckpt")

    def _has_health_path_override(self) -> bool:
        """是否沿用配置里的血量头专属导出路径（仅纯血量头臂，保持既有行为）。"""
        return self.health_enabled and not self.value_dist_enabled

    def _pt_path(self) -> str:
        """TorchScript 模型导出路径（供自对弈推理）。"""
        if self._has_health_path_override() and self.cfg.HEALTH_MODEL_PATH:
            return self.cfg.HEALTH_MODEL_PATH
        return os.path.join(self.ckpt_dir, f"{self._model_stem()}.pt")

    def onnx_path(self) -> str:
        """ONNX 模型导出路径（供调度器 worker / gatekeeper 下载使用）。

        公开方法：分布式 runner 的 RegistryPublisher 必须监听与训练侧实际写出
        完全一致的路径（启用血量头时是 last_health.onnx）。
        """
        if self._has_health_path_override() and self.cfg.HEALTH_ONNX_PATH:
            return self.cfg.HEALTH_ONNX_PATH
        return os.path.join(self.ckpt_dir, f"{self._model_stem()}.onnx")

    def last_ckpt_path(self):
        return self._ckpt_path()

    def _export_initial_model(self) -> None:
        """冷启动时导出初始模型（.pt/.onnx），供 Rust 自对弈加载。

        全新起训时 last.pt 不存在，若不自对弈先导出，Rust 自对弈会一直等待
        last.pt，而训练 worker 又因拿不到自对弈数据永不导出，形成死锁。
        这里在模型初始化后立即导出一次初始权重，打破该循环依赖。
        """
        pt_path = self._pt_path()
        onnx_path = self.onnx_path()
        if os.path.exists(pt_path):
            return
        try:
            self.model.eval()
            export_model_isolated(self.model, pt_path, onnx_path, self.variant, self.device)
            print(f"[TR-{self.variant.id}] 💾 冷启动：已导出初始模型 {pt_path}")
        except Exception as exc:  # noqa: BLE001 - 初始导出失败不阻塞启动，训练仍可推进
            print(f"[TR-{self.variant.id}] ⚠️ 冷启动初始模型导出失败: {exc}")

    def get_inference_model(self):
        if self.ema_enabled and self.ema_model is not None:
            self.ema_model.eval()
            return self.ema_model
        return self.model

    def get_global_step(self):
        return self.global_step

    def get_model_version(self):
        return self.version

    def get_checkpoint_path(self):
        return self.last_ckpt_path()

    def save_checkpoint(
        self,
        new_samples: int = 0,
        total_samples: Optional[int] = None,
        round_idx: int = 0,
        force: bool = False,
    ) -> None:
        save_every = max(int(self.cfg.CKPT_SAVE_EVERY), 1)
        export_every = max(int(self.cfg.CKPT_EXPORT_EVERY), 1)

        pt_path = self._pt_path()
        onnx_path = self.onnx_path()

        should_save_ckpt = force or (round_idx % save_every == 0) or (round_idx == 0)
        # round_idx==0 时仅在 pt 不存在时导出一次；round_idx>0 才按周期导出，
        # 避免冷启动/rule_selfplay 早期 round 固定时每个训练步都重导 TorchScript/ONNX 拖慢吞吐。
        should_export = force or (round_idx > 0 and round_idx % export_every == 0) \
            or not os.path.exists(pt_path)

        if not should_save_ckpt and not should_export:
            return

        path = self.last_ckpt_path()
        if should_save_ckpt:
            self.model.eval()
            if self.ema_model is not None:
                self.ema_model.eval()
            snapshot = {
                "model_state": self.model.state_dict(),
                "ema_model_state": self.ema_model.state_dict() if self.ema_model is not None else None,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "total_samples": total_samples if total_samples is not None else self.start_total_samples,
                "version": self.version,
                "pytorch_version": torch.__version__,
            }
            torch.save(snapshot, path)

        if should_export:
            export_target = self.get_inference_model()
            export_model_isolated(export_target, pt_path, onnx_path, self.variant, self.device)

        with self._last_ckpt_lock:
            self.metrics["last_ckpt_path"] = path
            self.metrics["last_global_step"] = self.global_step
        print(f"[TR-{self.variant.id}] 💾 checkpoint 已保存/更新: {path} + {pt_path} "
              f"(global_step={self.global_step}, v{self.version}, force={force})")
        # 通知旁路 worker（NNUE 蒸馏 / expectimax sidecar）checkpoint 已更新
        for ev in self.ckpt_events:
            ev.set()


    def _anneal_value_weight(self, round_idx: int):
        if self.anneal_rounds and self.buffer.cfg.VALUE_TARGET_MODE == "anneal":
            w = min(1.0, (round_idx + 1) / max(self.anneal_rounds, 1))
            self.buffer.value_result_weight = w
            print(f"[TR-{self.variant.id}] value 目标退火 w={w:.3f} (round {round_idx})")

    def _ensure_fixed_eval_from_selfplay(self, samples: List[Dict]) -> None:
        """无归档时，从自对弈原始样本池构建固定验证集。

        池子**跨轮累积**：固定验证集是「价值/策略头随训练变化」的仪表，构建失败
        （样本不足，或终局结果类别退化）不应把已攒的样本丢掉 —— 旧实现无论如何都清空
        池子，于是每轮都用同一小批样本重建，退化时永远修不好（历史事故：验证集整批
        同一类别 → corr(终局)/胜负区分度 恒为 0，268 轮无人发现）。
        构建成功后清空池子：仪表只需建一次，之后局面固定才可比。
        """
        if self._fixed_eval is not None:
            return
        n_fixed = self.cfg.VALUE_DRIFT_NUM_POSITIONS
        if n_fixed <= 0:
            return
        self._raw_sample_pool.extend(samples)
        if len(self._raw_sample_pool) < n_fixed:
            return
        # 有界池：上限放宽到 n_fixed*8（跨轮攒够两类终局样本），同时防止内存无界增长
        max_pool = max(n_fixed * 8, 4096)
        if len(self._raw_sample_pool) > max_pool:
            self._raw_sample_pool = self._raw_sample_pool[-max_pool:]
        pool = select_balanced_fixed_samples(self._raw_sample_pool, n_fixed)
        fixed = build_fixed_eval(pool, self.variant, source="自对弈池") if pool else None
        if fixed is None:
            # 退化 / 构建失败：保留池子继续攒，下轮重试（build_fixed_eval 已打印原因）
            return
        self._raw_sample_pool = []
        self._fixed_eval = fixed

    def _maybe_augment(self, episode_dict: Dict) -> List[Dict]:
        """按 config 对 episode 做空间对称增强（见 training/augment.py）。"""
        return self.augmenter.augment(episode_dict)

    def _safe_qsize(self) -> int:
        """线程安全地读取数据队列积压。"""
        try:
            if hasattr(self.data_queue, "qsize"):
                qsize = self.data_queue.qsize()
                return int(qsize) if qsize is not None else -1
            return -1
        except Exception:
            return -1

    def run(self, rounds: int = 100000):
        cfg = self.cfg
        version = self.version
        total_samples = self.start_total_samples
        # 批量训练：自对弈数据逐局到达，单局样本量远小于一个合理训练批次。
        # 若每局立即训练，max_batches 会按单局样本量被压到极小，训练碎片化且
        # 反复抽到旧数据。这里累积到足够新样本量才训练一次，让训练量充足且聚焦新数据
        # （selfplay 与 rule_selfplay 统一该逻辑）。阈值默认取 buffer 容量的 1/4，
        # 可用 MIN_NEW_SAMPLES_TO_TRAIN 显式指定；它同时决定每轮 batch 数，LR 计划若用
        # LR_DECAY_ROUNDS 表达则自动按该值折算（见 lr_schedule.resolve_lr_decay_batches）。
        batch_train_min_samples = cfg.min_new_samples_to_train()
        print(f"[TR-{self.variant.id}] 训练节流阈值={batch_train_min_samples} 新样本/轮"
              f"（MIN_NEW_SAMPLES_TO_TRAIN={cfg.MIN_NEW_SAMPLES_TO_TRAIN}，0=自动）"
              f" → 约 {cfg.batches_per_round()} batch/轮")
        pending_samples = 0   # 累积待训练的新样本数
        # 训练轮次由 trainer 自己维护并单调递增：分布式形态下 episode 不携带轮次，
        # 轮次恒为 0 会让 save_checkpoint 的 should_export 恒为假（训练期永不导出
        # onnx），闭环因此拿不到新网络。
        round_idx = 0
        while not is_stopped(self.stop_event):
            try:
                episode_dict = self.data_queue.get(timeout=2.0)
            except Exception:
                continue
            if episode_dict is None:
                break

            # 调度层热更配置：在轮边界消费（此刻不处于训练中，无需与 scheduler /
            # optimizer 争锁），LR 计划 / optimizer / EMA / 退火 / 节流阈值的重算
            # 全部在本线程内完成
            if self._apply_pending_config():
                batch_train_min_samples = cfg.min_new_samples_to_train()

            t0 = time.time()
            # 局面重搜：只收**原始** episode 的快照 —— 增强副本的棋盘已做对称变换，
            # 其快照与实际局面不符，收进去会用错局面重搜。
            if self.reanalysis_pool is not None:
                self.reanalysis_pool.add_episode(episode_dict)
            # 空间对称增强（关闭时原样返回）
            episode_dicts = self._maybe_augment(episode_dict)
            samples: List[Dict] = []
            for ed in episode_dicts:
                samples.extend(episode_to_samples(ed))
            self._ensure_fixed_eval_from_selfplay(samples)
            self.buffer.add_samples(samples)
            new_samples = len(samples)
            total_samples += new_samples
            pending_samples += new_samples

            min_samples = cfg.MIN_SAMPLES_TO_START
            if len(self.buffer) < min_samples:
                print(f"[TR-{self.variant.id}] 等待足够样本进行训练: "
                      f"{len(self.buffer)}/{min_samples}")
                self._maybe_save_early(round_idx)
                continue

            # ---- 批量训练门控：累积够新样本才训练，避免单局碎片化训练 ----
            if pending_samples < batch_train_min_samples:
                continue

            # 本次训练消化 pending_samples 这一批新样本；selfplay 与 rule_selfplay 统一
            new_samples = pending_samples
            pending_samples = 0
            round_idx += 1
            self._anneal_value_weight(round_idx)

            # ---- 训练量限制：与累积新增样本量匹配，避免旧数据反复训练 ----
            # 关键：max_batches 必须由本次累积的新样本量决定，而不是误触发全量训练。
            # 旧逻辑 `if new_samples >= capacity_base//4: max_batches=None` 在批量累积
            # 后（new_samples=pending≈capacity_base//4）会误判为"新增量足够大"而全量
            # 训练整个 buffer（含大量旧数据），导致旧数据反复过拟合、新数据占比被稀释。
            # 这里始终按新增样本量成比例设定训练量，聚焦消化新数据。
            max_batches = int(
                (new_samples / cfg.TRAIN_BATCH) * cfg.TRAIN_EPOCHS_PER_ROUND + 0.5
            )
            max_batches = max(max_batches, 1)

            self.model.train()
            epoch_results, total_batches = run_training_epochs(
                self.model, self.optimizer, self.scheduler, self.buffer,
                cfg.TRAIN_EPOCHS_PER_ROUND, self.device, max_batches=max_batches,
                ema_model=self.ema_model if self.ema_enabled else None,
                ema_decay=self.ema_decay,
                health_enabled=self.health_enabled,
                health_loss_weight=cfg.HEALTH_LOSS_WEIGHT,
                health_gauss_sigma=cfg.HEALTH_GAUSS_SIGMA,
                value_dist_enabled=self.value_dist_enabled,
                value_gauss_sigma=cfg.VALUE_GAUSS_SIGMA,
                fast_sample_weight=cfg.FAST_SAMPLE_LOSS_WEIGHT,
            )
            self.model.eval()

            self.global_step += total_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics["train_duration"] = time.time() - t0
            self.metrics["global_step"] = self.global_step
            self.metrics["last_lr"] = current_lr
            last_losses = epoch_results[-1] if epoch_results else None
            self.metrics["last_epoch_losses"] = last_losses
            if last_losses is not None:
                with self._stats_lock:
                    self.round_num = round_idx
                    self.total_loss_sum += last_losses[0] * total_batches
                    self.total_policy_loss_sum += last_losses[1] * total_batches
                    self.total_value_loss_sum += last_losses[2] * total_batches
                    self.total_health_loss_sum += last_losses[3] * total_batches
                    self.round_history.append({
                        "round": round_idx,
                        "train_loss": last_losses[0],
                        "policy_loss": last_losses[1],
                        "value_loss": last_losses[2],
                        "health_loss": last_losses[3],
                        "grad_norm": last_losses[4],
                        "entropy": last_losses[5],
                        "lr": current_lr,
                        "global_step": self.global_step,
                    })

                print(f"[TR-{self.variant.id}] round {round_idx}: "
                      f"epoch_avg_loss={last_losses[0]:.4f} "
                      f"(policy={last_losses[1]:.4f}, value={last_losses[2]:.4f}"
                      f"{', health=' + format(last_losses[3], '.4f') if self.health_enabled else ''}) "
                      f"grad_norm={last_losses[4]:.3f} entropy={last_losses[5]:.3f} "
                      f"value_mean={last_losses[6]:.3f} value_std={last_losses[7]:.3f} "
                      f"lr={current_lr:.2e} duration={self.metrics['train_duration']:.1f}s "
                      f"buffer={len(self.buffer)} global_step={self.global_step}")

                # 恢复 TensorBoard 训练过程标量记录
                step = self.global_step
                tag = f"[TR-{self.variant.id}]"
                add_scalar("train/loss", last_losses[0], step)
                add_scalar("train/policy_loss", last_losses[1], step)
                add_scalar("train/value_loss", last_losses[2], step)
                if self.health_enabled:
                    add_scalar("train/health_loss", last_losses[3], step)
                add_scalar("train/grad_norm", last_losses[4], step)
                add_scalar("train/policy_entropy", last_losses[5], step)
                add_scalar("train/value_mean", last_losses[6], step)
                add_scalar("train/value_std", last_losses[7], step)
                add_scalar("train/lr", current_lr, step)
                add_scalar("train/buffer_size", len(self.buffer), step)
                add_scalar("queue/backlog", self._safe_qsize(), step)
                # B6 观测：平局样本占比与平局样本子力差幅度（决定目标改造的收益上限）
                sig = self.buffer.take_signal_stats()
                if sig["n_samples"] > 0:
                    add_scalar("data/draw_sample_ratio", sig["draw_ratio"], step)
                    add_scalar("data/draw_hp_abs_mean", sig["draw_hp_abs_mean"], step)
                    add_scalar("data/draw_hp_nonzero_ratio", sig["draw_hp_nonzero_ratio"], step)
                    if cfg.VALUE_TARGET_MODE == "game_hp" or round_idx % 10 == 0:
                        print(
                            f"[TR-{self.variant.id}] 📈 样本信号: 平局占比={sig['draw_ratio']:.3f} "
                            f"平局|子力差|均值={sig['draw_hp_abs_mean']:.3f} "
                            f"非零占比={sig['draw_hp_nonzero_ratio']:.3f} "
                            f"（{int(sig['n_samples'])} 样本）"
                        )
                if cfg.VALUE_TARGET_MODE == "anneal":
                    add_scalar("train/value_anneal_w", self.buffer.value_result_weight, step)
                elif cfg.VALUE_TARGET_MODE == "mixed":
                    add_scalar("train/value_game_weight", cfg.VALUE_MIX_GAME_WEIGHT, step)

                eval_value_drift(self.model, self.device, self._fixed_eval, step, tag, round_idx)
                eval_policy_accuracy(self.model, self.device, self._fixed_eval, step, tag, round_idx)
                # 对战评估已移交调度器 gatekeeper rating（candidate vs best + GSPRT）

            self.save_checkpoint(new_samples=new_samples, total_samples=total_samples,
                                 round_idx=round_idx)
            self._maybe_submit_reanalysis(round_idx)
            version += 1
            self.version = version
            # 周期内存维护：强制 GC + glibc arena 归还（防 RSS 线性增长）
            self._maintain_memory()

            if round_idx >= rounds - 1:
                print(f"[TR-{self.variant.id}] 达到训练轮数上限 {rounds}，退出训练 worker")
                break


    def _maybe_save_early(self, round_idx):
        # 预热阶段（样本不足）也定期保存，避免长期无 checkpoint
        if round_idx % 10 == 0 and not os.path.exists(self.last_ckpt_path()):
            self.save_checkpoint(round_idx=round_idx)

    def _maybe_submit_reanalysis(self, round_idx: int) -> None:
        """按周期把位置池里的历史局面打包提交给调度器（提交成功才移出池子）。

        位置池是「先攒后交」：不足一批就继续攒（避免提交过小载荷）；提交被拒
        （调度器未启用 / 队列满）或 RPC 失败时保留在池中，下一轮重试，不丢数据。
        """
        pool = self.reanalysis_pool
        if pool is None:
            return
        every = max(int(self.cfg.REANALYSIS_SUBMIT_EVERY_N_ROUNDS), 1)
        if round_idx % every != 0:
            return
        stats = pool.take_stats()
        batch = max(int(self.cfg.REANALYSIS_BATCH_POSITIONS), 1)
        skipped = (
            f"；本轮跳过：无快照 {stats.skipped_no_positions} 局（collector 是否开了 collect_positions？）"
            f" / 无胜负 {stats.skipped_no_winner} 局"
            if (stats.skipped_no_positions or stats.skipped_no_winner)
            else ""
        )
        items = pool.peek(batch)
        if len(items) < batch:
            print(f"[TR-{self.variant.id}] 🔁 重搜位置池 {len(pool)}/{pool.capacity}"
                  f"（不足一批 {batch}，继续攒）{skipped}")
            return

        payload = encode_payload(items)
        try:
            accepted, message = self.reanalysis_submitter(
                self.variant.id, int(self.cfg.REANALYSIS_MCTS_SIMS), payload, len(items)
            )
        except Exception as exc:  # noqa: BLE001 - 网络抖动不应终止训练：位置留在池中下轮重试
            print(f"[TR-{self.variant.id}] ⚠️ 重搜提交异常（{exc}），{len(items)} 个位置保留待重试")
            return
        if accepted:
            pool.drop_front(len(items))
            print(f"[TR-{self.variant.id}] 🔁 已提交 {len(items)} 个局面重搜"
                  f"（池内剩余 {len(pool)}）{skipped}")
        else:
            print(f"[TR-{self.variant.id}] ⚠️ 重搜提交被拒（{message}），"
                  f"{len(items)} 个位置保留待重试{skipped}")

    def _maintain_memory(self, force: bool = False) -> None:
        """周期内存维护：手动 GC + 堆内存空闲页释放。"""
        import gc as _gc
        import ctypes
        self._round_mem_count = getattr(self, "_round_mem_count", 0) + 1
        _gc.collect()
        if (force or self._round_mem_count % 50 == 0) and hasattr(ctypes.CDLL("libc.so.6"), "malloc_trim"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            total = max(1, self.global_step)
            return {
                "round_num": self.round_num,
                "total_batches": self.global_step,
                "avg_loss": self.total_loss_sum / total,
                "avg_policy_loss": self.total_policy_loss_sum / total,
                "avg_value_loss": self.total_value_loss_sum / total,
                "avg_health_loss": self.total_health_loss_sum / total,
            }

    def round_history_snapshot(self) -> List[Dict]:
        """返回逐轮指标历史的浅拷贝。"""
        with self._stats_lock:
            return list(self.round_history)

    def finalize(self) -> None:
        """优雅退出/结束训练时触发最终 checkpoint 强制保存与导出。"""
        self.save_checkpoint(force=True)
        print(f"[TR-{self.variant.id}] 🎉 最终 Checkpoint 强制保存与导出已完成")

