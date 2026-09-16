"""banqi/training/lr_schedule.py — 学习率调度与训练循环小工具。"""

from __future__ import annotations

import torch.optim.lr_scheduler as lr_scheduler


def is_stopped(stop_event) -> bool:
    """统一停止信号判断：支持 None / list 间接引用 / Event / 布尔。"""
    if stop_event is None:
        return False
    if isinstance(stop_event, list):
        return bool(stop_event[0])
    if hasattr(stop_event, "is_set"):
        return stop_event.is_set()
    return bool(stop_event)


def resolve_lr_decay_batches(cfg) -> int:
    """LR 余弦的时间跨度（batch 步数）。

    `LR_DECAY_ROUNDS > 0` 时以「训练轮数」表达计划，按当前每轮训练量
    （`cfg.batches_per_round()`，由节流阈值 / TRAIN_BATCH / TRAIN_EPOCHS_PER_ROUND 推导）
    折算成 batch 步数 —— 这样调整节流或批次大小时，LR 计划在「多少轮/多少样本后退到
    MIN_LR」上的含义不变，不必重新猜 `LR_DECAY_STEPS`。
    否则沿用 `LR_DECAY_STEPS`（历史口径：直接按 batch 计）。
    """
    rounds = int(cfg.LR_DECAY_ROUNDS)
    if rounds > 0:
        return max(1, rounds * cfg.batches_per_round())
    return max(int(cfg.LR_DECAY_STEPS or 1000), 1)


def make_cosine_clamp_scheduler(optimizer, cfg, t_max_batches: int):
    """余弦衰减到 MIN_LR 后钳位保持，不周期回升。

    原生 CosineAnnealingLR 在训练步数超过 T_max 后学习率会按余弦周期回升，
    导致长周期自对弈训练后期梯度偏大、收敛震荡。这里用 LambdaLR 实现：
    前 `t_max_batches` 步按半周期余弦从 LEARNING_RATE 平滑降到 MIN_LR，
    之后钳位在 MIN_LR 保持，兼顾余弦退火的平滑收敛与长训练稳定性。

    `t_max_batches` 由 `resolve_lr_decay_batches(cfg)` 计算；调用方（训练循环）
    须在每个**实际发生参数更新**的 batch 上调用一次 `scheduler.step()`。
    """
    t_max = max(int(t_max_batches), 1)
    eta_min = float(cfg.MIN_LR or 1e-6)
    eta_max = float(cfg.LEARNING_RATE or 1e-4)
    # LambdaLR 的 lambda 返回的是相对 initial_lr 的比例因子
    min_ratio = eta_min / eta_max if eta_max > 0 else 1e-4

    def lr_lambda(epoch: int) -> float:
        import math
        t = min(epoch, t_max) / t_max            # 钳位到 [0,1]
        # 半周期余弦：t=0 -> 1.0，t=1 -> 0.0（即 MIN_LR）
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_ratio + (1.0 - min_ratio) * cos

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
