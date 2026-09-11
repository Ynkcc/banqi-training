"""banqi/training/lr_schedule.py — 学习率调度与训练循环小工具。"""

from __future__ import annotations

import torch.optim.lr_scheduler as lr_scheduler


def compute_lr_scale(global_step: int, cfg) -> float:
    """线性 warmup 系数：前 LR_DECAY_STEPS 步从 MIN_LR 线性升到 1.0。

    仅用于 checkpoint 恢复时对初始 lr 做缩放，避免从 0 直接起训导致的震荡；
    之后的余弦下降交由余弦钳位调度器处理。
    """
    if global_step <= 0:
        return 1.0
    decay_steps = max(int(cfg.LR_DECAY_STEPS or 1000), 1)
    min_lr = float(cfg.MIN_LR or 1e-6)
    frac = min(global_step, decay_steps) / decay_steps
    # 目标 = MIN_LR + (1 - MIN_LR)*frac，最小不低于 MIN_LR
    return max(min_lr, 1.0 - (1.0 - min_lr) * (1.0 - frac))


def is_stopped(stop_event) -> bool:
    """统一停止信号判断：支持 None / list 间接引用 / Event / 布尔。"""
    if stop_event is None:
        return False
    if isinstance(stop_event, list):
        return bool(stop_event[0])
    if hasattr(stop_event, "is_set"):
        return stop_event.is_set()
    return bool(stop_event)


def make_cosine_clamp_scheduler(optimizer, cfg):
    """余弦衰减到 MIN_LR 后钳位保持，不周期回升。

    原生 CosineAnnealingLR 在训练步数超过 T_max 后学习率会按余弦周期回升，
    导致长周期自对弈训练后期梯度偏大、收敛震荡。这里用 LambdaLR 实现：
    前 LR_DECAY_STEPS 步按半周期余弦从 LEARNING_RATE 平滑降到 MIN_LR，
    之后钳位在 MIN_LR 保持，兼顾余弦退火的平滑收敛与长训练稳定性。
    """
    t_max = max(int(cfg.LR_DECAY_STEPS or 1000), 1)
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
