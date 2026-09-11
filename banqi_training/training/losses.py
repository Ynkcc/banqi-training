"""banqi/training/losses.py — 单 batch 训练步骤与整轮训练调度。

train_step：单 batch 前向/反向 + 数值安全校验（跳过非有限输入/目标/梯度）。
run_training_epochs：在完整 replay buffer 上训练若干 epoch，scheduler 按
batch 步进以匹配 CosineAnnealingLR 的 T_max（batch 数）。
"""

from __future__ import annotations

import random
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def _resolve_device(spec: str) -> "torch.device":
    """按 config.TRAIN_DEVICE 解析训练设备；auto = CUDA 可用则用 CUDA。"""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def make_hl_gauss_target(target_bins: torch.Tensor, num_classes: int, sigma: float = 1.5) -> torch.Tensor:
    """由 1D 离散桶标签生成 HL-Gauss (Histogram Loss with Gaussian Label Smoothing) 高斯目标分布 (B, K)。"""
    bins = torch.arange(num_classes, device=target_bins.device, dtype=torch.float32).unsqueeze(0)
    targets = target_bins.unsqueeze(1).float()
    weights = torch.exp(-0.5 * ((bins - targets) / sigma) ** 2)
    return weights / weights.sum(dim=1, keepdim=True)


# 单 batch 训练统计（供 TensorBoard 记录）：
#   total/policy/value/health：四类 loss；grad_norm：clip 前梯度范数（发散预警）；
#   entropy：目标策略平均熵（探索健康度）；value_mean/std：价值目标分布。
TrainStepStats = namedtuple(
    "TrainStepStats", "total policy value health grad_norm entropy value_mean value_std"
)
_ZERO_STATS = TrainStepStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def train_step(model, optimizer, batch_data, device, ema_model=None, ema_decay: float = 0.999,
               health_enabled: bool = False, health_loss_weight: float = 0.0,
               health_gauss_sigma: float = 1.5) -> TrainStepStats:
    model.train()
    boards_t, scalars_t, target_probs_t, target_values_t, masks_t, full_t, target_health_bin_t = batch_data

    boards_t = boards_t.to(device, non_blocking=True)
    scalars_t = scalars_t.to(device, non_blocking=True)
    target_probs_t = target_probs_t.to(device, non_blocking=True)
    target_values_t = target_values_t.to(device, non_blocking=True).view(-1, 1)
    masks_t = masks_t.to(device, non_blocking=True)
    # 算力分配随机化的 Full Search 标记：1=Full（参与训练），0=Fast（仅保留，不训练）
    full_t = full_t.to(device, non_blocking=True).float()
    target_health_bin_t = target_health_bin_t.to(device, non_blocking=True).long().view(-1)

    # ---- 选择性使用：batch 内无 Full Search 样本则不训练（Fast 样本仅保留供未来逻辑）----
    if bool(full_t.sum() == 0):
        return _ZERO_STATS

    # ---- 数据有效性（非有限/非法策略）已由 DataBuffer.add_samples 入队时
    # 前置过滤，此处假定输入完全合法，不再做冗余防御校验。----
    optimizer.zero_grad()
    if health_enabled:
        logits, values, health_logits = model(boards_t, scalars_t)
    else:
        logits, values = model(boards_t, scalars_t)

    # ---- 安全 mask：用 -1e9 屏蔽非法动作（替代 (mask-1)*1e9）----
    # 原实现 logits + (mask-1)*1e9 在 logits 含 +inf 时会产生 inf -> log_softmax
    # 得到 NaN（inf-inf）。改用 masked_fill 只把非法位置置为极大负值，
    # 配合下方梯度有限性检查，从源头杜绝 NaN 传播。
    masked_logits = logits.masked_fill(masks_t < 0.5, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=1)
    # 选择性使用：仅让 Full Search 样本参与策略/价值 loss（Fast 样本按 0 权重屏蔽）
    num_full = full_t.sum().clamp_min(1.0)
    per_sample_policy = -(target_probs_t * log_probs).sum(dim=1)  # (B,)
    policy_loss = (per_sample_policy * full_t).sum() / num_full
    per_sample_value = F.mse_loss(values, target_values_t, reduction="none").view(-1)
    value_loss = (per_sample_value * full_t).sum() / num_full
    total_loss = policy_loss + value_loss

    # ---- 血量头：HL-Gauss (高斯标签平滑交叉熵/KL散度，仅启用时)，权重 α 缩放，Fast 样本按 0 权重屏蔽 ----
    health_loss = torch.tensor(0.0, device=device)
    if health_enabled:
        log_health_probs = F.log_softmax(health_logits, dim=1)
        target_health_dist = make_hl_gauss_target(target_health_bin_t, health_logits.size(1), sigma=health_gauss_sigma)
        per_sample_health = F.kl_div(log_health_probs, target_health_dist, reduction="none").sum(dim=1)
        health_loss = (per_sample_health * full_t).sum() / num_full
        total_loss = total_loss + health_loss_weight * health_loss

    # ---- 数值安全：loss / 梯度非有限直接抛出（不静默跳过 batch）----
    # 静默跳过只会掩盖上游 DataBuffer / MCTS 样本收集阶段引入非法数据的根因；
    # 一旦非有限就尽早失败，让根因暴露。数据合法性应在入队环节保证。
    if not torch.isfinite(total_loss):
        raise RuntimeError(
            f"[TR] 检测到非有限 loss（policy={float(policy_loss):.4f} "
            f"value={float(value_loss):.4f}）。数据应在 DataBuffer 入队时过滤，"
            f"请排查上游 MCTS/自对弈产生的脏数据，而非静默跳过 batch。"
        )

    total_loss.backward()
    # ---- 梯度有限性检查：NaN/Inf 梯度静默放行是权重被污染的主通道 ----
    # 一旦出现非有限梯度，clip_grad_norm_ 返回 NaN 且 optimizer.step() 会把
    # 整份权重写成 NaN。故在 clip 前显式检测并直接抛出。
    grad_ok = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if not grad_ok:
        raise RuntimeError(
            "[TR] 检测到非有限梯度，不更新权重。请排查数据/模型状态，而非静默跳过 batch。"
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # ---- EMA 影子权重与 BatchNorm 缓冲区更新 ----
    if ema_model is not None:
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
            for (n_ema, b_ema), (n_src, b_src) in zip(ema_model.named_buffers(), model.named_buffers()):
                if b_src.dtype.is_floating_point:
                    b_ema.data.mul_(ema_decay).add_(b_src.data, alpha=1.0 - ema_decay)

    # 记录目标分布统计（无需梯度）：策略熵 + 价值目标 mean/std，供 TB 观测
    with torch.no_grad():
        log_p = torch.log(target_probs_t.clamp_min(1e-12))
        entropy = float(-(target_probs_t * log_p).sum(dim=1).mean())
        value_mean = float(target_values_t.mean())
        value_std = float(target_values_t.std())

    return TrainStepStats(
        total=total_loss.item(),
        policy=policy_loss.item(),
        value=value_loss.item(),
        health=health_loss.item(),
        grad_norm=float(grad_norm),
        entropy=entropy,
        value_mean=value_mean,
        value_std=value_std,
    )


def run_training_epochs(model, optimizer, scheduler, buffer, num_epochs,
                        device, max_batches: Optional[int] = None,
                        ema_model=None, ema_decay: float = 0.999,
                        health_enabled: bool = False, health_loss_weight: float = 0.0,
                        health_gauss_sigma: float = 1.5):
    """
    在完整 replay buffer 上训练指定个 epoch。
    scheduler.step() 按 batch 步进以匹配 CosineAnnealingLR 的 T_max (batch 数)。

    max_batches: 限制本轮总训练批次数。当每轮新增数据量远小于 buffer（如 RL
    自对弈慢、每轮仅几百样本而 buffer 上万）时，若每轮对整个 buffer 训练多
    epoch，旧数据会被反复训练导致过拟合/棋力退化。限制训练量与新数据量匹配：
      每轮批次 ≈ 新样本数/batch × epochs，数据量大时自动恢复全覆盖训练。

    返回 (epoch 平均 loss 列表, 累计训练 batch 数)。
    """
    total_batches = 0
    epoch_results = []
    # 新近优先采样：RL 自对弈数据随模型演化而过时，聚焦最近新增样本可避免
    # 旧数据反复训练导致的棋力退化。仅在 max_batches 限制训练量（批量聚焦）时启用；
    # 全量训练（max_batches=None）时保留均匀采样以覆盖整个 buffer。
    recent_enabled = max_batches is not None and buffer.cfg.RECENT_SAMPLE_ENABLED
    for epoch in range(num_epochs):
        # 确定本轮实际训练的 batch 数：受 buffer 容量与剩余 max_batches 双重约束
        if max_batches is not None:
            remaining = max_batches - total_batches
            if remaining <= 0:
                break
            num_batches = min(
                len(buffer) // buffer.cfg.TRAIN_BATCH, remaining
            )
        else:
            num_batches = len(buffer) // buffer.cfg.TRAIN_BATCH
        if num_batches == 0:
            break
        # 按需采样：只抽取本轮实际需要的样本数，避免生成全量索引的巨大浪费。
        # 新近优先用指数衰减加权采样（聚焦最新数据），否则均匀采样。
        needed = num_batches * buffer.cfg.TRAIN_BATCH
        if recent_enabled:
            indices = buffer.sample_recent_indices(needed)
        else:
            indices = np.random.permutation(len(buffer))[:needed]
        batch_total_l, batch_pol_l, batch_val_l, batch_health_l = 0.0, 0.0, 0.0, 0.0
        batch_grad_l, batch_ent_l, batch_vm_l, batch_vs_l = 0.0, 0.0, 0.0, 0.0
        for step in range(num_batches):
            batch_indices = indices[step * buffer.cfg.TRAIN_BATCH: (step + 1) * buffer.cfg.TRAIN_BATCH]
            batch_data = buffer.get_batch(batch_indices)
            s = train_step(model, optimizer, batch_data, device, ema_model=ema_model,
                           ema_decay=ema_decay, health_enabled=health_enabled,
                           health_loss_weight=health_loss_weight,
                           health_gauss_sigma=health_gauss_sigma)
            scheduler.step()
            batch_total_l += s.total
            batch_pol_l += s.policy
            batch_val_l += s.value
            batch_health_l += s.health
            batch_grad_l += s.grad_norm
            batch_ent_l += s.entropy
            batch_vm_l += s.value_mean
            batch_vs_l += s.value_std
            total_batches += 1

        epoch_results.append((
            batch_total_l / num_batches,
            batch_pol_l / num_batches,
            batch_val_l / num_batches,
            batch_health_l / num_batches,
            batch_grad_l / num_batches,
            batch_ent_l / num_batches,
            batch_vm_l / num_batches,
            batch_vs_l / num_batches,
        ))
    return epoch_results, total_batches
