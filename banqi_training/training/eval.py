"""banqi/training/eval.py — 训练评估模块。

包含固定验证集构建、分层均衡筛选、价值漂移评估、策略头命中率评估及周期性对战评估。
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from banqi_training.constants import build_constants
from banqi_training.tb_logger import add_scalar
from banqi_training.variant import Variant

# 评估常量（原 banqi/eval.py 迁移而来）。
# 对战评估（vs prev / vs 规则对手）已移除：分布式形态下由调度器 gatekeeper
# rating（candidate vs best + GSPRT 判停）承担模型对战评估职责。


def select_balanced_fixed_samples(pool: List[Dict], n_fixed: int) -> List[Dict]:
    """从原始样本池中按终局结果分层均衡筛选固定验证局面。"""
    if not pool or n_fixed <= 0:
        return []
    buckets: Dict[int, List[Dict]] = {1: [], -1: [], 0: []}
    for s in pool:
        gr = s.get("game_result_value", 0.0)
        key = 1 if gr > 0 else (-1 if gr < 0 else 0)
        buckets[key].append(s)
    per_bucket = max(1, n_fixed // 3)
    selected: List[Dict] = []
    for key in (1, -1, 0):
        selected.extend(buckets[key][:per_bucket])
    if len(selected) < n_fixed:
        seen = {id(s) for s in selected}
        for s in pool:
            if id(s) in seen:
                continue
            selected.append(s)
            if len(selected) >= n_fixed:
                break
    return selected[:n_fixed]


def build_fixed_eval(samples: List[Dict], variant: Variant) -> Optional[Dict]:
    """将 Dict 列表样本构建为 numpy array 组成的固定验证集。"""
    if not samples:
        return None
    C = build_constants(variant)
    aspace = C.ACTION_SPACE_SIZE
    try:
        masks = np.array([s["action_mask"] for s in samples], dtype=np.float32)
        if masks.ndim == 1:
            masks = np.ones((len(samples), aspace), dtype=np.float32)
        return {
            "boards": np.stack(
                [
                    np.array(s["board_state"], dtype=np.float32).reshape(
                        C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS
                    )
                    for s in samples
                ]
            ),
            "scalars": np.stack(
                [np.array(s["scalar_state"], dtype=np.float32) for s in samples]
            ),
            "results": np.array(
                [s.get("game_result_value", 0.0) for s in samples],
                dtype=np.float32,
            ),
            "masks": masks,
            "teacher_actions": np.array(
                [
                    int(s["teacher_action"])
                    if s.get("teacher_action") is not None
                    else -1
                    for s in samples
                ],
                dtype=np.int64,
            ),
        }
    except Exception:
        return None


def prefill_from_archive(buffer, variant: Variant, cfg) -> Optional[Dict]:
    """从冷存储归档加载历史 episode 预填充训练 buffer 并构建固定验证集。"""
    n_games = cfg.ARCHIVE_PREFILL_GAMES
    if not n_games:
        return None
    from banqi_training.storage import load_jsonl_episodes
    from banqi_training.training.buffer import episode_to_samples

    here = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dirs = [
        cfg.ARCHIVE_PREFILL_DIR,
        variant.archive_dir or "",
        os.path.join(here, "training_data", f"archive_{variant.id}"),
        os.path.join(here, "training_data", f"archive_{variant.id}_imitate"),
    ]
    archive_dir = next((d for d in dirs if d and os.path.isdir(d)), None)
    if not archive_dir:
        print(f"[TR-{variant.id}] ⚠️ 冷存储预填充：未找到归档目录，跳过")
        return None
    try:
        t0 = time.time()
        episodes = load_jsonl_episodes(archive_dir, limit_games=n_games)
        samples: List[Dict] = []
        for ep in episodes:
            samples.extend(episode_to_samples(ep))
        if samples:
            buffer.add_samples(samples)
            print(
                f"[TR-{variant.id}] 🗃️ 冷存储预填充: 从 {archive_dir} 加载 "
                f"{len(episodes)} 局 → {len(samples)} 样本 "
                f"(Buffer={len(buffer)}, 耗时 {time.time() - t0:.1f}s)"
            )
        n_fixed = cfg.VALUE_DRIFT_NUM_POSITIONS
        if n_fixed > 0 and samples:
            fixed = build_fixed_eval(samples[:n_fixed], variant)
            if fixed:
                print(
                    f"[TR-{variant.id}] 🎯 固定价值验证集（归档）"
                    f"{len(fixed['boards'])} 局面已就绪"
                )
                return fixed
    except Exception as e:
        print(f"[TR-{variant.id}] ⚠️ 冷存储预填充失败 ({e})，继续正常训练")
    return None


def eval_value_drift(
    model: torch.nn.Module,
    device: torch.device,
    fixed_eval: Optional[Dict],
    global_step: int,
    tag: str,
    round_num: int,
) -> None:
    """在固定验证集上评估价值头预测，监测价值漂移。"""
    if fixed_eval is None:
        return
    try:
        model.eval()
        with torch.inference_mode():
            b = torch.from_numpy(
                np.ascontiguousarray(fixed_eval["boards"])
            ).to(device)
            s = torch.from_numpy(
                np.ascontiguousarray(fixed_eval["scalars"])
            ).to(device)
            if getattr(model, "enable_health", False):
                _, values, _ = model(b, s)
            else:
                _, values = model(b, s)
        pred = values.cpu().numpy().reshape(-1).astype(np.float32)
        model.train()
        gr = fixed_eval["results"]
        corr = (
            float(np.corrcoef(pred, gr)[0, 1])
            if len(pred) > 2 and np.std(pred) > 1e-6 and np.std(gr) > 1e-6
            else 0.0
        )
        sep = (
            float(pred[gr > 0].mean() - pred[gr < 0].mean())
            if (np.any(gr > 0) and np.any(gr < 0))
            else 0.0
        )
        print(
            f"{tag} 📊 价值漂移 Round#{round_num}: pred_mean={pred.mean():+.3f} "
            f"std={pred.std():.3f} corr(终局)={corr:.3f} 胜负区分度={sep:.3f}"
        )
        add_scalar("value_drift/pred_mean", pred.mean(), global_step)
        add_scalar("value_drift/pred_std", pred.std(), global_step)
        add_scalar("value_drift/corr_result", corr, global_step)
        add_scalar("value_drift/sep", sep, global_step)
    except Exception as e:
        print(f"{tag} ⚠️ 价值漂移评估失败 ({e})")


def eval_policy_accuracy(
    model: torch.nn.Module,
    device: torch.device,
    fixed_eval: Optional[Dict],
    global_step: int,
    tag: str,
    round_num: int,
) -> None:
    """在固定验证集上评估策略头 Top-1 / Top-3 命中率。"""
    if fixed_eval is None:
        return
    teacher = fixed_eval["teacher_actions"]
    if teacher.size == 0 or int((teacher >= 0).sum()) == 0:
        return
    try:
        model.eval()
        with torch.inference_mode():
            b = torch.from_numpy(
                np.ascontiguousarray(fixed_eval["boards"])
            ).to(device)
            s = torch.from_numpy(
                np.ascontiguousarray(fixed_eval["scalars"])
            ).to(device)
            if getattr(model, "enable_health", False):
                logits, _, _ = model(b, s)
            else:
                logits, _ = model(b, s)
        logits = logits.cpu().numpy().astype(np.float32)
        model.train()
        masks = fixed_eval["masks"].astype(np.float32)
        ml_all = np.where(np.isfinite(logits), logits, -1e9).copy()
        ml_all = np.where(masks >= 0.5, ml_all, -1e9)
        valid = teacher >= 0
        if int(valid.sum()) == 0:
            return
        ml = ml_all[valid]
        ta = teacher[valid]
        top1_idx = np.argmax(ml, axis=1)
        k = min(3, ml.shape[1])
        topk_idx = np.argpartition(-ml, k - 1, axis=1)[:, :k]
        hit1 = float(np.mean(top1_idx == ta))
        hit3 = float(np.mean(np.any(topk_idx == ta[:, None], axis=1)))
        n_eval = int(valid.sum())
        print(
            f"{tag} 🎯 策略头命中 Round#{round_num}: Top-1={hit1:.3f} "
            f"Top-3={hit3:.3f}（{n_eval} 局面 vs 启发式/MCTS 最优动作）"
        )
        add_scalar("policy_acc/top1_vs_teacher", hit1, global_step)
        add_scalar("policy_acc/top3_vs_teacher", hit3, global_step)
        add_scalar("policy_acc/n_positions", n_eval, global_step)
    except Exception as e:
        print(f"{tag} ⚠️ 策略头验证失败 ({e})")

