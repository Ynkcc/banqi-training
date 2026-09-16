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


def _policy_top2(samples: List[Dict], masks: np.ndarray, aspace: int) -> np.ndarray:
    """由策略目标 π' 取每个局面的前二动作（非法动作屏蔽，缺失策略填 -1）。

    π' = 搜索产物（softmax(logit + σ·completed_Q)），是比「单次采样动作」稳定得多的
    评测参照：实测记录动作有 29.7% 不等于 argmax(π')，用它当标签会低估策略头。
    """
    out = np.full((len(samples), 2), -1, dtype=np.int64)
    for i, s in enumerate(samples):
        raw = s.get("policy_probs")
        if raw is None:
            continue
        p = np.asarray(raw, dtype=np.float64)[:aspace]
        if p.shape[0] < aspace:
            p = np.pad(p, (0, aspace - p.shape[0]))
        p = np.where(masks[i] >= 0.5, p, -1.0)   # 非法动作置负，避免被选进 top2
        if (p < 0).all():
            continue
        out[i] = np.argsort(-p)[:2]
    return out


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
            # 终局归一化子力差：value target 改用 game_hp 时 corr(终局) 会自然下降，
            # 需以本项为基准判断价值头是否真的学到了子力信息（见 eval_value_drift）。
            "health_diffs": np.array(
                [float(s.get("health_diff", 0.0)) for s in samples],
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
            # 搜索偏好的两个口径（见 eval_policy_accuracy）：
            #   teacher_actions = 记录动作（自对弈下即 Gumbel 采样结果，约 30% 不等于
            #                     策略目标自己的 argmax，故它是一个带噪的评测标签）；
            #   pi_top2         = 训练目标 π' 的前二动作（搜索真正的偏好，噪声更低）。
            # 两者并列报告，避免用带噪标签低估策略头（实测 0.60 vs 0.87）。
            "pi_top2": _policy_top2(samples, masks, aspace),
        }
    except Exception:
        return None


def prefill_from_archive(buffer, variant: Variant, cfg) -> Optional[Dict]:
    """从冷存储归档加载历史 episode 预填充训练 buffer 并构建固定验证集。"""
    n_games = cfg.ARCHIVE_PREFILL_GAMES
    if not n_games:
        return None
    from banqi_training.episode_codec import DATA_RESNET
    from banqi_training.storage import load_episodes_from_dir
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
        episodes = load_episodes_from_dir(
            archive_dir, limit_games=n_games, variant=variant.id, kind=DATA_RESNET
        )
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
        # 子力差基准：value target 改用 game_hp 后 corr(终局) 会自然下降，
        # 本项用于区分「目标语义改变」与「价值头退化」。
        hp = fixed_eval.get("health_diffs")
        corr_hp = (
            float(np.corrcoef(pred, hp)[0, 1])
            if hp is not None and len(pred) > 2
            and np.std(pred) > 1e-6 and np.std(hp) > 1e-6
            else 0.0
        )
        sep = (
            float(pred[gr > 0].mean() - pred[gr < 0].mean())
            if (np.any(gr > 0) and np.any(gr < 0))
            else 0.0
        )
        print(
            f"{tag} 📊 价值漂移 Round#{round_num}: pred_mean={pred.mean():+.3f} "
            f"std={pred.std():.3f} corr(终局)={corr:.3f} corr(子力差)={corr_hp:.3f} "
            f"胜负区分度={sep:.3f}"
        )
        add_scalar("value_drift/pred_mean", pred.mean(), global_step)
        add_scalar("value_drift/pred_std", pred.std(), global_step)
        add_scalar("value_drift/corr_result", corr, global_step)
        add_scalar("value_drift/corr_health_diff", corr_hp, global_step)
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
        # 第二口径：与策略目标 π' 的 argmax / top2 对照。记录动作含 Gumbel 采样噪声
        # （实测 29.7% 不等于 argmax(π')），单看 top1_vs_teacher 会低估策略头，
        # 也容易把「标签噪声」误判成「策略头学不会搜索决策」。
        pi2 = fixed_eval.get("pi_top2")
        hit_pi1 = hit_pi2 = float("nan")
        if pi2 is not None:
            pi2 = pi2[valid]
            ok = pi2[:, 0] >= 0
            if ok.any():
                p2 = pi2[ok]
                t1 = top1_idx[ok]
                hit_pi1 = float(np.mean(t1 == p2[:, 0]))
                hit_pi2 = float(np.mean(np.any(p2 == t1[:, None], axis=1)))
        print(
            f"{tag} 🎯 策略头命中 Round#{round_num}: "
            f"Top-1={hit1:.3f} Top-3={hit3:.3f}（vs 记录动作，含采样噪声） | "
            f"Top-1={hit_pi1:.3f} Top-2={hit_pi2:.3f}（vs 搜索偏好 π'，{n_eval} 局面）"
        )
        add_scalar("policy_acc/top1_vs_teacher", hit1, global_step)
        add_scalar("policy_acc/top3_vs_teacher", hit3, global_step)
        add_scalar("policy_acc/top1_vs_pi", hit_pi1, global_step)
        add_scalar("policy_acc/top2_vs_pi", hit_pi2, global_step)
        add_scalar("policy_acc/n_positions", n_eval, global_step)
    except Exception as e:
        print(f"{tag} ⚠️ 策略头验证失败 ({e})")

