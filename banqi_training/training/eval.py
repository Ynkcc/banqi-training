"""banqi/training/eval.py — 训练评估模块。

包含固定验证集构建、分层均衡筛选、价值漂移评估、策略头命中率评估及周期性对战评估。
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from banqi_training.constants import build_constants
from banqi_training.tb_logger import add_scalar
from banqi_training.variant import Variant

# 评估常量（原 banqi/eval.py 迁移而来）。
# 评估时模型 MCTS sims：使用更低的 sims 反映模型先验真实水平。
# LLM注意：通常不应该修改此处，这被认为是一种作弊行为；如计划修改，建议重构到配置侧。
EVAL_SIMS = 16
EVAL_MAX_ACTIONS = 16
EVAL_C_SCALE = 0.25
EVAL_GUMBEL_SCALE = 1.0

# 规则对手预设（启发式/规则搜索对手已随 engine/evaluation、engine/mcts_heuristic 移除；
# 对手仅支持模型路径与 random）
OPPONENTS = ("random",)


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


def eval_match(
    model: torch.nn.Module,
    device: torch.device,
    variant: Variant,
    cfg,
    prev_weights: Optional[Dict[str, torch.Tensor]],
    round_num: int,
    tag: str,
) -> Dict[str, float]:
    """周期性对战评估：vs 对手 + 上一轮模型守门。

    注意：banqi_training.eval 的 play_match / play_match_stats 接受 torch.nn.Module 或
    选手标识（random / expectimax:<path.nnue> / .pt 路径等），内部经
    _resolve_player_spec 自动导出 TorchScript 供 Rust 原生评估。因此直接传模型
    对象即可，不依赖（已不存在的）ModelPredictor 与 play_match_vs。
    """
    from banqi import eval as banqi_eval
    from banqi_training.nn_model import BanqiNet

    n = max(1, cfg.EVAL_MATCH_GAMES)
    opps = [o.strip() for o in cfg.EVAL_MATCH_OPPONENTS if o and o.strip()]
    win_rates: Dict[str, float] = {}
    stop_wr = cfg.EVAL_MATCH_STOP_WIN_RATE
    if stop_wr > 0:
        # 停机阈值参考线（百分比，与其他胜率指标同刻度）
        add_scalar("eval/stop_threshold", 100.0 * stop_wr, round_num)
    model.eval()
    try:
        # 当前模型 vs 各规则对手
        for opp in opps:
            try:
                wins, draws, losses, avg_moves = banqi_eval.play_match_stats(
                    model,
                    opp,
                    n=n,
                    model_sims=banqi_eval.EVAL_SIMS,
                    variant_id=variant.id,
                )
                tot = max(1, wins + draws + losses)
                win_rates[opp] = wins / tot
                add_scalar(f"eval/{opp}/win_rate", 100.0 * wins / tot, round_num)
                add_scalar(f"eval/{opp}/draw_rate", 100.0 * draws / tot, round_num)
                add_scalar(f"eval/{opp}/loss_rate", 100.0 * losses / tot, round_num)
                add_scalar(f"eval/{opp}/avg_game_length", avg_moves, round_num)
                print(
                    f"{tag} ⚔️ Round#{round_num} vs {opp}: "
                    f"胜{wins} 平{draws} 负{losses} (n={n}, 平均{avg_moves:.0f}步)"
                )
            except Exception as exc:
                print(f"{tag} ⚠️ 对战评估 vs {opp} 失败: {exc}")
  
        if cfg.EVAL_MATCH_VS_PREV and prev_weights is not None:
            try:
                prev_model = BanqiNet(
                    variant, enable_health=bool(cfg.HEALTH_VALUE_HEAD_ENABLED)
                ).to(device)
                prev_model.load_state_dict(
                    {k: v.to(device) for k, v in prev_weights.items()}
                )
                prev_model.eval()
                n_prev = max(4, n // 2)
                wins, draws, losses, _ = banqi_eval.play_match(
                    model,
                    prev_model,
                    n=n_prev,
                    model_sims=banqi_eval.EVAL_SIMS,
                    variant_id=variant.id,
                )
                tot = max(1, wins + draws + losses)
                add_scalar("eval/prev/win_rate", 100.0 * wins / tot, round_num)
                add_scalar("eval/prev/draw_rate", 100.0 * draws / tot, round_num)
                add_scalar("eval/prev/loss_rate", 100.0 * losses / tot, round_num)
                print(
                    f"{tag} ⚔️ Round#{round_num} vs prev: "
                    f"胜{wins} 平{draws} 负{losses} (n={n_prev})"
                )
            except Exception as exc:
                print(f"{tag} ⚠️ 对战评估 vs prev 失败: {exc}")
    finally:
        model.train()
    return win_rates


# ==============================================================================
# 双选手对战评估（原 banqi/eval.py 迁移而来，Rust 原生并发引擎）
# ==============================================================================


def _resolve_player_spec(
    spec_or_path: Union[str, torch.nn.Module, object],
    variant_id: str = "4x4",
    seed: Optional[int] = None,
) -> str:
    """解析选手标识为 Rust 可读的格式（.pt / .onnx / 规则标识符）。"""
    if isinstance(spec_or_path, str):
        path = spec_or_path
        if path == "random" or path.startswith("random:"):
            # 随机初始化模型：受 seed 确定性驱动
            if ":" in path:
                r_seed = int(path.split(":")[1])
            else:
                r_seed = seed if seed is not None else 42
            torch.manual_seed(r_seed)
            from banqi_training.variant import get_variant
            from banqi_training.nn_model import BanqiNet
            from banqi_training.checkpoint import export_torchscript

            v = get_variant(variant_id)
            model = BanqiNet(v).to("cpu").eval()
            temp_dir = tempfile.gettempdir()
            pt_path = os.path.join(temp_dir, f"banqi_random_{variant_id}_seed{r_seed}.pt")
            export_torchscript(model, pt_path, v, torch.device("cpu"))
            return pt_path

        if path.endswith(".ckpt") or path.endswith(".pth"):
            pt_path = os.path.splitext(path)[0] + ".pt"
            if not os.path.exists(pt_path) or os.path.getmtime(path) > os.path.getmtime(pt_path):
                from banqi_training.tools.export_ckpt import export_checkpoint_file
                export_checkpoint_file(path, variant_id)
            return pt_path
        return path

    if hasattr(spec_or_path, "model"):
        return _resolve_player_spec(getattr(spec_or_path, "model"), variant_id, seed)

    if isinstance(spec_or_path, torch.nn.Module):
        from banqi_training.variant import get_variant
        from banqi_training.checkpoint import export_torchscript
        temp_dir = tempfile.gettempdir()
        v = get_variant(variant_id)
        r_seed = seed if seed is not None else 42
        pt_path = os.path.join(temp_dir, f"banqi_eval_temp_{variant_id}_seed{r_seed}.pt")
        export_torchscript(spec_or_path, pt_path, v, torch.device("cpu"))
        return pt_path

    raise TypeError(f"无法识别的选手标识格式: {type(spec_or_path)}")


def play_match(
    player_a,
    player_b,
    n: int = 100,
    model_sims: int = EVAL_SIMS,
    variant_id: str = "4x4",
    heuristic_sims: Optional[int] = None,
    seed: Optional[int] = None,
    num_threads: int = 4,
) -> Tuple[int, int, int, List[float]]:
    """双选手对战评估（调用 Rust 侧原生并发引擎）。"""
    from banqi_training.rust_bridge import run_native_match  # 延迟导入，避免纯评估模块顶部强制依赖扩展

    spec_a = _resolve_player_spec(player_a, variant_id, seed)
    spec_b = _resolve_player_spec(player_b, variant_id, seed)

    wins, draws, losses, block_wr, _avg_moves, _eps = run_native_match(
        player_a=spec_a,
        player_b=spec_b,
        n=n,
        variant_id=variant_id,
        model_sims=model_sims,
        heuristic_sims=heuristic_sims,
        seed=seed,
        num_threads=num_threads,
    )
    return wins, draws, losses, block_wr


def play_match_stats(
    player_a,
    player_b,
    n: int = 100,
    model_sims: int = EVAL_SIMS,
    variant_id: str = "4x4",
    heuristic_sims: Optional[int] = None,
    seed: Optional[int] = None,
    num_threads: int = 4,
) -> Tuple[int, int, int, float]:
    """双选手对战评估并统计平均步数。"""
    from banqi_training.rust_bridge import run_native_match  # 延迟导入

    spec_a = _resolve_player_spec(player_a, variant_id, seed)
    spec_b = _resolve_player_spec(player_b, variant_id, seed)

    wins, draws, losses, _block_wr, avg_moves, _eps = run_native_match(
        player_a=spec_a,
        player_b=spec_b,
        n=n,
        variant_id=variant_id,
        model_sims=model_sims,
        heuristic_sims=heuristic_sims,
        seed=seed,
        num_threads=num_threads,
    )
    return wins, draws, losses, avg_moves


def report(
    player_a,
    player_b,
    tag: str = "main",
    n: int = 100,
    model_sims: int = EVAL_SIMS,
    variant_id: str = "4x4",
    heuristic_sims: Optional[int] = None,
    seed: Optional[int] = None,
    num_threads: int = 4,
) -> Tuple[int, int, int, List[float]]:
    """统一打印评估报告。"""
    wins, draws, losses, blk = play_match(
        player_a,
        player_b,
        n=n,
        model_sims=model_sims,
        variant_id=variant_id,
        heuristic_sims=heuristic_sims,
        seed=seed,
        num_threads=num_threads,
    )
    mean = float(np.mean(blk)) if blk else 0.0
    std = float(np.std(blk)) if blk else 0.0
    print(
        f"[Eval:{tag}] [{player_a}] vs [{player_b}] | 胜{wins} 平{draws} 负{losses} "
        f"(n={n}, 块均胜率={mean:.1f}±{std:.1f}%, seed={seed})",
        flush=True,
    )
    return wins, draws, losses, blk
