"""banqi/trainer_cli/config_resolver.py — 配置解析与校验。

根据命令行/编程式参数构造 Config 实例（含命令行覆盖），并输出配置告警。
"""

from __future__ import annotations

import os

from banqi_training.config import Config, make_config
from banqi_training.variant import get_variant


def _print_config_warnings(config: "Config") -> None:
    """打印配置冲突/异常告警（不阻断运行）。"""
    if getattr(config, "USE_RULE_BASED_SELF_PLAY", False) and (
            config.TRAIN_MODE or "selfplay") == "selfplay":
        print("⚠️ [配置] USE_RULE_BASED_SELF_PLAY=True，将注入规则启发式")
    if config.ARCHIVE_ENABLED and config.MONGO_URI in (None, "", "mongodb://localhost:27017"):
        print("⚠️ [配置] ARCHIVE_ENABLED=True 但 MONGO_URI 为默认/空，归档可能失败")
    if config.TRAIN_BATCH <= 0 or config.PREDICT_BATCH <= 0:
        print("⚠️ [配置] BATCH 尺寸非正，将导致训练/推理异常")


def make_config_from_args(variant_id: str, args) -> "Config":
    """根据 argparse.Namespace 构造 Config（含命令行覆盖）。"""
    config = make_config(variant_id)
    config._variant = get_variant(variant_id)

    # 命令行覆盖（仅设置非 None 字段）
    if getattr(args, "train_mode", None):
        config.TRAIN_MODE = args.train_mode
    if getattr(args, "mcts_sims", None) is not None:
        config.MCTS_SIMS = args.mcts_sims
    if getattr(args, "games_per_iter", None) is not None:
        config.GAMES_PER_ITER = args.games_per_iter
    if getattr(args, "self_play_processes", None) is not None:
        config.SELF_PLAY_PROCESSES = args.self_play_processes
    if getattr(args, "train_steps", None) is not None:
        config.TRAIN_STEPS = args.train_steps
    if getattr(args, "seed", None) is not None:
        config.SEED = args.seed
    if getattr(args, "models_dir", None):
        os.environ.setdefault("BANQI_MODELS_DIR", args.models_dir)
    if getattr(args, "no_benchmark", False):
        config.BENCHMARK_ENABLED = False
    if getattr(args, "no_tensorboard", False):
        config.TENSORBOARD_ENABLED = False
    if getattr(args, "no_monitor", False):
        config.MONITOR_ENABLED = False

    _print_config_warnings(config)
    return config
