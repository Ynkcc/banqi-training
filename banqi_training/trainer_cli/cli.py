"""banqi_training/trainer_cli/cli.py — 命令行解析与程序化入口（分布式 Trainer）。"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config_resolver import make_config_from_args
from .runners import main as _run_main


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m banqi_training.trainer_cli",
        description="Banqi 分布式训练 CLI（经 banqi-scheduler 调度）",
    )
    p.add_argument("variant", nargs="?", default="4x8", help="变体 id（默认 4x8，实际以调度器 GetInfo 下发为准）")
    p.add_argument("--train-mode", default="distributed",
                   choices=["distributed"],
                   help="本仓库仅支持分布式训练")
    p.add_argument("--mcts-sims", type=int, default=None, help="MCTS 模拟次数")
    p.add_argument("--games-per-iter", type=int, default=None, help="每轮对局数")
    p.add_argument("--self-play-processes", type=int, default=None, help="自对弈子进程数")
    p.add_argument("--train-steps", type=int, default=None, help="训练步数预算")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--models-dir", default=None, help="模型根目录（覆盖 BANQI_MODELS_DIR）")
    p.add_argument("--no-benchmark", action="store_true", help="禁用 benchmark")
    p.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard")
    p.add_argument("--no-monitor", action="store_true", help="禁用系统资源监控")
    return p.parse_args(argv)


def programmatic_entry(argv: Optional[List[str]] = None):
    """供 `python -m banqi_training.trainer_cli` 调用的入口。"""
    args = parse_args(argv)
    variant_id = args.variant
    # 若命令行指定了 train-mode，则直接用 make_config 覆盖后运行；
    # 否则走 runners.main 的标准 config.TRAIN_MODE 分派。
    # 命令行覆盖全部生效：make_config 按 variant_id 缓存，
    # 此处先通过 make_config_from_args 把覆盖写入缓存对象，
    # runners.main 内部的 make_config 拿到的即是已覆盖的同一实例。
    make_config_from_args(variant_id, args)
    _run_main(variant_id)


if __name__ == "__main__":
    programmatic_entry()
