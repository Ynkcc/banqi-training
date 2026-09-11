"""banqi_training/trainer_cli/runners/ — 训练模式分派与运行编排（包）。

本仓库仅支持分布式形态：变体由调度器下发（GetInfo），
episode 经 scheduler ListEpisodes 拉取，模型经预签名上传 + RegisterNetwork 登记。

共享基础设施（可选依赖探测 / 日志落盘 / TB 元信息 / 队列计数 / 变体维度缓存）
统一在 runners/context.py。
"""

from __future__ import annotations

from .context import build_const, log_meta_tb, setup_variant_logging
from .distributed import run_distributed

__all__ = [
    "main",
    "build_const",
    "setup_variant_logging",
    "log_meta_tb",
]


def main(variant_id: str) -> None:
    """统一训练入口：变体由调度器下发，忽略命令行位置参数。"""
    from banqi_training.infra import scheduler_variant

    server_variant = scheduler_variant()
    if server_variant != variant_id:
        print(f"[distributed] 变体由调度器下发: {server_variant}（命令行传入 {variant_id} 被忽略）")
    run_distributed(server_variant)
