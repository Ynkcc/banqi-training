"""banqi/rust_bridge.py — Rust PyO3 扩展（banqi_4x8）的唯一导入门面。

所有对 Rust 绑定的访问必须经由此模块，收益：
1. 缺失扩展时的统一报错信息（提示 maturin develop）；
2. 契约面集中可盘点：Rust 侧接口变更时只需改这一个文件；
3. 便于测试 mock（monkeypatch banqi_training.rust_bridge.run_native_match 即可，
   无需逐文件 patch banqi_4x8）。

当前契约面（与 src/bridge/python/ 对应）：
    variant_dims                — 变体维度核对（constants.py）
    SelfPlayConfig              — 自对弈配置（selfplay/rule_teacher）
    run_native_match            — Rust 原生对战/自对弈（eval/selfplay/worker）
    run_python_match            — Python 推理自对弈（selfplay/worker）
    get_action_symmetry_table   — 动作对称置换表（training/augment）
    transform_board / transform_policy / transform_action — 对称增强
"""

from __future__ import annotations

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

# ---- 变体维度 ----
variant_dims = banqi_4x8.variant_dims

# ---- 自对弈 ----
SelfPlayConfig = banqi_4x8.SelfPlayConfig
run_native_match = banqi_4x8.run_native_match
run_python_match = banqi_4x8.run_python_match

# ---- Expectimax + NNUE 自对弈（NNUE 蒸馏回环 sidecar） ----
if hasattr(banqi_4x8, "run_expectimax_self_play"):
    run_expectimax_self_play = banqi_4x8.run_expectimax_self_play
else:  # pragma: no cover — 旧版扩展未编译 expectimax 入口
    run_expectimax_self_play = None

# ---- 对称增强 ----
get_action_symmetry_table = banqi_4x8.get_action_symmetry_table
transform_board = banqi_4x8.transform_board
transform_policy = banqi_4x8.transform_policy
transform_action = banqi_4x8.transform_action

__all__ = [
    "variant_dims",
    "SelfPlayConfig",
    "run_native_match",
    "run_python_match",
    "run_expectimax_self_play",
    "get_action_symmetry_table",
    "transform_board",
    "transform_policy",
    "transform_action",
]
