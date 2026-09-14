"""banqi_training/tools/random_onnx.py — 随机参数模型的 ONNX 导出

用途：不训练即得到与真实模型「同结构、同导出契约」的随机权重模型（同一 `BanqiNet`
+ 同一 `checkpoint.export_onnx`：输入 board/scalars，输出 policy_logits/value[/health]），
供 banqi-tauri 的 MctsOnnx 对手做「搜索规模 vs 强度」基线测试。

用法：
  python -m banqi_training.tools.random_onnx
  python -m banqi_training.tools.random_onnx --seed 1 --no-health
  python -m banqi_training.tools.random_onnx --variant 4x4 --out /tmp/4x4_random.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from banqi_training.checkpoint import export_onnx
from banqi_training.constants import build_constants
from banqi_training.nn_model import BanqiNet, count_params
from banqi_training.variant import get_variant


def random_onnx(
    variant_id: str,
    out_path: str | None,
    seed: int,
    enable_health: bool,
) -> str:
    """按 variant 结构生成随机权重模型并导出 ONNX，返回导出路径。"""
    variant = get_variant(variant_id)
    c = build_constants(variant)

    torch.manual_seed(seed)
    model = BanqiNet(variant, enable_health=enable_health)
    model.eval()

    if out_path is None:
        name = "random_health.onnx" if enable_health else "random.onnx"
        out_path = os.path.join(variant.checkpoints_dir, name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(
        f"[random_onnx] variant={variant.id} seed={seed} health={enable_health} "
        f"params={count_params(model)} "
        f"board=({c.TOTAL_INPUT_CHANNELS},{c.BOARD_ROWS},{c.BOARD_COLS}) "
        f"scalars={c.SCALAR_FEATURE_COUNT} action={c.ACTION_SPACE_SIZE}"
        + (f" health_bins={c.HEALTH_DIFF_BINS}" if enable_health else "")
    )

    if not export_onnx(model, out_path, variant, torch.device("cpu")):
        sys.exit(f"[random_onnx] ❌ ONNX 导出失败: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出随机参数模型的 ONNX（供 banqi-tauri MctsOnnx 对手测试）"
    )
    parser.add_argument("--variant", choices=["4x8", "4x4", "4x2"], default="4x2")
    parser.add_argument(
        "--out",
        default=None,
        help="输出路径（默认 outputs/<variant>/checkpoints/random[_health].onnx）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机初始化种子（默认 0）")
    parser.add_argument(
        "--no-health",
        dest="enable_health",
        action="store_false",
        help="不导出血量差异第三头（默认导出，与 4x2 训练配置的 HEALTH_VALUE_HEAD_ENABLED 一致）",
    )
    args = parser.parse_args()

    path = random_onnx(args.variant, args.out, args.seed, args.enable_health)
    print(f"[random_onnx] ✅ 已生成: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
