"""python/banqi/tools/export_ckpt.py — 命令行导出工具

将训练产生的 .ckpt（或 .pth）权重导出为 TorchScript .pt 和 .onnx 模型。
用法：
  python -m banqi_training.tools.export_ckpt python/outputs/4x2/checkpoints/last.ckpt [--variant 4x2]
"""

from __future__ import annotations

import argparse
import os
import sys
import torch

from banqi_training.variant import get_variant
from banqi_training.nn_model import BanqiNet
from banqi_training.checkpoint import export_torchscript, export_onnx


def infer_variant_from_path(path: str) -> str:
    """根据文件路径自动推导变体 ID（4x2, 4x4, 4x8）。"""
    lower = path.lower()
    if "4x2" in lower or "mini" in lower:
        return "4x2"
    if "4x4" in lower:
        return "4x4"
    return "4x8"


def export_checkpoint_file(ckpt_path: str, variant_id: str | None = None) -> bool:
    if not os.path.exists(ckpt_path):
        print(f"❌ 文件不存在: {ckpt_path}")
        return False

    v_id = variant_id or infer_variant_from_path(ckpt_path)
    variant = get_variant(v_id)
    print(f"📦 导出目标: {ckpt_path} (变体: {variant.id})")

    device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")

    # 由权重 key / 形状检测结构开关，保证与 checkpoint 匹配：
    #   - 血量差异头：存在 health_* 权重；
    #   - 分布化价值头：value_fc2 输出维度 > 1（标量价值头为 1）。
    enable_health = any(k.startswith("health_") for k in state_dict)
    value_out_dim = int(state_dict["value_fc2.weight"].shape[0])
    enable_value_dist = value_out_dim > 1
    model = BanqiNet(
        variant,
        enable_health=enable_health,
        enable_value_dist=enable_value_dist,
        value_dist_bins=value_out_dim if enable_value_dist else 65,
    )
    model.load_state_dict(state_dict)
    model.eval()

    base_dir = os.path.dirname(ckpt_path)
    base_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    pt_path = os.path.join(base_dir, f"{base_name}.pt")
    onnx_path = os.path.join(base_dir, f"{base_name}.onnx")

    ok_pt = export_torchscript(model, pt_path, variant, device)
    ok_onnx = export_onnx(model, onnx_path, variant, device)

    if ok_pt or ok_onnx:
        print(f"🎉 导出完成！目标格式可被 banqi-tauri 加载。")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="导出 PyTorch .ckpt 到 .pt 与 .onnx")
    parser.add_argument("ckpt_path", help="输入 .ckpt 或 .pth 文件路径")
    parser.add_argument("--variant", choices=["4x8", "4x4", "4x2"], help="变体 ID（默认从路径自动推导）")
    args = parser.parse_args()

    success = export_checkpoint_file(args.ckpt_path, args.variant)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
