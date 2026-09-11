"""banqi/checkpoint.py — 通用 checkpoint 保存 / 恢复（含 variant 元数据）

面向新格式：保存时写入 variant 关键维度到 model_config；
恢复时校验维度匹配（不兼容旧文件，见项目决策「不保留旧版本兼容」）。
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from banqi_training.constants import Constants, build_constants
from banqi_training.variant import Variant


def _model_config(c: Constants) -> dict:
    return {
        "variant": c.variant.id,
        "input_channels": c.TOTAL_INPUT_CHANNELS,
        "board_rows": c.BOARD_ROWS,
        "board_cols": c.BOARD_COLS,
        "scalar_features": c.SCALAR_FEATURE_COUNT,
        "action_space": c.ACTION_SPACE_SIZE,
        "health_diff_bins": c.HEALTH_DIFF_BINS,
    }


def _default_onnx_path(model_path: str) -> str:
    """由 TorchScript 模型路径推导默认 ONNX 路径（同名 .onnx）。"""
    base, _ = os.path.splitext(model_path)
    return base + ".onnx"


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    model_path: str,
    state_dict_path: str,
    device: torch.device,
    variant: Variant,
    onnx_path: Optional[str] = None,
) -> None:
    """保存完整训练状态：.pth（可恢复）+ .pt（TorchScript 供推理）+ .onnx（可选）。

    onnx_path 为 None 时由 model_path 推导（同名 .onnx）；传入空串则跳过 ONNX 导出。
    先写临时文件再原子 replace，避免进程中断留下半成品。
    """
    c = build_constants(variant)
    pt_temp = model_path + ".tmp"
    pth_temp = state_dict_path + ".tmp"
    trace_model = getattr(model, "_orig_mod", model)
    try:
        model.eval()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": _model_config(c),
            },
            pth_temp,
        )
        os.replace(pth_temp, state_dict_path)

        with torch.inference_mode():
            example_board = torch.randn(
                1, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS, device=device
            )
            example_scalars = torch.randn(1, c.SCALAR_FEATURE_COUNT, device=device)
            traced = torch.jit.trace(trace_model, (example_board, example_scalars))
            traced.save(pt_temp)
        os.replace(pt_temp, model_path)

        # ONNX 导出（供 RustOnnxCollector / MctsOnnx 推理）
        target_onnx = _default_onnx_path(model_path) if onnx_path is None else onnx_path
        if target_onnx:
            export_onnx(model, target_onnx, variant, device)

        print(f"[checkpoint] ✅ 保存成功: {state_dict_path} + {model_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[checkpoint] ❌ 保存失败: {exc}")
        for tmp in (pt_temp, pth_temp):
            if os.path.exists(tmp):
                os.remove(tmp)


def export_torchscript(
    model,
    pt_path: str,
    variant: Variant,
    device: torch.device,
) -> bool:
    """导出模型为 TorchScript .pt 文件（供 Rust tch-rs / MctsDL 推理）。"""
    c = build_constants(variant)
    trace_model = getattr(model, "_orig_mod", model)
    pt_temp = pt_path + ".tmp"
    try:
        model.eval()
        with torch.inference_mode():
            example_board = torch.randn(
                1, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS, device=device
            )
            example_scalars = torch.randn(1, c.SCALAR_FEATURE_COUNT, device=device)
            traced = torch.jit.trace(trace_model, (example_board, example_scalars))
            traced.save(pt_temp)
        os.replace(pt_temp, pt_path)
        print(f"[checkpoint] ✅ TorchScript 已导出: {pt_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[checkpoint] ❌ TorchScript 导出失败: {exc}")
        if os.path.exists(pt_temp):
            os.remove(pt_temp)
        return False



def export_onnx(
    model,
    onnx_path: str,
    variant: Variant,
    device: torch.device,
) -> bool:
    """导出模型为 ONNX（供 RustOnnxCollector / MctsOnnx / onnxruntime 推理）。

    输入名固定为 "board" / "scalars"，输出名固定为 "policy_logits" / "value"
    （与 Rust 侧 src/onnx/mod.rs 的契约一致），batch 维度动态。
    启用血量差异头时追加第三输出 "health"（[B, K]，K=HEALTH_DIFF_BINS）。
    失败时打印原因并返回 False（不抛异常，避免中断主训练流程）。
    """
    c = build_constants(variant)
    trace_model = getattr(model, "_orig_mod", model)
    onnx_temp = onnx_path + ".tmp"
    health_enabled = bool(getattr(trace_model, "enable_health", False))
    output_names = ["policy_logits", "value"]
    dynamic_axes = {
        "board": {0: "batch"},
        "scalars": {0: "batch"},
        "policy_logits": {0: "batch"},
        "value": {0: "batch"},
    }
    if health_enabled:
        output_names.append("health")
        dynamic_axes["health"] = {0: "batch"}
    try:
        model.eval()
        with torch.inference_mode():
            example_board = torch.randn(
                1, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS, device=device
            )
            example_scalars = torch.randn(1, c.SCALAR_FEATURE_COUNT, device=device)
            torch.onnx.export(
                trace_model,
                (example_board, example_scalars),
                onnx_temp,
                # dynamo=False：使用传统 TorchScript 导出器（无需安装 onnxscript）；
                # 与 torch.jit.trace 同样只跟踪前向计算图，BatchNorm 已在 eval 模式
                # 下常量折叠，输出与 .pt TorchScript 完全等价。
                dynamo=False,
                input_names=["board", "scalars"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=13,
                do_constant_folding=True,
            )
        os.replace(onnx_temp, onnx_path)
        print(f"[checkpoint] ✅ ONNX 已导出: {onnx_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[checkpoint] ❌ ONNX 导出失败: {exc}")
        if os.path.exists(onnx_temp):
            os.remove(onnx_temp)
        return False


def _export_worker_proc(pipe_conn, pt_path: Optional[str], onnx_path: Optional[str], variant_id: str, device_str: str, enable_health: bool) -> None:
    """子进程独立导出入口：通过 Pipe 接收共享内存句柄 (share_memory_)。

    torch.jit.trace 会在 PyTorch C++ 内部生成极难释放的 CompilationUnit 缓存。
    在独立子进程中导出，可以在导出完成后由 OS 物理清空子进程堆内存，彻底消除主进程泄露。
    """
    model_state = pipe_conn.recv()
    from banqi_training.nn_model import BanqiNet
    from banqi_training.variant import get_variant
    v = get_variant(variant_id)
    dev = torch.device(device_str)
    model = BanqiNet(v, enable_health=enable_health).to(dev)
    model.load_state_dict(model_state)
    if pt_path:
        export_torchscript(model, pt_path, v, dev)
    if onnx_path:
        export_onnx(model, onnx_path, v, dev)


def export_model_isolated(
    model: torch.nn.Module,
    pt_path: Optional[str],
    onnx_path: Optional[str],
    variant: Variant,
    device: torch.device,
) -> None:
    """使用 spawn 独立子进程 + Pipe 共享内存句柄导出 TorchScript/ONNX，零拷贝隔离内存泄露。"""
    import torch.multiprocessing as tmp
    raw_model = getattr(model, "_orig_mod", model)
    enable_health = bool(getattr(raw_model, "enable_health", False))
    # 将模型 state_dict 转为 CPU 共享内存 Tensor，只跨进程发送句柄 (Zero-copy)
    state_dict = {
        k: v.detach().cpu().clone().share_memory_()
        for k, v in raw_model.state_dict().items()
    }
    parent_conn, child_conn = tmp.Pipe()
    ctx = tmp.get_context("spawn")
    p = ctx.Process(
        target=_export_worker_proc,
        args=(child_conn, pt_path, onnx_path, variant.id, str(device), enable_health),
    )
    p.start()
    parent_conn.send(state_dict)
    p.join(timeout=60)
    if p.is_alive():
        p.terminate()
        raise RuntimeError(f"模型隔离导出子进程超时 (60s): {pt_path} / {onnx_path}")
    if p.exitcode != 0:
        raise RuntimeError(f"模型隔离导出子进程异常退出 (exitcode={p.exitcode})")




def load_checkpoint(
    model,
    optimizer,
    scheduler,
    state_dict_path: str,
    device: torch.device,
    variant: Variant,
) -> bool:
    """从 .pth 恢复完整训练状态；无 checkpoint 时返回 False（全新模型）。

    任何加载/维度校验/状态恢复失败都直接抛出，绝不静默降级为全新初始化：
    静默重置 Optimizer/Scheduler 会掩盖文件损坏或维度变化，导致训练在不知不觉中
    收敛震荡甚至崩溃。项目决策「不保留旧版本兼容」，故不做 .pt 权重回退。
    """
    c = build_constants(variant)
    if not os.path.exists(state_dict_path):
        print("[checkpoint] 📝 初始化全新模型（无 checkpoint）")
        return False
    checkpoint = torch.load(state_dict_path, map_location=device)
    _check_dimensions(checkpoint.get("model_config"), c)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"[checkpoint] ✅ 从 {state_dict_path} 恢复完整训练状态")
    return True


def _check_dimensions(cfg: Optional[dict], c: Constants) -> None:
    """校验 checkpoint 的维度与当前 variant 一致；不一致抛错（明确失败）。"""
    if not cfg:
        return
    expect = {
        "input_channels": c.TOTAL_INPUT_CHANNELS,
        "board_rows": c.BOARD_ROWS,
        "board_cols": c.BOARD_COLS,
        "scalar_features": c.SCALAR_FEATURE_COUNT,
        "action_space": c.ACTION_SPACE_SIZE,
        "health_diff_bins": c.HEALTH_DIFF_BINS,
    }
    for k, val in expect.items():
        got = cfg.get(k)
        if got is not None and got != val:
            raise ValueError(
                f"checkpoint 维度 {k}={got} 与当前变体({c.variant.id}) 预期 {val} 不一致"
            )
