"""banqi/tb_logger.py — TensorBoard 训练日志封装（共享实现）

单进程内共享一个全局 SummaryWriter，供训练 / 自对弈 / 系统监控各线程写入，
避免各自创建 writer 产生重复 event 文件。

用法（训练入口调用一次初始化，各线程随时写入）：
    from banqi_training.tb_logger import init_summary_writer, add_scalar, close_summary_writer

    init_summary_writer(log_dir="runs/20260814-100000")  # 启动时
    add_scalar("train/loss", 0.12, step=100)             # 任意线程（线程安全）
    close_summary_writer()                               # 结束时

依赖：tensorboard（pyproject.toml 已有）。未安装或初始化失败时，
所有调用自动降级为 no-op，不影响训练。
"""

from __future__ import annotations

import os
import threading
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    HAS_TENSORBOARD = False

_writer: Optional[SummaryWriter] = None
_lock = threading.Lock()


def init_summary_writer(
    log_dir: str = "runs",
    enabled: bool = True,
    flush_secs: int = 30,
) -> bool:
    """
    初始化全局 SummaryWriter（幂等，保留首次创建的 writer）。

    参数:
        log_dir:   事件文件目录
        enabled:   总开关（对应 config.TENSORBOARD_ENABLED）
        flush_secs:事件刷新到磁盘的间隔秒数

    返回是否成功启用（True 表示后续 add_scalar 会真正写入）。
    """
    global _writer
    if not enabled or not HAS_TENSORBOARD:
        return False
    if _writer is not None:
        return True
    if not os.path.isabs(log_dir):
        py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(py_dir, log_dir)
    try:
        _writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        return True
    except Exception as exc:  # noqa: BLE001 - 初始化失败原因多样
        print(f"[TB] ⚠️ TensorBoard 初始化失败，日志记录已禁用: {exc}")
        _writer = None
        return False


def _safe_call(fn_name: str, *args, **kwargs):
    """线程安全地调用 writer 方法；writer 未初始化或调用失败时静默忽略。"""
    if _writer is None:
        return None
    with _lock:
        try:
            return getattr(_writer, fn_name)(*args, **kwargs)
        except Exception:  # noqa: BLE001 - 避免监控线程因日志异常而崩溃
            return None


def add_scalar(
    tag: str,
    scalar_value: float,
    global_step: Optional[int] = None,
    walltime: Optional[float] = None,
    **kwargs,
) -> None:
    """
    线程安全的标量写入；writer 未初始化时 no-op，写失败静默忽略。
    兼容 torch SummaryWriter 的 step= 关键字（作为 global_step 的别名）。
    """
    if "step" in kwargs:
        global_step = kwargs["step"]
    _safe_call("add_scalar", tag, scalar_value, global_step, walltime)


def add_histogram(
    tag: str,
    values,
    global_step: Optional[int] = None,
    **kwargs,
) -> None:
    """
    线程安全的直方图写入（与 add_scalar 相同降级策略）。
    兼容 torch SummaryWriter 的 step= 关键字。
    values 为 numpy 数组 / torch tensor 均可。
    """
    if "step" in kwargs:
        global_step = kwargs["step"]
    _safe_call("add_histogram", tag, values, global_step)


def add_text(
    tag: str,
    text: str,
    global_step: Optional[int] = None,
    **kwargs,
) -> None:
    """线程安全的文本写入（运行标注 / 超参数说明等）。"""
    if "step" in kwargs:
        global_step = kwargs["step"]
    _safe_call("add_text", tag, text, global_step)


def add_hparams(
    hparam_dict: dict,
    metric_dict: Optional[dict] = None,
    run_name: str = ".",
    **kwargs,
) -> None:
    """写入超参数（TensorBoard HParams 面板）。默认 run_name="." 避免创建额外时间戳子目录。"""
    _safe_call("add_hparams", hparam_dict, metric_dict or {}, run_name=run_name, **kwargs)


def close_summary_writer() -> None:
    """关闭并 flush 全局 writer（幂等）。"""
    global _writer
    if _writer is None:
        return
    with _lock:
        try:
            _writer.close()
        except Exception:  # noqa: BLE001
            pass
        _writer = None
