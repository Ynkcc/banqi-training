"""banqi/memory_guard.py — 跨平台内存看门狗守护线程（psutil）

在训练 / 基准等长期运行进程中，后台周期性检查：

  1. 当前进程 RSS 是否超过阈值（应对单个进程内存膨胀 / 泄漏）
  2. 系统全局剩余内存是否跌破安全线（应对多 worker 并发膨胀导致整机 OOM）

一旦超限，直接 os._exit(1) 紧急终止，避免长时间卡死浪费算力 / 拖垮整机。

依赖（pyproject.toml）：psutil>=5.9.0

用法：
    from banqi_training.memory_guard import start_memory_guard
    start_memory_guard(max_process_gb=24.0, min_sys_free_gb=1.5)
"""

from __future__ import annotations

import os
import threading
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False


def start_memory_guard(
    max_process_gb: float = 24.0,
    min_sys_free_gb: float = 1.5,
    check_interval: float = 1.0,
) -> None:
    """启动跨平台内存守护线程 (Windows / Linux 通用)。

    守护线程为 daemon，进程退出自动结束；不阻塞调用方。

    :param max_process_gb: 当前进程最大允许占用的内存 (GB)
    :param min_sys_free_gb: 系统必须保留的最小空闲内存 (GB)
    :param check_interval: 检查周期 (秒)
    """
    if not HAS_PSUTIL:  # pragma: no cover
        print("[Memory Guard] ⚠️ 未安装 psutil，内存守护已跳过（pip install psutil）")
        return

    def _monitor() -> None:
        process = psutil.Process(os.getpid())
        while True:
            # 1. 检查当前进程占用
            proc_mem_gb = process.memory_info().rss / (1024**3)
            if proc_mem_gb > max_process_gb:
                print(
                    f"\n[Memory Guard] 进程内存已达 {proc_mem_gb:.2f}GB (限制:"
                    f" {max_process_gb}GB)，主动终止以防卡死！"
                )
                os._exit(1)

            # 2. 检查系统全局剩余内存 (应对多 worker 并发膨胀)
            sys_avail_gb = psutil.virtual_memory().available / (1024**3)
            if sys_avail_gb < min_sys_free_gb:
                print(
                    f"\n[Memory Guard] 系统剩余内存仅剩 {sys_avail_gb:.2f}GB (安全线:"
                    f" {min_sys_free_gb}GB)，紧急终止！"
                )
                os._exit(1)

            time.sleep(check_interval)

    guard = threading.Thread(target=_monitor, daemon=True, name="MemoryGuard")
    guard.start()
    print(
        f"[Memory Guard] 已启动：进程上限 {max_process_gb:.1f}GB，"
        f"系统空闲安全线 {min_sys_free_gb:.1f}GB，检查周期 {check_interval:.1f}s"
    )
