"""
banqi/system_monitor.py — 训练过程系统资源监控（psutil + pynvml）

监控 CPU / 内存 / GPU 用量，两种使用方式：

    1. 集成模式（推荐）：随训练闭环自动启动监控线程
       由 config.MONITOR_* 控制开关与采样间隔

    2. 独立监控模式：单独运行，监控任何进程运行期间的整机资源
        python python/banqi/system_monitor.py [--interval 10] [--count 0]
                                             [--csv monitor.csv] [--per-core]

监控内容：
  - CPU  : 整机总使用率 + 当前进程使用率（--per-core 额外显示每核）
  - 内存 : 系统 total / used / percent + 当前进程 RSS / VMS
  - GPU  : 每张卡利用率 / 显存占用 / 温度 / 功耗（pynvml，无驱动时自动降级跳过）
  - CSV  : --csv 将每次采样落盘，便于事后绘图分析

依赖（pyproject.toml）：
    psutil>=5.9.0
    nvidia-ml-py>=12.0.0   # import 名仍是 pynvml（兼容旧 pynvml 包）
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from typing import Dict, List, Optional

import psutil

# 兼容：官方新包 nvidia-ml-py 与旧包 pynvml 的 import 名一致
try:
    import pynvml  # type: ignore[import-untyped]
    HAS_NVML = True
except ImportError:  # pragma: no cover
    HAS_NVML = False


def _fmt_gb(nbytes: int | float) -> str:
    """字节数格式化为 GiB 字符串。"""
    return f"{nbytes / 2**30:.2f}"


# ============================================================================
# GPU 监控（pynvml 封装）
# ============================================================================

class GpuMonitor:
    """pynvml 惰性初始化封装；驱动缺失 / 采样失败时自动降级不抛异常。"""

    def __init__(self) -> None:
        self._ready = False
        self._device_count = 0
        self._names: List[str] = []
        if not HAS_NVML:
            print("[Monitor] ⚠️ 未安装 pynvml，GPU 监控已禁用（pip install nvidia-ml-py）")
            return
        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._names = [
                pynvml.nvmlDeviceGetName(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).decode(errors="replace")
                for i in range(self._device_count)
            ]
            self._ready = True
        except Exception as exc:  # noqa: BLE001 - NVML 失败原因多样
            print(f"[Monitor] ⚠️ NVML 初始化失败，GPU 监控已禁用: {exc}")

    @property
    def enabled(self) -> bool:
        return self._ready

    def sample(self) -> List[Dict]:
        """返回每张 GPU 的监控快照 dict 列表；不可用时返回 []。"""
        if not self._ready:
            return []
        results: List[Dict] = []
        for i in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                except Exception:  # noqa: BLE001 - 部分设备不支持功耗查询
                    power_mw = None
                results.append({
                    "index": i,
                    "name": self._names[i] if i < len(self._names) else f"GPU{i}",
                    "gpu_percent": int(util.gpu),
                    "mem_percent": int(util.memory),
                    "mem_used": int(mem.used),
                    "mem_total": int(mem.total),
                    "temp": int(temp),
                    "power_w": None if power_mw is None else power_mw / 1000.0,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"[Monitor] ⚠️ GPU{i} 采样失败: {exc}")
                results.append({"index": i, "name": f"GPU{i}", "error": str(exc)})
        return results

    def close(self) -> None:
        if self._ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._ready = False


# ============================================================================
# 系统监控线程
# ============================================================================

class SystemMonitor(threading.Thread):
    """
    后台守护线程：周期性采样并打印 CPU / 内存 / GPU 用量。

    参数:
        interval:     采样间隔（秒），最小 1s
        max_samples:  最大采样次数，0 表示无限（直到 stop_event 置位）
        show_per_core: 额外显示每核 CPU 使用率
        csv_path:     采样数据 CSV 落盘路径（None 则不落盘）
        log_to_tb:    是否把采样写入 TensorBoard（sys/* 标量）
        stop_event:   共享停止信号（threading.Event），set() 后线程在下一分片（≤0.1s）退出
    """

    def __init__(
        self,
        interval: float = 10.0,
        max_samples: int = 0,
        show_per_core: bool = False,
        csv_path: Optional[str] = None,
        log_to_tb: bool = False,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__(name="SystemMonitor", daemon=True)
        self.interval = max(1.0, float(interval))
        self.max_samples = max(0, int(max_samples))
        self.show_per_core = show_per_core
        self.csv_path = csv_path
        self.log_to_tb = log_to_tb
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._proc = psutil.Process()
        self._gpu = GpuMonitor()
        self._csv_fp = None
        self._csv_writer = None
        # 预热 CPU 采样（psutil 首次 cpu_percent() 返回 0.0，无实际意义）
        _ = psutil.cpu_percent(None)
        _ = self._proc.cpu_percent(None)

    # ---- 采样 ----

    def sample(self) -> Dict:
        """采集一次系统快照。"""
        cpu_percent = psutil.cpu_percent(None)
        mem = psutil.virtual_memory()
        proc_mem = self._proc.memory_info()
        try:
            proc_cpu = self._proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_cpu = None
        return {
            "ts": time.time(),
            "cpu_percent": cpu_percent,
            "per_core": psutil.cpu_percent(percpu=True) if self.show_per_core else None,
            "proc_cpu_percent": proc_cpu,
            "mem_total": mem.total,
            "mem_used": mem.used,
            "mem_percent": mem.percent,
            "proc_rss": proc_mem.rss,
            "proc_vms": proc_mem.vms,
            "gpus": self._gpu.sample(),
        }

    # ---- 输出 ----

    def _format(self, s: Dict) -> str:
        parts: List[str] = []
        cpu = s["cpu_percent"]
        proc_cpu = s["proc_cpu_percent"]
        cpu_str = f"{cpu:.1f}%" if cpu is not None else "N/A"
        proc_cpu_str = f"{proc_cpu:.1f}%" if proc_cpu is not None else "N/A"
        parts.append(f"CPU {cpu_str} (本进程 {proc_cpu_str})")
        parts.append(
            f"内存 {_fmt_gb(s['mem_used'])}/{_fmt_gb(s['mem_total'])} GB "
            f"({s['mem_percent']:.1f}%) 本进程 {_fmt_gb(s['proc_rss'])} GB"
        )
        if self.show_per_core and s["per_core"]:
            core_str = "/".join(f"{c:.0f}" for c in s["per_core"])
            parts.append(f"每核 [{core_str}]")
        for g in s["gpus"]:
            if "error" in g:
                parts.append(f"{g['name']}: 采样失败")
                continue
            power = f" | {g['power_w']:.0f}W" if g["power_w"] is not None else ""
            parts.append(
                f"{g['name']}: 利用 {g['gpu_percent']}% | "
                f"显存 {_fmt_gb(g['mem_used'])}/{_fmt_gb(g['mem_total'])} GB "
                f"({g['mem_percent']}%) | {g['temp']}°C{power}"
            )
        return " | ".join(parts)

    def _write_csv(self, s: Dict) -> None:
        if not self.csv_path:
            return
        if self._csv_writer is None:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True
            )
            self._csv_fp = open(self.csv_path, "w", newline="", encoding="utf-8")
            fieldnames = [
                "timestamp", "cpu_percent", "proc_cpu_percent",
                "mem_percent", "mem_used_gb", "proc_rss_gb",
            ]
            for g in s["gpus"]:
                if "error" not in g:
                    fieldnames += [
                        f"gpu{g['index']}_util",
                        f"gpu{g['index']}_mem_percent",
                        f"gpu{g['index']}_mem_used_gb",
                        f"gpu{g['index']}_temp",
                    ]
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        row: Dict[str, object] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["ts"])),
            "cpu_percent": f"{s['cpu_percent']:.1f}",
            "proc_cpu_percent": f"{s['proc_cpu_percent']:.1f}",
            "mem_percent": f"{s['mem_percent']:.1f}",
            "mem_used_gb": _fmt_gb(s["mem_used"]),
            "proc_rss_gb": _fmt_gb(s["proc_rss"]),
        }
        for g in s["gpus"]:
            if "error" not in g:
                row[f"gpu{g['index']}_util"] = g["gpu_percent"]
                row[f"gpu{g['index']}_mem_percent"] = g["mem_percent"]
                row[f"gpu{g['index']}_mem_used_gb"] = _fmt_gb(g["mem_used"])
                row[f"gpu{g['index']}_temp"] = g["temp"]
        self._csv_writer.writerow(row)
        self._csv_fp.flush()

    def _log_to_tb(self, s: Dict, step: int) -> None:
        """把本次采样写入 TensorBoard（tag 前缀 sys/，x 轴为采样序号）。"""
        try:
            from banqi_training.tb_logger import add_scalar
        except ImportError:  # pragma: no cover
            return
        add_scalar("sys/cpu_percent", s["cpu_percent"], step)
        if s["proc_cpu_percent"] is not None:
            add_scalar("sys/proc_cpu_percent", s["proc_cpu_percent"], step)
        add_scalar("sys/mem_percent", s["mem_percent"], step)
        add_scalar("sys/mem_used_gb", s["mem_used"] / 2**30, step)
        add_scalar("sys/proc_rss_gb", s["proc_rss"] / 2**30, step)
        for g in s["gpus"]:
            if "error" not in g:
                add_scalar(f"sys/gpu{g['index']}_util", g["gpu_percent"], step)
                add_scalar(f"sys/gpu{g['index']}_mem_percent", g["mem_percent"], step)
                add_scalar(f"sys/gpu{g['index']}_temp", g["temp"], step)

    # ---- 线程主体 ----

    def run(self) -> None:
        samples_done = 0
        try:
            while not self.stop_event.is_set():
                if self.max_samples and samples_done >= self.max_samples:
                    break
                s = self.sample()
                self._write_csv(s)
                if self.log_to_tb:
                    self._log_to_tb(s, samples_done)
                print(f"[Monitor] {time.strftime('%H:%M:%S')} | {self._format(s)}")
                samples_done += 1
                # 分片等待，快速响应 stop_event
                deadline = time.time() + self.interval
                while not self.stop_event.is_set() and time.time() < deadline:
                    time.sleep(0.1)
        finally:
            self.stop()

    def stop(self) -> None:
        """释放资源：关闭 CSV 文件与 NVML。"""
        if self._csv_fp is not None:
            self._csv_fp.close()
            self._csv_fp = None
        self._gpu.close()


# ============================================================================
# 独立监控入口
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练过程系统资源监控（psutil + pynvml）"
    )
    parser.add_argument("--interval", type=float, default=10.0,
                        help="采样间隔秒（最小 1s，默认 10）")
    parser.add_argument("--count", type=int, default=0,
                        help="采样次数，0=无限（默认 0）")
    parser.add_argument("--csv", type=str, default=None,
                        help="采样数据 CSV 落盘路径（默认不落盘）")
    parser.add_argument("--per-core", action="store_true",
                        help="显示每核 CPU 使用率")
    args = parser.parse_args()

    stop_event = threading.Event()
    mon = SystemMonitor(
        interval=args.interval,
        max_samples=args.count,
        show_per_core=args.per_core,
        csv_path=args.csv,
        stop_event=stop_event,
    )
    mon.start()
    print(f"[Monitor] 开始采样（间隔 {args.interval:.0f}s"
          f"{', 共 ' + str(args.count) + ' 次' if args.count else ''}，"
          f"Ctrl-C 停止）...")
    try:
        mon.join()  # 阻塞直到 stop_event 置位或采样次数达到上限
    except KeyboardInterrupt:
        print("\n[Monitor] 收到 Ctrl-C，停止采样...")
    finally:
        stop_event.set()
        mon.join(timeout=2)
    print("[Monitor] 采样结束")


if __name__ == "__main__":
    main()
