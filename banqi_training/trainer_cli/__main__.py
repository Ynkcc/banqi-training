"""banqi/trainer_cli/__main__.py — 支持 `python -m banqi_training.trainer_cli [variant]`。"""

from .cli import programmatic_entry

if __name__ == "__main__":
    # 支持 `python -m banqi_training.trainer_cli 4x8 [--train-mode ...]`，
    # 把位置参数 variant 透传给 argparse。
    programmatic_entry()
