"""banqi/trainer_cli — 训练 CLI 子包。

拆分自原单文件 train.py，按职责分层：
  config_resolver.py : 配置构造、冲突校验、运行时预算
  runners.py         : main / _run_offline / _run_selfplay 模式编排
  cli.py             : argparse 与程序化入口

train.py 保留为向后兼容的 re-export 入口。
"""

from .runners import main, build_const
from .cli import parse_args, programmatic_entry

__all__ = ["main", "build_const", "parse_args", "programmatic_entry"]
