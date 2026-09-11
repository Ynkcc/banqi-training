# banqi-training

Banqi 分布式训练器（从 `rust_4x8/python` 抽取，仅保留分布式形态），被
[banqi-scheduler](../banqi-scheduler) 调度。

## 职责

- 经 scheduler gRPC `ListEpisodes` 拉取 worker 直传的 episode 批次（预签名 GET，trainer 零存储配置）
- `TrainWorker` 消费训练（buffer / augment / losses / checkpoint）
- watch 导出的 onnx，经 `SignNetworkUpload` 预签名直传 + `RegisterNetwork` 登记（gatekeeper 判停晋级）

自对弈 worker（banqi-collector）独立部署，不在本仓库范围内。

## 启动

```bash
pip install -r requirements.txt

export SCHEDULER_ENDPOINT=http://<scheduler>:50051
python -m banqi_training.trainer_cli 4x8
```

变体 id 以调度器 `GetInfo` 下发为准，命令行位置参数仅作默认值。

## 结构

```
proto/scheduler.proto        gRPC 契约（从 banqi-scheduler 复制，单一来源在 scheduler 仓库）
banqi_training/
  proto/                     pb2 生成物（见 banqi_training/proto/__init__.py 注释的生成命令）
  infra/                     SchedulerEpisodeStore / SchedulerModelRegistry / scheduler_variant
  training/                  TrainWorker、buffer、losses、eval、lr_schedule、augment
  trainer_cli/               CLI 与 runners/distributed.py 编排入口
  config.py + config.default.yaml
  variant/actions/constants/nn_model/checkpoint/storage/tb_logger/memory_guard/system_monitor
  symmetry.py                空间对称增强（纯 Python：动作表 / D4 置换 / board 重排）
tests/test_symmetry.py       对称增强单测（置换合法性 / 动作计数 / 增强一致性）
```

## 与其他仓库的关系

- `banqi-scheduler`：gRPC 契约（`proto/scheduler.proto`）；proto 变更先改 scheduler 仓库再复制过来重新生成 pb2
- `rust_4x8`：无运行期依赖（2026-09-11 移除 PyO3 扩展 `banqi_4x8`：维度核对改用 Python
  `build_constants` 派生值，对称增强下沉为 `symmetry.py` 纯 Python 实现，对战评估
  移交调度器 gatekeeper rating）；本仓库代码独立演进，不再回写 rust_4x8

