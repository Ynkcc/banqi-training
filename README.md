# banqi-training

Banqi 分布式训练器（从 `rust_4x8/python` 抽取，仅保留分布式形态），被
[banqi-scheduler](../banqi-scheduler) 调度。

## 职责

- 经 scheduler gRPC `ListEpisodes` 拉取 worker 直传的 episode 批次（预签名 GET，trainer 零存储配置）
- `TrainWorker` 消费训练（buffer / augment / losses / checkpoint）
- watch 导出的 onnx，经 `SignNetworkUpload` 预签名直传 + `RegisterNetwork` 登记（gatekeeper 判停晋级）

自对弈 worker（banqi-collector）独立部署，不在本仓库范围内。

## 安装

wheel 的构建与分发见 [`../deploy/training/`](../deploy/training)：

```bash
deploy/training/build.sh                  # 构建 wheel
deploy/training/deploy.sh --host <目标机>  # 分发并安装
```

开发机本地安装：`pip install -e .`（非 torch 依赖）；需要 torch 时 `pip install -e ".[torch]"`。

## 启动

```bash
export BANQI_CONFIG=/path/to/config.yaml      # 不设置则读包目录内的 config.local.yaml
export SCHEDULER_ENDPOINT=http://<scheduler>:50051
python -m banqi_training.trainer_cli 4x8
```

变体 id 以调度器 `GetInfo` 下发为准，命令行位置参数仅作默认值。

## 结构

```
proto/scheduler.proto        gRPC 契约 + 训练数据记录 schema（从 banqi-scheduler 复制，单一来源在 scheduler 仓库）
banqi_training/
  proto/                     pb2 生成物（见 banqi_training/proto/__init__.py 注释的生成命令）
  episode_codec.py           EpisodeBatch 二进制解码（训练数据唯一解码实现，零拷贝还原张量）
  infra/                     SchedulerEpisodeStore / SchedulerModelRegistry / scheduler_variant
  training/                  TrainWorker、buffer、losses、eval、lr_schedule、augment
  trainer_cli/               CLI 与 runners/distributed.py 编排入口
  config.py + config.default.yaml
  variant/actions/constants/nn_model/checkpoint/storage/tb_logger/memory_guard/system_monitor
  symmetry.py                空间对称增强（纯 Python：动作表 / D4 置换 / board 重排）
tests/test_symmetry.py       对称增强单测（置换合法性 / 动作计数 / 增强一致性）
tests/test_episode_codec.py  episode 记录解码单测（位平面布局 / 张量还原 / 契约校验）
```

训练数据格式：worker 把一批 episode 编码为 `EpisodeBatch`（proto 定义，字段号 +
`schema_version`）→ gzip → 直传 R2，对象键 `episodes/<sha>/<id>.epb.gz`；本端由
`episode_codec.decode_episode_batch` 解码为 numpy 张量（棋盘位平面 / 掩码位图位打包，
标量策略等走稠密小端缓冲）。版本不认识、变体不符、长度不符一律抛错，不静默降级。

## 与其他仓库的关系

- `banqi-scheduler`：gRPC 契约（`proto/scheduler.proto`）；proto 变更先改 scheduler 仓库再复制过来重新生成 pb2
- `rust_4x8`：无运行期依赖（2026-09-11 移除 PyO3 扩展 `banqi_4x8`：维度核对改用 Python
  `build_constants` 派生值，对称增强下沉为 `symmetry.py` 纯 Python 实现，对战评估
  移交调度器 gatekeeper rating）；本仓库代码独立演进，不再回写 rust_4x8

