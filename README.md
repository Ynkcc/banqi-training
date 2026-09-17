# banqi-training

Banqi 分布式训练器（从 `rust_4x8/python` 抽取，仅保留分布式形态），被
[banqi-scheduler](../banqi-scheduler) 调度。

## 职责

- 经 scheduler gRPC `ListEpisodes` 拉取 worker 直传的 episode 批次（预签名 GET，trainer 零存储配置）
- `TrainWorker` 消费训练（buffer / augment / losses / checkpoint）
- watch 导出的 onnx，经 `SignNetworkUpload` 预签名直传 + `RegisterNetwork` 登记（gatekeeper 判停晋级）
- 按 `SHOULD_STOP_POLL_SECONDS` 轮询 `GetInfo.should_stop`：调度器按绝对强度判据置位后优雅停止
- 训练超参可远程调整：启动时经 `GetTrainConfig` 引导（覆盖本地 YAML 对应字段），运行中按同一轮询节奏热更；可调字段白名单与取值校验由调度器裁定（见 [banqi-scheduler §5.1](../banqi-scheduler/ARCHITECTURE.md)），本端 `config.py::TRAIN_CONFIG_OVERRIDABLE` 须与调度器白名单同步增删。删除覆盖即回落本地值

自对弈 worker（banqi-collector）独立部署，不在本仓库范围内。

## 评估口径（重要）

本仓库**只负责训练**，不执行对局（分布式形态下 trainer 没有棋引擎）。两件事分别在别处：

| 事项 | 位置 |
|---|---|
| **绝对强度**（vs 规则/内建对手的胜率阶梯） | 调度器下发 `TASK_EVAL` 给 collector 执行，结果落 `eval_results`，见调度器 WebUI「绝对强度」页 |
| **相对强度**（candidate vs 当前 best 的晋级判定） | 调度器 gatekeeper + GSPRT |
| **训练信号健康度**（价值/策略头随训练的变化） | 本仓库：固定验证集指标（`value_drift/*`、`policy_acc/*`，TensorBoard） |

⚠️ 相对强度门禁**结构上测不出**「所有版本都打不过一个 3 行启发式」——必须配套绝对强度阶梯。
⚠️ 训练信号健康度用 `value_drift/auc_win_loss`（价值头胜负 AUC）判断，**不要**只看 `corr(终局)`：
`game_hp` 目标下该项天然偏低；而**恒为 0.000 或 n/a 说明固定验证集退化**（终局类别不足两类，
`build_fixed_eval` 会拒绝构建并打印原因）。历史上该仪表静默失效 268 轮无人发现。

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
tests/test_config_overrides.py  训练超参覆盖单测（白名单一致 / 校验 / 类型转换 / 原子拒绝 / 基线回落）
```

训练数据格式：worker 把一批 episode 编码为 `EpisodeBatch`（proto 定义，字段号 +
`schema_version`）→ gzip → 直传 R2，对象键 `episodes/<sha>/<id>.epb.gz`；本端由
`episode_codec.decode_episode_batch` 解码为 numpy 张量（棋盘位平面 / 掩码位图位打包，
标量策略等走稠密小端缓冲）。版本不认识、变体不符、长度不符一律抛错，不静默降级。

数据类别：`EpisodeBatch.kind` 区分 `resnet`（Gumbel MCTS 稠密特征）与 `nnue`
（Expectimax 稀疏特征），两类互斥。主闭环只消费 `resnet`（`SchedulerEpisodeStore`
的 `kind` 参数既用于服务端 `ListEpisodes` 过滤，也用于解码校验）；NNUE 数据由
独立的蒸馏消费方处理（尚未接通）。

## 与其他仓库的关系

- `banqi-scheduler`：gRPC 契约（`proto/scheduler.proto`）；proto 变更先改 scheduler 仓库再复制过来重新生成 pb2
- `rust_4x8`：无运行期依赖（2026-09-11 移除 PyO3 扩展 `banqi_4x8`：维度核对改用 Python
  `build_constants` 派生值，对称增强下沉为 `symmetry.py` 纯 Python 实现，对战评估
  移交调度器 gatekeeper rating）；本仓库代码独立演进，不再回写 rust_4x8

