"""banqi/training/buffer.py — 向量化 replay buffer 与 episode 转换。

DataBuffer 把自对弈 episode 转换出的 sample dict 列表压缩进内存向量
（board / scalar / policy / value / mask / root_visit），并按 config
的 VALUE_TARGET_MODE 计算价值目标、过滤异常样本。
"""

from __future__ import annotations

import random
import numpy as np
import torch
from typing import Dict, List

from banqi_training.config import validate_value_target
from banqi_training.constants import build_constants
from banqi_training.variant import Variant


def episode_to_samples(episode_dict: Dict) -> List[Dict]:
    """
    把一个 episode（episode_codec 解码结果，或增强后的同形 dict）转换为
    DataBuffer 可消费的 sample dict 列表。

    字段逐个直取：解码器已按 schema 保证全部字段存在，缺失即报错——不做
    「缺字段就兜底」的静默降级（那正是数据契约漂移的藏身处）。
    策略头验证用的 ground truth 取 actions（MCTS 实际选出的最优动作）。
    """
    samples = []
    boards = episode_dict["boards"]
    scalars = episode_dict["scalars"]
    policies = episode_dict["policies"]
    mcts_values = episode_dict["mcts_values"]
    completed_qs = episode_dict["completed_qs"]
    root_visits = episode_dict["root_visits"]
    game_results = episode_dict["game_results"]
    masks = episode_dict["action_masks"]
    health_diffs = episode_dict["health_diffs"]
    actions = episode_dict["actions"]
    is_full_search = episode_dict["is_full_search"]
    for board, scalar, policy, mcts_val, completed_q, root_visit, game_result, mask, \
            health_diff, action, is_full in zip(
        boards, scalars, policies, mcts_values, completed_qs, root_visits,
        game_results, masks, health_diffs, actions, is_full_search,
    ):
        samples.append({
            "board_state": board,
            "scalar_state": scalar,
            "policy_probs": policy,
            "mcts_value": float(mcts_val),
            "completed_q": float(completed_q),
            "root_visit_count": int(root_visit),
            "game_result_value": float(game_result),
            "action_mask": mask,
            "teacher_action": int(action),
            "health_diff": float(health_diff),
            "is_full_search": bool(is_full),
        })
    return samples


class DataBuffer:
    """向量化环形缓冲区（预分配固定容量 numpy 数组，避免反复分配/释放导致的
    内存碎片化/RSS 线性增长），value 目标按 config.VALUE_TARGET_MODE 计算。

    样本按环形写入覆盖最旧数据；`len()` 返回当前有效样本数；`get_batch` 用
    高级索引在预分配数组上切片，不产生新的大分配。
    """

    def __init__(self, capacity: int, variant: Variant, cfg) -> None:
        self.capacity = max(int(capacity), 1)
        self.variant = variant
        self.cfg = cfg
        # value / policy 目标取值域校验：与运行时热更（config.apply_overrides）共用
        # 同一套判据，保证远程改完的取值域与启动时一致
        validate_value_target(
            mode=cfg.VALUE_TARGET_MODE,
            temperature=cfg.POLICY_TARGET_TEMPERATURE,
            action_mix=cfg.POLICY_TARGET_ACTION_MIX,
        )
        self.C = build_constants(variant)
        self.capacity = max(self.capacity, self.C.TRAIN_BATCH
                            if hasattr(self.C, "TRAIN_BATCH") else 32)
        c = self.capacity
        C = self.C
        # 预分配固定容量数组（一次性分配，后续只覆盖写入，不触发反复 malloc/free）
        self.boards = np.empty((c, C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS),
                               dtype=np.float32)
        self.scalars = np.empty((c, C.SCALAR_FEATURE_COUNT), dtype=np.float32)
        self.probs = np.empty((c, C.ACTION_SPACE_SIZE), dtype=np.float32)
        self.values = np.empty(c, dtype=np.float32)
        self.masks = np.empty((c, C.ACTION_SPACE_SIZE), dtype=np.float32)
        self.root_visits = np.empty(c, dtype=np.int64)
        # 整型血量差（己方视角）分桶索引：d + INITIAL_HEALTH ∈ [0, HEALTH_DIFF_BINS)
        self.health_labels = np.empty(c, dtype=np.int64)
        # 算力分配随机化的 Full Search 标记：True=Full（参与训练），False=Fast（仅保留）。
        self.is_full = np.ones(c, dtype=bool)
        self._size = 0          # 当前有效样本数
        self._head = 0          # 环形写入位置（下一个覆盖点）
        # anneal 模式下 game_result 的权重（0~1），由 TrainWorker 按轮更新
        self.value_result_weight = 0.0
        # 累计丢弃的异常样本数（NaN/Inf/非法策略），供 TB 数据质量监控
        self.total_dropped = 0
        # B6 信号强度统计：平局样本占比与平局样本的子力差幅度。用于判断
        # 「平局标签改用子力差」的收益上限（若平局子力差恒为 0 则该改造无信息）。
        self._sig_total = 0
        self._sig_draw = 0
        self._sig_draw_hp_abs = 0.0
        self._sig_draw_hp_nonzero = 0

    def _target_value(self, s: Dict) -> float:
        """按 value 目标模式计算训练 target：
          mcts   -> mcts_value（搜索/教师平滑评估，噪声小）
          game   -> game_result_value（AlphaZero 标准，终局真值 ±1）
          completed_q -> completed_q（所选动作的 Q(s,a)，根玩家视角）。
                    与 mcts_value（根 V = 子节点 Q 均值）同量纲、同视角；搜索次数少时
                    Q(s,a) 比 V(s) 的偏差更小、方差更低，可直接作为 value 目标。
          mixed  -> (1-λ)*mcts_value + λ*game_result，λ = VALUE_MIX_GAME_WEIGHT
          anneal -> (1-w)*mcts_value + w*game_result，w 按轮退火
          game_hp-> 胜负用 game_result 真值，平局改用终局子力差（见 _draw_hp_target）。
                    4x2 平局占比过半，平局样本的价值目标原为常数 0（不提供梯度信息）。
        """
        mode = self.cfg.VALUE_TARGET_MODE
        mv = s.get('mcts_value', 0.0)
        gr = s.get('game_result_value', 0.0)
        if mode == "game":
            return float(gr)
        if mode == "completed_q":
            return float(s.get('completed_q', 0.0))
        if mode == "game_hp":
            gr = float(gr)
            return gr if gr != 0.0 else self._draw_hp_target(s.get('health_diff', 0.0))
        if mode == "mixed":
            lam = self.cfg.VALUE_MIX_GAME_WEIGHT
            return (1.0 - lam) * float(mv) + lam * float(gr)
        if mode == "anneal":
            w = self.value_result_weight
            return (1.0 - w) * float(mv) + w * float(gr)
        return float(mv)  # mcts（默认）

    def _target_policy(self, probs: np.ndarray, teacher_action) -> np.ndarray:
        """策略目标变换（两个旋钮默认均为恒等，逐位等价于旧行为）：

          T < 1  : 温度锐化 p' ∝ p^(1/T)，重新归一化（只放大既有分布，不引入新信息）
          ε > 0  : 与搜索动作的 one-hot 混合 p'' = (1-ε)·p' + ε·onehot(a)

        ⚠️ ε 通道要先看数据：实测「记录动作」有 29.7% 不等于 argmax(π')（Gumbel 采样
        噪声），用它做 one-hot 相当于把探索噪声当正确答案，默认关闭。
        teacher_action 非法（缺失/越界）时静默跳过 ε 通道。
        """
        temp = float(self.cfg.POLICY_TARGET_TEMPERATURE)
        mix = float(self.cfg.POLICY_TARGET_ACTION_MIX)
        if temp == 1.0 and mix == 0.0:
            return probs
        out = probs
        if temp != 1.0:
            sharpened = np.power(out, 1.0 / temp)
            total = sharpened.sum()
            if total > 0.0 and np.isfinite(total):
                out = sharpened / total
        if mix > 0.0 and teacher_action is not None:
            a = int(teacher_action)
            if 0 <= a < out.shape[0]:
                onehot = np.zeros_like(out)
                onehot[a] = 1.0
                out = (1.0 - mix) * out + mix * onehot
        return out.astype(np.float32)

    def _draw_hp_target(self, health_diff: float) -> float:
        """平局样本的价值目标：终局子力差 / INITIAL_HEALTH ∈ [-1, 1]。

        尺度必须与胜负样本的 ±1 同量纲：value head 的输出会被 MCTS 当作叶节点
        初始 V，尺度不一致会扭曲搜索先验。health_diff 为归一化值，经
        `Constants.health_diff_int` 精确反推整型子力差（无量化误差）。
        """
        d = self.C.health_diff_int(float(health_diff))
        return max(-1.0, min(1.0, d / float(self.C.INITIAL_HEALTH)))

    def add_samples(self, samples: List[Dict]) -> None:
        C = self.C
        dropped = 0
        for s in samples:
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS
            )
            scalar_arr = np.array(s['scalar_state'], dtype=np.float32)
            if scalar_arr.shape[0] > C.SCALAR_FEATURE_COUNT:
                scalar_arr = scalar_arr[:C.SCALAR_FEATURE_COUNT]
            probs = np.array(s['policy_probs'], dtype=np.float32)
            mask = np.array(s['action_mask'], dtype=np.float32)
            target_val = self._target_value(s)
            # 整型血量差标签：由归一化值反推（Rust 精确整数，见 Constants.health_diff_int），
            # 映射到分桶索引 d + INITIAL_HEALTH；血量差缺失/非有限时落到中位桶（0 差）。
            health_norm = float(s.get('health_diff', 0.0))
            health_bin = self.C.health_diff_int(health_norm) + self.C.INITIAL_HEALTH
            health_bin = max(0, min(self.C.HEALTH_DIFF_BINS - 1, health_bin))

            # ---- NaN/Inf 与非法策略/价值目标过滤（来源校验，防污染训练）----
            # 丢弃含非有限值的 board/scalar/policy/mask/value，以及 policy 含
            # 负值或行和≈0 的样本（此类样本会让 log_softmax/交叉熵产生 NaN 或
            # 梯度消失）。value target 来自 mcts_value/game_result 的组合，若
            # 上游 mcts_value 为 NaN（权重被污染的后遗症）会得到 NaN target，
            # 应在此处拦截而非累积进 buffer。
            if (
                not np.isfinite(board).all()
                or not np.isfinite(scalar_arr).all()
                or not np.isfinite(probs).all()
                or not np.isfinite(mask).all()
                or not np.isfinite(target_val)
                or not np.isfinite(health_norm)
                or (probs < 0.0).any()
                or probs.sum() <= 0.0
            ):
                dropped += 1
                continue

            # 策略目标变换（温度锐化 / 与搜索动作 one-hot 混合；默认恒等）：
            # 放在合法性校验之后，避免对非法输入（负概率）先做 p^(1/T) 产生 NaN。
            probs = self._target_policy(probs, s.get('teacher_action'))

            # 环形写入：覆盖 head 位置，然后 head 前移（超容量则淘汰最旧样本）
            i = self._head
            self.boards[i] = board
            self.scalars[i] = scalar_arr
            self.probs[i] = probs
            self.values[i] = target_val
            self.masks[i] = mask
            self.root_visits[i] = int(s.get('root_visit_count', 0))
            self.health_labels[i] = health_bin
            self.is_full[i] = bool(s.get('is_full_search', True))
            self._head = (self._head + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1

            # B6 信号统计：仅统计成功入队的有效样本
            self._sig_total += 1
            if float(s.get('game_result_value', 0.0)) == 0.0:
                self._sig_draw += 1
                hp_abs = abs(health_norm)
                self._sig_draw_hp_abs += hp_abs
                if hp_abs > 1e-9:
                    self._sig_draw_hp_nonzero += 1

        if dropped:
            self.total_dropped += dropped
            print(
                f"[TR-{self.variant.id}] ⚠️ DataBuffer 丢弃 {dropped} 个异常样本"
                f"（累计 {self.total_dropped}，NaN/Inf/非法策略），Blocked 来自自对弈或冷存储"
            )

    def take_signal_stats(self) -> Dict[str, float]:
        """取走并重置样本信号统计（平局占比 / 平局子力差幅度），供 TB 与日志消费。

        仅统计通过入队校验的有效样本。``game_hp`` 目标的收益上限由
        ``draw_hp_abs_mean`` / ``draw_hp_nonzero_ratio`` 直接反映（全 0 即无信息）。
        """
        total = max(self._sig_total, 1)
        draws = self._sig_draw
        stats = {
            "n_samples": float(self._sig_total),
            "draw_ratio": draws / total,
            "draw_hp_abs_mean": (self._sig_draw_hp_abs / draws) if draws else 0.0,
            "draw_hp_nonzero_ratio": (self._sig_draw_hp_nonzero / draws) if draws else 0.0,
        }
        self._sig_total = 0
        self._sig_draw = 0
        self._sig_draw_hp_abs = 0.0
        self._sig_draw_hp_nonzero = 0
        return stats

    def __len__(self) -> int:
        return self._size

    def sample_recent_indices(self, n: int, alpha: float = 2.0):
        """平滑指数衰减加权采样索引（按需抽取 n 个）。

        RL 自对弈数据随模型演化而"过时"：越新的样本越贴近当前模型水平。
        KataGo 等主流实践使用平滑衰减权重，而非硬分段切块。这里给 Buffer 内
        每个样本按其新旧程度赋指数衰减权重（最新样本权重最高、最旧平滑衰减），
        并按需只抽取训练实际需要的 n 个索引，避免生成/打乱全量 List 的浪费。
        """
        size = self._size
        if size == 0:
            return np.array([], dtype=np.int64)
        n = min(n, size)
        # 索引越接近 size-1（最新），权重越高：w_i = exp(alpha * i/size)
        weights = np.exp(np.linspace(-alpha, 0.0, size))
        probs = weights / weights.sum()
        return np.random.choice(size, size=n, replace=False, p=probs).astype(np.int64)

    def get_batch(self, indices):
        """用高级索引在预分配数组上切片（不产生新的大分配，内存稳定）。

        indices 为 [0, len(self)) 内的整数索引（logical 索引），映射到环形
        存储的实际位置后做 fancy indexing。未满时 logical i 对应物理 i；已满
        时 logical 0 是最旧样本，对应物理位置 head。
        """
        idx = np.asarray(indices, dtype=np.int64)
        if self._size < self.capacity:
            actual = idx
        else:
            actual = (idx + self._head) % self.capacity
        b = torch.from_numpy(self.boards[actual])
        s = torch.from_numpy(self.scalars[actual])
        p = torch.from_numpy(self.probs[actual])
        v = torch.from_numpy(self.values[actual])
        m = torch.from_numpy(self.masks[actual])
        f = torch.from_numpy(self.is_full[actual])
        h = torch.from_numpy(self.health_labels[actual]).long()
        return b, s, p, v, m, f, h
