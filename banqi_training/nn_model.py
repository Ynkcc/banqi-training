"""banqi/nn_model.py — 参数化策略-价值网络（AlphaZero 风格）

一份 `BanqiNet` 服务 4x2 / 4x4 / 4x8 三个变体：所有维度（输入通道、棋盘尺寸、
标量维度、动作空间、残差块数、头尺寸）由 `Variant` / `Constants` 派生。

网络类名统一为 `BanqiNet`（旧 Banqi4x4Net / MiniBanqiNet 不再需要）。
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from banqi_training.constants import Constants, build_constants
from banqi_training.variant import Variant


class BasicBlock(nn.Module):
    """标准残差块：Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out)


class BanqiNet(nn.Module):
    """AlphaZero 策略-价值网络，结构由 variant 决定。

    输入: board (N, C, R, C)，scalars (N, S)
    输出: policy_logits (N, A)，value (N, 1)，
          以及（enable_health 时）health_logits (N, K)，K=2*INITIAL_HEALTH+1 个整型血量差分桶。

    value 的语义由 `enable_value_dist` 决定，**对外形状恒为 (N, 1)**，Rust 侧
    ONNX/TorchScript 契约不随开关变化：
      - false：tanh 标量回归（MSE 训练）；
      - true ：K 桶分布 + HL-Gauss 交叉熵训练，导出值为分布期望 Σ p_i·c_i。
    `forward(..., return_value_logits=True)` 额外返回 value 分布 logits（仅训练侧
    取损失用，导出路径不传该参数，故 ONNX 图内不含该输出）。
    """

    def __init__(
        self,
        variant: Variant,
        enable_health: bool = False,
        enable_value_dist: bool = False,
        value_dist_bins: int = 65,
    ) -> None:
        super().__init__()
        self.variant_id = variant.id
        self.enable_health = bool(enable_health)
        self.enable_value_dist = bool(enable_value_dist)
        if self.enable_value_dist and (value_dist_bins < 3 or value_dist_bins % 2 == 0):
            raise ValueError(
                f"VALUE_DIST_BINS 必须为 ≥3 的奇数（保证存在 0 中心桶）: {value_dist_bins}"
            )
        self.value_dist_bins = int(value_dist_bins)
        c: Constants = build_constants(variant)
        hidden = c.HIDDEN_CHANNELS
        rows, cols = c.BOARD_ROWS, c.BOARD_COLS
        scalar = c.SCALAR_FEATURE_COUNT

        # 1. 输入卷积
        self.conv_input = nn.Conv2d(
            c.TOTAL_INPUT_CHANNELS, hidden, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(hidden)

        # 2. 残差塔
        self.res_tower = nn.ModuleList(
            [BasicBlock(hidden) for _ in range(c.NUM_RES_BLOCKS)]
        )

        # 3. 策略头
        self.policy_channels = c.POLICY_HEAD_CHANNELS
        self.policy_conv = nn.Conv2d(
            hidden, self.policy_channels, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(self.policy_channels)
        self.policy_flat_size = self.policy_channels * rows * cols
        self.policy_fc_input = self.policy_flat_size + scalar
        self.policy_fc1 = nn.Linear(self.policy_fc_input, c.POLICY_FC1_HIDDEN)
        self.policy_fc2 = nn.Linear(c.POLICY_FC1_HIDDEN, c.ACTION_SPACE_SIZE)

        # 4. 价值头
        #    分布化（enable_value_dist）时输出 K 个分桶 logits（覆盖归一化价值
        #    v∈[-1,1] 的等距桶心），经 softmax 求期望后作为 value 输出；
        #    关闭时与旧版逐位等价（单标量 + tanh）。
        self.value_channels = c.VALUE_HEAD_CHANNELS
        self.value_conv = nn.Conv2d(
            hidden, self.value_channels, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(self.value_channels)
        self.value_flat_size = self.value_channels * rows * cols
        self.value_fc_input = self.value_flat_size + scalar
        self.value_fc1 = nn.Linear(self.value_fc_input, c.VALUE_FC1_HIDDEN)
        self.value_fc2 = nn.Linear(
            c.VALUE_FC1_HIDDEN, self.value_dist_bins if self.enable_value_dist else 1
        )
        if self.enable_value_dist:
            self.register_buffer(
                "value_centers", torch.linspace(-1.0, 1.0, self.value_dist_bins)
            )

        # 5. 血量差异头（可选，离散分类，非标量回归）
        #    输出 K=HEALTH_DIFF_BINS 个 logits（整型血量差 -D..+D 的分桶分布），
        #    标签为 One-hot 桶索引；关闭时完全不加该头，与旧模型逐位等价。
        if self.enable_health:
            self.health_channels = c.VALUE_HEAD_CHANNELS
            self.health_conv = nn.Conv2d(
                hidden, self.health_channels, kernel_size=1, bias=False
            )
            self.health_bn = nn.BatchNorm2d(self.health_channels)
            self.health_flat_size = self.health_channels * rows * cols
            self.health_fc_input = self.health_flat_size + scalar
            self.health_fc1 = nn.Linear(self.health_fc_input, c.VALUE_FC1_HIDDEN)
            self.health_fc2 = nn.Linear(c.VALUE_FC1_HIDDEN, c.HEALTH_DIFF_BINS)

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor, return_value_logits: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """前向推理。返回顺序：policy_logits, value[, health_logits][, value_logits]。

        return_value_logits=True 时在末尾追加 value 分布 logits（[N, K]），仅供
        训练侧计算 HL-Gauss 交叉熵；`checkpoint.export_*` 不传该参数，因此导出的
        ONNX/TorchScript 图输出与关闭分布化时逐位一致（policy_logits/value[/health]）。
        """
        x = self.conv_input(board)
        x = self.bn_input(x)
        x = F.relu(x)
        for block in self.res_tower:
            x = block(x)

        # 策略头
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc2(F.relu(self.policy_fc1(torch.cat([p, scalars], dim=1))))

        # 价值头
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        value_out = self.value_fc2(F.relu(self.value_fc1(torch.cat([v, scalars], dim=1))))
        if self.enable_value_dist:
            value = (F.softmax(value_out, dim=1) * self.value_centers).sum(dim=1, keepdim=True)
        else:
            value = torch.tanh(value_out)

        outs: list[torch.Tensor] = [policy_logits, value]
        # 血量头：K 维 logits（离散分桶），不经过 tanh（损失侧 softmax + CE）
        if self.enable_health:
            h = self.health_conv(x)
            h = self.health_bn(h)
            h = F.relu(h)
            h = h.view(h.size(0), -1)
            outs.append(
                self.health_fc2(F.relu(self.health_fc1(torch.cat([h, scalars], dim=1))))
            )
        # 注：用 `is True` 而非真值判断——ONNX 传统导出器（dynamo=False）在求值 bool
        # 形参的真值时会对每次导出误报 TracerWarning（torch 2.9 实测），恒等比较不经过
        # 该路径。调用方（train_step）传入的是真 bool，语义等价。
        if return_value_logits is True and self.enable_value_dist:
            outs.append(value_out)

        return tuple(outs)


def load_model_weights(model: nn.Module, path: str, device: torch.device) -> None:
    """从 path 加载权重，兼容 TorchScript(.pt) / state_dict(.pth) / checkpoint dict。

    按文件扩展名判定加载格式；任何加载/装载失败都直接抛出，不做静默降级
    （静默吞掉异常会掩盖文件损坏、张量 Shape 变化等真正错误）。
    """
    if os.path.splitext(path)[1].lower() == ".pt":
        jit_model = torch.jit.load(path, map_location=device)
        model.load_state_dict(jit_model.state_dict())
        return
    state = torch.load(path, map_location=device, weights_only=True)
    if hasattr(state, "state_dict"):
        model.load_state_dict(state.state_dict())
    elif isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _dummy_shapes(variant: Variant) -> Dict[str, Any]:
    c = build_constants(variant)
    return {
        "board": (c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS),
        "scalar": c.SCALAR_FEATURE_COUNT,
        "action": c.ACTION_SPACE_SIZE,
    }


if __name__ == "__main__":
    from banqi_training.variant import VARIANTS
    for vid, v in VARIANTS.items():
        c = build_constants(v)
        for enable_health, enable_value_dist in ((False, False), (True, False), (False, True), (True, True)):
            model = BanqiNet(
                v, enable_health=enable_health, enable_value_dist=enable_value_dist
            ).eval()
            batch = 2
            board = torch.randn(batch, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS)
            scalars = torch.randn(batch, c.SCALAR_FEATURE_COUNT)
            with torch.inference_mode():
                out = model(board, scalars)
                # 导出路径（默认参数）必须不含 value_logits，保证 ONNX 输出数不变
                assert len(out) == (3 if enable_health else 2), f"{vid} 导出输出数异常: {len(out)}"
                logits, value = out[0], out[1]
                assert logits.shape == (batch, c.ACTION_SPACE_SIZE), f"{vid} logits shape"
                assert value.shape == (batch, 1), f"{vid} value shape"
                assert bool((value.abs() <= 1.0).all()), f"{vid} value 超出 [-1,1]"
                extra = ""
                if enable_health:
                    health = out[2]
                    assert health.shape == (batch, c.HEALTH_DIFF_BINS), f"{vid} health shape"
                    extra += f" health={tuple(health.shape)} bins={c.HEALTH_DIFF_BINS}"
                if enable_value_dist:
                    train_out = model(board, scalars, True)
                    value_logits = train_out[-1]
                    assert value_logits.shape == (batch, model.value_dist_bins), \
                        f"{vid} value_logits shape"
                    expect = (torch.softmax(value_logits, dim=1)
                              * model.value_centers).sum(dim=1, keepdim=True)
                    assert torch.allclose(expect, value, atol=1e-6), f"{vid} 期望值与导出值不一致"
                    extra += f" value_dist={model.value_dist_bins}桶"
            print(f"[banqi_training.nn_model] {vid} health={enable_health} "
                  f"vdist={enable_value_dist}: "
                  f"input={tuple(board.shape[1:])} scalar={c.SCALAR_FEATURE_COUNT} "
                  f"action={c.ACTION_SPACE_SIZE} params={count_params(model)} "
                  f"-> logits={tuple(logits.shape)} value={tuple(value.shape)}{extra}")
    print("[banqi_training.nn_model] all OK")
