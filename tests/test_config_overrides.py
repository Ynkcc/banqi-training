"""config.apply_overrides / validate_value_target 的取值域与回退语义。"""

from __future__ import annotations

import pytest

from banqi_training.config import (
    TRAIN_CONFIG_OVERRIDABLE,
    Config,
    apply_overrides,
    snapshot_overridable,
    validate_value_target,
)

# 白名单全部字段的合法假值：apply_overrides / snapshot_overridable 会遍历整个白名单，
# 冒烟测试不需要真实 YAML，直接用 object.__new__ 造一个只带这些字段的 Config。
_VALID = {
    "LEARNING_RATE": 1e-3,
    "MIN_LR": 1e-5,
    "LR_DECAY_STEPS": 1000,
    "LR_DECAY_ROUNDS": 0,
    "TRAIN_BATCH": 256,
    "TRAIN_EPOCHS_PER_ROUND": 1,
    "MIN_NEW_SAMPLES_TO_TRAIN": 0,
    "WEIGHT_DECAY": 0.0,
    "EMA_DECAY": 0.999,
    "RECENT_SAMPLE_ENABLED": False,
    "FAST_SAMPLE_LOSS_WEIGHT": 0.0,
    "VALUE_TARGET_MODE": "mcts",
    "VALUE_TARGET_ANNEAL_ROUNDS": 0,
    "VALUE_MIX_GAME_WEIGHT": 0.0,
    "POLICY_TARGET_TEMPERATURE": 1.0,
    "POLICY_TARGET_ACTION_MIX": 0.0,
    "HEALTH_LOSS_WEIGHT": 0.0,
    "HEALTH_GAUSS_SIGMA": 0.5,
    "VALUE_GAUSS_SIGMA": 0.5,
    "DATA_AUGMENT_ENABLED": False,
    "DATA_AUGMENT_K": 1,
    "DATA_AUGMENT_KEEP_ORIGINAL": True,
    "REANALYSIS_BATCH_POSITIONS": 32,
    "REANALYSIS_SUBMIT_EVERY_N_ROUNDS": 1,
    "REANALYSIS_MCTS_SIMS": 0,
    "MAX_RUNTIME_SECONDS": 0,
    "SHOULD_STOP_POLL_SECONDS": 0,
}


def _bare_config(**overrides) -> Config:
    cfg = object.__new__(Config)
    for name, value in {**_VALID, **overrides}.items():
        setattr(cfg, name, value)
    return cfg


def test_whitelist_names_exist_on_config():
    """白名单字段名必须都是 Config 的字段：两侧清单漂移会静默下发到不存在的字段。"""
    fields = set(Config.__dataclass_fields__)
    assert TRAIN_CONFIG_OVERRIDABLE <= fields, TRAIN_CONFIG_OVERRIDABLE - fields


def test_value_target_rejects_unknown_mode():
    with pytest.raises(ValueError, match="VALUE_TARGET_MODE"):
        validate_value_target(mode="mctss", temperature=1.0, action_mix=0.0)


def test_value_target_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="POLICY_TARGET_TEMPERATURE"):
        validate_value_target(mode="mcts", temperature=0.0, action_mix=0.0)


def test_rejects_field_outside_whitelist():
    cfg = _bare_config()
    with pytest.raises(ValueError, match="不支持远程下发"):
        apply_overrides(cfg, {"HEALTH_VALUE_HEAD_ENABLED": "true"})


def test_casts_values_by_field_type():
    cfg = _bare_config()
    changed = apply_overrides(
        cfg, {"TRAIN_BATCH": "64", "POLICY_TARGET_TEMPERATURE": "0.5", "RECENT_SAMPLE_ENABLED": "true"}
    )
    assert changed == ["POLICY_TARGET_TEMPERATURE", "RECENT_SAMPLE_ENABLED", "TRAIN_BATCH"]
    assert cfg.TRAIN_BATCH == 64 and isinstance(cfg.TRAIN_BATCH, int)
    assert cfg.POLICY_TARGET_TEMPERATURE == 0.5
    assert cfg.RECENT_SAMPLE_ENABLED is True


def test_rejects_whole_batch_when_any_item_invalid():
    cfg = _bare_config()
    with pytest.raises(ValueError, match="POLICY_TARGET_TEMPERATURE"):
        apply_overrides(cfg, {"TRAIN_BATCH": "64", "POLICY_TARGET_TEMPERATURE": "0"})
    # 整批拒绝：不允许停在「TRAIN_BATCH 已改、温度没改」的半套状态
    assert cfg.TRAIN_BATCH == _VALID["TRAIN_BATCH"]
    assert cfg.POLICY_TARGET_TEMPERATURE == _VALID["POLICY_TARGET_TEMPERATURE"]


def test_baseline_restores_local_value_when_override_removed():
    cfg = _bare_config()
    baseline = snapshot_overridable(cfg)
    apply_overrides(cfg, {"TRAIN_BATCH": "64", "EMA_DECAY": "0.99"}, baseline)
    assert (cfg.TRAIN_BATCH, cfg.EMA_DECAY) == (64, 0.99)

    # 调度层清空全部覆盖：应回落到本地基线，而不是保持上一次的下发值
    changed = apply_overrides(cfg, {}, baseline)
    assert changed == ["EMA_DECAY", "TRAIN_BATCH"]
    assert cfg.TRAIN_BATCH == _VALID["TRAIN_BATCH"]
    assert cfg.EMA_DECAY == _VALID["EMA_DECAY"]


def test_unchanged_values_are_not_reported():
    cfg = _bare_config()
    assert apply_overrides(cfg, {"TRAIN_BATCH": str(_VALID["TRAIN_BATCH"])}) == []
