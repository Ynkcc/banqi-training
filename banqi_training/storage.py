# banqi/storage.py — 可复用存储层（共享实现）
#
# 集中存放 JSON 安全序列化、本地文件归档（FileSaver）、MongoDB 归档（MongoSaver）
# 与本地 JSONL 冷存储读取/转换工具，供 self_play / predictor_entry / archiver
# / train_imitate 复用，消除跨脚本重复定义。
#
# 由原 python/storage.py（4x8）与 game_4x4/storage.py（4x4）合并：
#   - MongoSaver 默认库名统一为 DEFAULT_DB_NAME（调用方按变体显式传 db_name）
#   - JSONL 读取 / 流式迭代 / episode->samples 转换来自 4x4 版
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_DB_NAME = "banqi_training"
DEFAULT_COLLECTION = "games"


# ============================================================================
# JSON 安全序列化
# ============================================================================

def to_json_safe(obj):
    """把 numpy 标量 / ndarray / 嵌套容器转换为可 JSON 序列化的纯 Python 对象。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


# ============================================================================
# 本地文件保存器
# ============================================================================

class FileSaver:
    """
    本地数据保存器：
      - save_format="jsonl": 每局一行 JSON，追加写
          output/iter_{iteration}_worker_{worker_id}.jsonl
      - save_format="json":  每局一个 JSON 文件，归档到
          output/games/game_{game_count}.json
    """

    def __init__(self, output_dir: str, save_format: str = "jsonl") -> None:
        self.output_dir = output_dir
        self.save_format = save_format
        os.makedirs(output_dir, exist_ok=True)
        if save_format == "json":
            os.makedirs(os.path.join(output_dir, "games"), exist_ok=True)
        # 追加写模式下复用已打开的句柄，避免反复 open
        self._open_fp: Optional[object] = None
        self._open_key: Optional[Tuple[int, int]] = None

    def _get_fp(self, iteration: int, worker_id: int):
        key = (iteration, worker_id)
        if self.save_format == "jsonl":
            if self._open_key != key:
                self.close()
                path = os.path.join(
                    self.output_dir, f"iter_{iteration:06d}_worker_{worker_id:03d}.jsonl"
                )
                self._open_fp = open(path, "a", encoding="utf-8")
                self._open_key = key
            return self._open_fp
        return None

    def save_episodes(
        self,
        episode_dicts: List[Dict],
        iteration: int,
        worker_id: int,
        game_start: int,
    ) -> None:
        if self.save_format == "jsonl":
            fp = self._get_fp(iteration, worker_id)
            for ep in episode_dicts:
                fp.write(json.dumps(to_json_safe(ep), ensure_ascii=False))
                fp.write("\n")
            fp.flush()
        else:  # json 归档
            games_dir = os.path.join(self.output_dir, "games")
            for idx, ep in enumerate(episode_dicts):
                path = os.path.join(games_dir, f"game_{game_start + idx:06d}.json")
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(to_json_safe(ep), fp, ensure_ascii=False)

    def close(self) -> None:
        if self._open_fp is not None:
            self._open_fp.close()
            self._open_fp = None
            self._open_key = None


# ============================================================================
# 本地 JSONL 冷存储读取（复用训练数据）
# ============================================================================

def list_jsonl_files(archive_dir: str) -> List[str]:
    """列出归档目录下所有 *.jsonl 文件（升序，保证迭代顺序）。"""
    if not os.path.isdir(archive_dir):
        return []
    return sorted(
        os.path.join(archive_dir, f)
        for f in os.listdir(archive_dir)
        if f.endswith(".jsonl")
    )


def load_jsonl_episodes(archive_dir: str, limit_games: Optional[int] = None) -> List[Dict]:
    """从本地 JSONL 冷存储加载 episode dict 列表。

    每行一局（FileSaver jsonl 格式），按文件/行序加载。可选限制局数，
    用于控制内存占用与训练分布。
    """
    episodes: List[Dict] = []
    for path in list_jsonl_files(archive_dir):
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                episodes.append(json.loads(line))
                if limit_games is not None and len(episodes) >= limit_games:
                    return episodes
    return episodes


def iter_jsonl_episodes(archive_dir: str):
    """流式迭代 JSONL 归档中的 episode dict（逐行 yield，不一次性物化全部）。

    用于大幅降低内存占用：一批大对局数组不会同时驻留内存。
    按文件/行序 yield；调用方负责及时丢弃。
    """
    for path in list_jsonl_files(archive_dir):
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def episode_dict_to_samples(ep: Dict) -> List[Dict]:
    """把归档的 episode dict 转成训练样本列表（value 取 mcts_value 平滑评估）。"""
    samples: List[Dict] = []
    boards = ep.get("boards") or ep.get("board_states") or []
    scalars = ep.get("scalars") or ep.get("scalar_states") or []
    policies = ep.get("policies") or ep.get("policy_probs") or []
    mcts_values = ep.get("mcts_values") or []
    game_results = ep.get("game_results") or []
    masks = ep.get("action_masks") or []
    health_diffs = ep.get("health_diffs") or []
    # 策略头验证 ground truth：优先 teacher_actions（规则/启发式最优），
    # fallback 到 actions（MCTS 最优动作）；两者都缺时置 None（验证时跳过）。
    teacher_actions = ep.get("teacher_actions") or []
    actions = ep.get("actions") or []
    for i, (board, scalar, policy, mv, gr, mask) in enumerate(zip(
            boards, scalars, policies, mcts_values, game_results, masks)):
        teacher_action = None
        if i < len(teacher_actions) and teacher_actions[i] is not None:
            teacher_action = int(teacher_actions[i])
        elif i < len(actions) and actions[i] is not None:
            teacher_action = int(actions[i])
        samples.append({
            "board_state": board,
            "scalar_state": scalar,
            "policy_probs": policy,
            "mcts_value": float(mv) if mv is not None else 0.0,
            "game_result_value": float(gr) if gr is not None else 0.0,
            "action_mask": mask,
            "teacher_action": teacher_action,
            "health_diff": float(health_diffs[i]) if i < len(health_diffs) else 0.0,
        })
    return samples


def save_episodes_to_archive(episode_dicts: List[Dict], archive_dir: str,
                             iteration: int = 0, worker_id: int = 0) -> int:
    """把 episode dict 列表以 JSONL 追加写方式存到冷存储（复用 FileSaver）。

    返回写入局数。iteration/worker_id 控制归档文件名（避免同名覆盖）；
    同 iteration 内多次调用会追加到同一文件。
    """
    saver = FileSaver(archive_dir, save_format="jsonl")
    try:
        n = len(episode_dicts)
        if n:
            saver.save_episodes(episode_dicts, iteration=iteration,
                                worker_id=worker_id, game_start=0)
        return n
    finally:
        saver.close()


# ============================================================================
# MongoDB 保存器
# ============================================================================

class MongoSaver:
    """
    可选 MongoDB 保存器：把 episode_dict 转换为与 mongodb_storage.rs 的
    GameDocument / SampleDocument 一致的结构后 insert_many。
    连接失败时抛异常，由上层降级为本地保存。

    注意：库名/集合名默认取 DEFAULT_DB_NAME/DEFAULT_COLLECTION，
    多变体共用时请显式传 db_name（如 make_config(vid).DB_NAME）。
    """

    def __init__(self, uri: str, db_name: str = DEFAULT_DB_NAME,
                 collection: str = DEFAULT_COLLECTION) -> None:
        import pymongo  # 延迟导入，缺失时上层捕获降级
        from datetime import datetime, timezone

        self._now_utc = lambda: datetime.now(timezone.utc)
        self.client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.collection = self.client[db_name][collection]
        # 主动 ping 一次，验证连接
        self.client.admin.command("ping")
        print(f"[MongoSaver] 连接成功: {uri} -> {db_name}.{collection}")

    def save_episodes(self, episode_dicts: List[Dict], iteration: int, **_) -> None:
        documents: List[Dict] = []
        for ep in episode_dicts:
            samples = []
            health_diffs = ep.get("health_diffs") or [0.0] * len(ep["boards"])
            for step_idx, (board, scalar, policy, mcts_val, completed_q,
                            root_visit, game_result, mask) in enumerate(zip(
                ep["boards"], ep["scalars"], ep["policies"], ep["mcts_values"],
                ep["completed_qs"], ep["root_visits"], ep["game_results"],
                ep["action_masks"],
            )):
                samples.append({
                    "board_state": list(board),
                    "scalar_state": list(scalar),
                    "policy_probs": list(policy),
                    "mcts_value": float(mcts_val),
                    "completed_q": float(completed_q),
                    "root_visit_count": int(root_visit),
                    "game_result_value": float(game_result),
                    "action_mask": list(mask),
                    "health_diff": float(health_diffs[step_idx]),
                })
            documents.append({
                "iteration": int(iteration),
                "game_length": int(ep["game_length"]),
                "winner": ep["winner"],
                "samples": samples,
                "timestamp": self._now_utc(),
            })
        if documents:
            self.collection.insert_many(documents)
            print(f"  [MongoSaver] 已保存 {len(documents)} 局 (iteration={iteration})")

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
