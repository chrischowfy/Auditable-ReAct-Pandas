from __future__ import annotations

import ast
import gzip
import hashlib
import json
import math
import os
import re
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

import fire
import pandas as pd
from tqdm import tqdm

from auditable_react_pandas import mmqa_eval_common as eval_common
from auditable_react_pandas.agent import model as model_module
from auditable_react_pandas.agent.react_minimal import MinimalReActAgent
from auditable_react_pandas.agent.model import Model


BEHAVIOR_CONFIG_EXCLUDE = {
    "code_version_hash",
    "config",
    "dataset",
    "load_exist",
    "log_dir",
    "n_worker",
    "enable_judge",
    "judge_model_name",
    "resume_from",
    "stop_at",
    "verbose",
}

JUDGE_PROMPT = """You are the Judge Agent. Your ONLY task is to determine whether two answers convey the SAME factual content.

Question: "{query}"

Agent's Answer: {agent_answer}
Official Label: {label}

Strict Rules:
1. If Agent's Answer is EMPTY, it is ALWAYS a MISMATCH.
2. If Agent's Answer contains "Error" or "Failed", it is ALWAYS a MISMATCH.
3. ALL key facts in the Official Label must appear in the Agent's Answer. A partial match is a MISMATCH.
4. The Agent's Answer must NOT contain contradictory facts compared to the Label.

Permissible Differences:
5. IGNORE formatting: brackets, quotes, extra whitespace, filler words.
6. IGNORE case.
7. IGNORE list ordering.
8. IGNORE number format: 115897 and 115897.0 are the SAME.

Output EXACTLY one of:
- "JUDGE: MATCH"
- "JUDGE: MISMATCH"

Do NOT explain. Output ONLY the judgment line.
"""


def load_dataset(dataset_path: str, stop_at: int = -1) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    tag = path.name
    for suffix in (".gz", ".json"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fp:
        rows = json.load(fp)
    if stop_at >= 0:
        rows = rows[:stop_at]
    dataset = []
    for i, item in enumerate(rows):
        item = dict(item)
        if "id" in item:
            item["orig_id"] = item["id"]
        elif "id_" in item:
            item["orig_id"] = item["id_"]
        item["id"] = f"{tag}-{i}"
        if "table_id" not in item:
            item["table_id"] = "_".join(item.get("table_names", [])) or f"mmqa_{i}"
        dataset.append(item)
    return dataset


def _normalize_atom(value: Any) -> str:
    return eval_common.normalize_atom(value)


def _answer_items(value: Any) -> List[Any]:
    return eval_common.answer_items(value)


def normalize_answer(value: Any) -> str:
    return eval_common.normalize_answer(value)


def render_answer_for_judge(value: Any) -> str:
    return eval_common.render_answer_for_judge(value)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return "unavailable"


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    files = [
        Path(__file__).resolve(),
        root / "agent" / "react_minimal.py",
        root / "agent" / "model.py",
    ]
    model_path = getattr(model_module, "_MODEL_PATH", None)
    if isinstance(model_path, Path):
        files.append(model_path.resolve())

    payload = {}
    for path in files:
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = str(path)
        payload[label] = _file_sha256(path)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def behavior_config_hash(config: Dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if key not in BEHAVIOR_CONFIG_EXCLUDE
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def evaluate_qa(results: Iterable[Dict[str, Any]]) -> float:
    preds = defaultdict(Counter)
    labels = {}
    for result in results:
        qid = result["id"]
        labels.setdefault(qid, normalize_answer(result.get("label", "")))
        preds[qid].update([normalize_answer(result.get("answer", ""))])
    if not labels:
        return 0.0
    correct = sum(preds[qid].most_common(1)[0][0] == label for qid, label in labels.items())
    return correct / len(labels)


def select_effective_results(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for result in results:
        qid = result["id"]
        if qid not in grouped:
            order.append(qid)
        grouped[qid].append(result)

    effective: List[Dict[str, Any]] = []
    for qid in order:
        candidates = grouped[qid]
        if len(candidates) == 1:
            effective.append(candidates[0])
            continue
        votes = Counter(normalize_answer(item.get("answer", "")) for item in candidates)
        winning_answer = votes.most_common(1)[0][0]
        selected = next(
            item for item in candidates
            if normalize_answer(item.get("answer", "")) == winning_answer
        )
        selected = dict(selected)
        selected["sc_raw_run_count"] = len(candidates)
        selected["sc_vote_count"] = votes[winning_answer]
        effective.append(selected)
    return effective


def _deterministic_judge_match(answer: str, label: str) -> bool | None:
    answer_text = str(answer).strip()
    label_text = str(label).strip()
    if not answer_text or answer_text.lower().startswith("error"):
        return False
    if normalize_answer(answer_text) == normalize_answer(label_text):
        return True
    try:
        answer_num = float(answer_text)
        label_num = float(label_text)
        return abs(answer_num - label_num) < 1e-9
    except Exception:
        return None


def evaluate_with_judge(results: List[Dict[str, Any]], model_name: str) -> float:
    if not results:
        return 0.0
    max_workers = max(1, int(os.environ.get("JUDGE_N_WORKER", "8")))
    max_tokens = max(1, int(os.environ.get("JUDGE_MAX_TOKENS", "32")))

    def judge_one(result: Dict[str, Any]) -> bool:
        deterministic = _deterministic_judge_match(result.get("answer", ""), result.get("label", ""))
        if deterministic is not None:
            return deterministic
        model = Model(model_name)
        prompt = JUDGE_PROMPT.format(
            query=result.get("query", ""),
            agent_answer=result.get("answer", ""),
            label=render_answer_for_judge(result.get("label", "")),
        )
        raw = model.query(prompt, temperature=0.0, top_p=1.0, stop=None, max_tokens=max_tokens)
        return "JUDGE: MATCH" in str(raw).upper()

    correct = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(judge_one, result) for result in results]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judge"):
            try:
                correct += int(bool(future.result()))
            except Exception as exc:
                print(f"Judge warning: treating failed judge call as mismatch: {type(exc).__name__}: {exc}")
    return correct / len(results)


def _executor_class():
    mode = str(os.environ.get("RUN_EXECUTOR", "process") or "process").strip().lower()
    if mode in {"thread", "threads", "threadpool"}:
        return ThreadPoolExecutor
    return ProcessPoolExecutor


def solve(args):
    agent_args, data, sc_id = args
    agent = MinimalReActAgent(**agent_args)
    return agent.run(data, sc_id=sc_id)


def _faithfulness_rate(results: List[Dict[str, Any]], key: str) -> float:
    values = []
    for result in results:
        faithfulness = result.get("faithfulness") or {}
        value = faithfulness.get(key)
        if isinstance(value, bool):
            values.append(value)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def main(
    dataset_path: str,
    model_name: str = "deepseek-v4-flash",
    judge_model_name: str = "",
    enable_judge: bool = False,
    agent_type: str = "MinimalReAct",
    retrieve_mode: str = "bm25",
    embed_model_name: str = "text-embedding-3-large",
    log_dir: str = "local_outputs/react_minimal",
    db_dir: str = "db/",
    top_k: int = 5,
    sr: int = 0,
    sc: int = 1,
    max_encode_cell: int = 1000,
    stop_at: int = -1,
    resume_from: int = 0,
    load_exist: bool = False,
    n_worker: int = 1,
    verbose: bool = False,
    max_steps_two: int = 4,
    max_steps_three: int = 6,
    enable_relation_hints: bool = True,
    relation_hints_min_tables: int = 3,
    enable_cell_evidence: bool = True,
    cell_evidence_top_k: int = 12,
    cell_evidence_max_value_len: int = 80,
    cell_evidence_include_rows: bool = True,
    enable_faithfulness_report: bool = True,
    faithfulness_mode: str = "warn_only",
    enable_projection_verifier: bool = True,
    projection_verifier_feedback: bool = False,
    enable_strict_recovery: bool = True,
    enable_one_shot_repair: bool = False,
    enable_safe_merge: bool = True,
    enable_static_checks: bool = True,
    enable_answer_contracts: bool = True,
    enable_schema_pruning: bool = True,
    schema_pruning_max_columns: int = 32,
    schema_pruning_head_rows: int = 4,
    enable_longtablebench_diagnostic_rules: bool = False,
    enable_candidate_selector: bool = False,
    candidate_selector_samples: int = 3,
    candidate_selector_min_frac: float = 0.67,
    candidate_selector_max_candidates: int = 24,
    candidate_selector_model_name: str = "",
    execution_timeout_seconds: int = 30,
):
    del db_dir
    candidate_selector_model_name = candidate_selector_model_name or model_name
    os.makedirs(os.path.join(log_dir, "log"), exist_ok=True)
    dataset = load_dataset(dataset_path, stop_at=stop_at)
    if stop_at < 0:
        stop_at = len(dataset)

    code_version_hash = code_fingerprint()
    config = {key: value for key, value in locals().items() if key not in {"dataset"}}
    run_config_hash = behavior_config_hash(config)
    config["run_config_hash"] = run_config_hash
    with open(os.path.join(log_dir, "config.json"), "w") as fp:
        json.dump(config, fp, ensure_ascii=False, indent=2)

    agent_args = {
        "model_name": model_name,
        "retrieve_mode": retrieve_mode,
        "embed_model_name": embed_model_name,
        "agent_type": agent_type,
        "top_k": top_k,
        "sr": sr,
        "max_encode_cell": max_encode_cell,
        "log_dir": log_dir,
        "load_exist": load_exist,
        "verbose": verbose,
        "max_steps_two": max_steps_two,
        "max_steps_three": max_steps_three,
        "enable_relation_hints": enable_relation_hints,
        "relation_hints_min_tables": relation_hints_min_tables,
        "enable_cell_evidence": enable_cell_evidence,
        "cell_evidence_top_k": cell_evidence_top_k,
        "cell_evidence_max_value_len": cell_evidence_max_value_len,
        "cell_evidence_include_rows": cell_evidence_include_rows,
        "enable_faithfulness_report": enable_faithfulness_report,
        "faithfulness_mode": faithfulness_mode,
        "enable_projection_verifier": enable_projection_verifier,
        "projection_verifier_feedback": projection_verifier_feedback,
        "enable_strict_recovery": enable_strict_recovery,
        "enable_one_shot_repair": enable_one_shot_repair,
        "enable_safe_merge": enable_safe_merge,
        "enable_static_checks": enable_static_checks,
        "enable_answer_contracts": enable_answer_contracts,
        "enable_schema_pruning": enable_schema_pruning,
        "schema_pruning_max_columns": schema_pruning_max_columns,
        "schema_pruning_head_rows": schema_pruning_head_rows,
        "enable_longtablebench_diagnostic_rules": enable_longtablebench_diagnostic_rules,
        "enable_candidate_selector": enable_candidate_selector,
        "candidate_selector_samples": candidate_selector_samples,
        "candidate_selector_min_frac": candidate_selector_min_frac,
        "candidate_selector_max_candidates": candidate_selector_max_candidates,
        "candidate_selector_model_name": candidate_selector_model_name or model_name,
        "execution_timeout_seconds": execution_timeout_seconds,
        "run_config_hash": run_config_hash,
    }

    results = []
    work_items = [
        (agent_args, data, sc_id)
        for data in dataset[resume_from:stop_at]
        for sc_id in range(sc)
    ]
    if n_worker == 1:
        for item in tqdm(work_items, desc="Run"):
            results.append(solve(item))
    else:
        executor_cls = _executor_class()
        with executor_cls(max_workers=n_worker) as executor:
            futures = [executor.submit(solve, item) for item in work_items]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Run"):
                results.append(future.result())

    effective_results = select_effective_results(results)
    acc = evaluate_qa(effective_results)
    print(f"Accuracy (string match): {acc}")
    effective_judge_model_name = judge_model_name or model_name
    if str(enable_judge).lower() in {"1", "true", "yes", "y"}:
        judge_acc = evaluate_with_judge(effective_results, model_name=effective_judge_model_name)
        print(f"Accuracy (Judge Agent):  {judge_acc}")
    else:
        judge_acc = None
        effective_judge_model_name = ""
        print("Accuracy (Judge Agent):  skipped")

    stats = pd.DataFrame.from_records(results)
    stat_keys = [key for key in ["n_iter", "init_prompt_token_count", "total_token_count"] if key in stats]
    result_dict = stats[stat_keys].mean().to_dict() if stat_keys else {}
    schema_text_chars = []
    hidden_columns = []
    selected_columns = []
    for result in results:
        diag = result.get("schema_pruning") or {}
        if isinstance(diag.get("schema_text_chars"), (int, float)):
            schema_text_chars.append(diag["schema_text_chars"])
        table_diags = diag.get("tables") or []
        if table_diags:
            hidden_columns.append(sum(int(table.get("hidden_column_count") or 0) for table in table_diags))
            selected_columns.append(sum(int(table.get("selected_column_count") or 0) for table in table_diags))

    def _avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    result_dict.update({
        "accuracy": acc,
        "judge_accuracy": judge_acc,
        "model_name": model_name,
        "enable_judge": bool(str(enable_judge).lower() in {"1", "true", "yes", "y"}),
        "judge_model_name": effective_judge_model_name,
        "code_version_hash": code_version_hash,
        "run_config_hash": run_config_hash,
        "retrieve_mode": retrieve_mode,
        "embed_model_name": embed_model_name,
        "task": "mmqa",
        "agent_type": agent_type,
        "top_k": top_k,
        "max_encode_cell": max_encode_cell,
        "sr": sr,
        "sc": sc,
        "max_steps_two": max_steps_two,
        "max_steps_three": max_steps_three,
        "enable_relation_hints": enable_relation_hints,
        "relation_hints_min_tables": relation_hints_min_tables,
        "enable_cell_evidence": enable_cell_evidence,
        "cell_evidence_top_k": cell_evidence_top_k,
        "cell_evidence_max_value_len": cell_evidence_max_value_len,
        "cell_evidence_include_rows": cell_evidence_include_rows,
        "enable_faithfulness_report": enable_faithfulness_report,
        "faithfulness_mode": faithfulness_mode,
        "enable_projection_verifier": enable_projection_verifier,
        "projection_verifier_feedback": projection_verifier_feedback,
        "enable_strict_recovery": enable_strict_recovery,
        "enable_one_shot_repair": enable_one_shot_repair,
        "enable_safe_merge": enable_safe_merge,
        "enable_static_checks": enable_static_checks,
        "enable_answer_contracts": enable_answer_contracts,
        "enable_schema_pruning": enable_schema_pruning,
        "schema_pruning_max_columns": schema_pruning_max_columns,
        "schema_pruning_head_rows": schema_pruning_head_rows,
        "schema_pruning_avg_schema_text_chars": _avg(schema_text_chars),
        "schema_pruning_avg_hidden_columns": _avg(hidden_columns),
        "schema_pruning_avg_selected_columns": _avg(selected_columns),
        "enable_longtablebench_diagnostic_rules": enable_longtablebench_diagnostic_rules,
        "enable_candidate_selector": enable_candidate_selector,
        "candidate_selector_samples": candidate_selector_samples,
        "candidate_selector_min_frac": candidate_selector_min_frac,
        "candidate_selector_max_candidates": candidate_selector_max_candidates,
        "candidate_selector_model_name": candidate_selector_model_name or model_name,
        "execution_timeout_seconds": execution_timeout_seconds,
        "faithfulness_pass_rate": sum(
            1 for result in effective_results
            if (result.get("faithfulness") or {}).get("status") == "PASS"
        ) / len(effective_results) if effective_results else 0.0,
        "answer_from_execution_rate": _faithfulness_rate(effective_results, "answer_from_execution"),
        "entity_anchor_supported_rate": _faithfulness_rate(effective_results, "entity_anchor_supported"),
        "join_path_supported_rate": _faithfulness_rate(effective_results, "join_path_supported"),
        "projection_verified_rate": _faithfulness_rate(effective_results, "projection_verified"),
        "projection_complete_rate": _faithfulness_rate(effective_results, "projection_complete"),
        "operation_verified_rate": _faithfulness_rate(effective_results, "operation_verified"),
        "raw_run_count": len(results),
        "effective_result_count": len(effective_results),
        "data": Path(dataset_path).stem,
    })
    with open(os.path.join(log_dir, "result.json"), "w") as fp:
        json.dump(result_dict, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.argv = [arg for arg in sys.argv if arg != "-"]
    fire.Fire(main)
