from __future__ import annotations

import ast
import contextlib
import difflib
import io
import json
import math
import os
import re
import signal
import threading
import traceback
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from auditable_react_pandas.agent.model import Model


NULL_TEXT = {"", "none", "nan", "null", "nat"}
PROJECTION_ROW_LIMIT = 1000

CELL_EVIDENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "find",
    "for",
    "from",
    "give",
    "had",
    "has",
    "have",
    "in",
    "is",
    "list",
    "of",
    "on",
    "or",
    "show",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}

DANGEROUS_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}

DANGEROUS_MODULE_NAMES = {
    "builtins",
    "importlib",
    "io",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}

DANGEROUS_ATTRS = {
    "DataSource",
    "NpzFile",
    "ctypeslib",
    "dump",
    "dumps",
    "eval",
    "fromfile",
    "fromregex",
    "genfromtxt",
    "io",
    "load",
    "loadtxt",
    "loads",
    "memmap",
    "open_memmap",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "save",
    "savez",
    "savez_compressed",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "tofile",
    "to_hdf",
    "to_json",
    "to_latex",
    "to_markdown",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_stata",
}


@contextlib.contextmanager
def execution_time_limit(seconds: int):
    seconds = int(seconds or 0)
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"generated code exceeded {seconds}s execution limit")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer and old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])

def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"0", "false", "no", "off", ""}


def infer_dtype(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
                if df[col].dtype == "object":
                    df[col] = pd.to_datetime(df[col], errors="raise")
            except Exception:
                pass
    return df


def safe_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: Optional[Any] = None,
    right_on: Optional[Any] = None,
    how: str = "inner",
    on: Optional[Any] = None,
    **kwargs,
) -> pd.DataFrame:
    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("safe_merge received both `on` and `left_on`/`right_on`.")
        left_on = right_on = on
    if left_on is None or right_on is None:
        raise ValueError("safe_merge requires either `on` or both `left_on` and `right_on`.")
    left = left_df.copy()
    right = right_df.copy()

    def key_list(keys: Any) -> List[Any]:
        if isinstance(keys, (list, tuple)):
            return list(keys)
        return [keys]

    left_keys = key_list(left_on)
    right_keys = key_list(right_on)
    if len(left_keys) != len(right_keys):
        raise ValueError("safe_merge requires the same number of left and right join keys.")
    for left_key, right_key in zip(left_keys, right_keys):
        if left_key in left.columns and right_key in right.columns:
            left[left_key] = left[left_key].astype(str)
            right[right_key] = right[right_key].astype(str)
    return pd.merge(left, right, left_on=left_on, right_on=right_on, how=how, **kwargs)


def normalize_answer_text(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("[]")
    return text


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in NULL_TEXT


def is_id_like_column(col: str) -> bool:
    raw = str(col).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", raw)
    return (
        compact == "id"
        or compact.endswith("id")
        or compact.endswith("code")
        or compact in {"no", "number", "key"}
    )


def requested_identifier_terms(query: str) -> List[str]:
    q = str(query or "").lower()
    answer_head = re.split(
        r"\b(?:of|with|whose|having|where|that|who|from|under|for|associated|have|has)\b",
        q,
        maxsplit=1,
    )[0]
    stop_terms = {"the", "a", "an", "and", "or", "what", "which", "who", "list", "show", "find", "give", "provide", "id", "ids"}
    terms = []
    for match in re.finditer(r"\b([a-z][a-z0-9]*)_ids?\b", answer_head):
        term = match.group(1)
        if term not in stop_terms:
            terms.append(term)
    for match in re.finditer(r"\b([a-z][a-z0-9]*)\s+ids?\b", answer_head):
        term = match.group(1)
        if term not in stop_terms:
            terms.append(term)
    for match in re.finditer(r"\bids?\s+(?:for|of)\s+(?:the\s+)?([a-z][a-z0-9]*)", q):
        term = match.group(1)
        if term not in stop_terms:
            terms.append(term)
    if not terms and re.search(r"\bids?\b", answer_head):
        terms.append("id")
    return dedupe_keep_order(terms)


def identifier_column_matches_query(query: str, col: str) -> bool:
    terms = requested_identifier_terms(query)
    if not terms:
        return False
    col_compact = compact_name(col)
    for term in terms:
        term_compact = compact_name(term)
        variants = {term_compact}
        if term_compact.endswith("ies"):
            variants.add(term_compact[:-3] + "y")
        if term_compact.endswith("s"):
            variants.add(term_compact[:-1])
        variants.add(term_compact.replace("organization", "organisation"))
        variants.add(term_compact.replace("organisation", "organization"))
        if term_compact == "faculty":
            variants.add("fac")
        if any(variant and variant in col_compact for variant in variants):
            return True
    return False


def query_asks_identifier(query: str) -> bool:
    return bool(requested_identifier_terms(query))


def is_number_like_value(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)) and not is_empty_value(value):
        return True
    text = str(value).strip().replace(",", "")
    if not text:
        return False
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text))


def split_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def compact_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def normalize_join_value(value: Any) -> str:
    text = normalize_answer_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def column_kind(col: str) -> str:
    raw = str(col).lower()
    compact = compact_name(raw)
    if compact in {"id", "idx", "index"} or compact.endswith("id"):
        return "id"
    if compact.endswith("code") or "code" in compact:
        return "code"
    if "name" in compact:
        return "name"
    if "title" in compact:
        return "title"
    if "date" in compact:
        return "date"
    if "year" in compact:
        return "year"
    if "key" in compact:
        return "key"
    return ""


SLOT_TEMPLATES: List[Dict[str, Any]] = [
    {"name": "name", "kind": "text", "cues": ["name", "names", "first name", "last name", "person", "people", "who"]},
    {"name": "email", "kind": "email", "cues": ["email", "e-mail", "mail"]},
    {"name": "address", "kind": "address", "cues": ["address", "addresses", "street address", "location"]},
    {"name": "phone", "kind": "phone", "cues": ["phone", "phone number", "telephone", "mobile", "cell"]},
    {"name": "date", "kind": "date", "cues": ["date", "year", "day", "month", "when"]},
    {"name": "time", "kind": "time", "cues": ["time", "hour", "start time", "end time"]},
    {"name": "room", "kind": "room", "cues": ["room", "classroom"]},
    {"name": "title", "kind": "title", "cues": ["title", "book", "movie", "film", "song", "album"]},
    {"name": "institution", "kind": "institution", "cues": ["institution", "university", "school", "college"]},
    {"name": "department", "kind": "department", "cues": ["department", "departments", "dept", "division"]},
    {"name": "city", "kind": "city", "cues": ["city", "cities", "town"]},
    {"name": "country", "kind": "country", "cues": ["country", "nation"]},
    {"name": "state", "kind": "state", "cues": ["state", "province", "state province", "state provinces"]},
    {"name": "count", "kind": "number", "cues": ["count", "counts", "number", "how many"]},
    {"name": "metric", "kind": "number", "cues": ["total", "sum", "average", "avg", "quantity", "amount", "population", "sales", "enrollment"]},
    {"name": "detail", "kind": "text", "cues": ["detail", "details", "description", "value", "answer"]},
]

SLOT_TEMPLATE_BY_NAME = {slot["name"]: slot for slot in SLOT_TEMPLATES}


def _text_contains_cue(text: str, cue: str) -> bool:
    text_l = f" {str(text or '').lower()} "
    cue_l = str(cue or "").lower().strip()
    if not cue_l:
        return False
    if " " in cue_l:
        return cue_l in text_l
    return bool(re.search(rf"\b{re.escape(cue_l)}s?\b", text_l))


def _answer_focus_text(query: str) -> str:
    q = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    q = q.replace("e mail", "email")
    q = re.sub(
        r"^(?:please\s+)?(?:what|which|who|where|when|list|show|give|find|provide)\s+",
        "",
        q,
    )
    q = re.sub(r"^(?:is|are|was|were|do|does|did)\s+", "", q)
    q = re.sub(r"^(?:the|all|both)\s+", "", q)

    split_patterns = [
        r"\b(?:of|for|whose|who|that|where|with|from|under|associated with|having|marked|running|opened|after|before|during|without)\b",
        r"\b(?:is|are|was|were)\b",
        r"\b(?:has|have|had)\s+(?:the\s+)?(?:highest|lowest|largest|smallest|maximum|minimum|most|least)\b",
    ]
    cut = len(q)
    for pattern in split_patterns:
        match = re.search(pattern, q)
        if match and match.start() > 0:
            cut = min(cut, match.start())
    focus = q[:cut].strip()
    return focus or q


def _rank_subject_question(query: str) -> bool:
    q = f" {str(query or '').lower()} "
    if not re.search(r"\b(highest|lowest|largest|smallest|maximum|minimum|most|least|greatest|fewest)\b", q):
        return False
    return bool(
        re.search(r"\b(?:which|who)\b", q)
        or re.search(
            r"\bwhat\s+(?:(?:is|are|was|were)\s+)?(?:the\s+)?"
            r"(?:name|person|player|student|customer|employee|staff|asset|company|"
            r"department|institution|university|school|college|city|country|title|room)\b",
            q,
        )
    )


def _numeric_extreme_value_question(query: str) -> bool:
    q = f" {str(query or '').lower()} "
    if _rank_subject_question(query):
        return False
    if not re.search(r"\b(highest|lowest|largest|smallest|maximum|minimum|max|min|most|least|greatest|fewest|shortest|longest)\b", q):
        return False
    return bool(
        re.search(
            r"\b(price|speed|length|duration|salary|amount|quantity|total|sum|score|rate|"
            r"credits?|population|enrollment|lap|cost|revenue|sales|value|number|count)\b",
            q,
        )
    )


def _slot_from_template(name: str, required: bool = True, extra_cues: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    template = dict(SLOT_TEMPLATE_BY_NAME.get(name, {"name": name, "kind": "text", "cues": [name]}))
    cues = list(template.get("cues") or [])
    cues.extend(str(cue) for cue in (extra_cues or []) if cue)
    template["cues"] = dedupe_keep_order(cues)
    template["required"] = bool(required)
    return template


def _slot_key(slot: Mapping[str, Any]) -> str:
    return str(slot.get("name") or slot.get("kind") or "answer").lower()


def _add_slot(slots: List[Dict[str, Any]], name: str, required: bool = True, extra_cues: Optional[Sequence[str]] = None) -> None:
    new_slot = _slot_from_template(name, required=required, extra_cues=extra_cues)
    key = _slot_key(new_slot)
    for slot in slots:
        if _slot_key(slot) == key:
            slot["required"] = bool(slot.get("required", True) or required)
            slot["cues"] = dedupe_keep_order(list(slot.get("cues") or []) + list(new_slot.get("cues") or []))
            return
    slots.append(new_slot)


def infer_required_output_slots(
    query: str,
    answer_type: str,
    operation: str,
    target_terms: Sequence[str],
) -> List[Dict[str, Any]]:
    q = f" {str(query or '').lower()} "
    focus = _answer_focus_text(query)
    numeric_output_requested = bool(
        re.search(r"\b(?:count|counts|number|numbers|total|sum|average|avg|quantity|amount)\b", focus)
        or re.search(
            r"\b(?:also|along with|together with|as well as|including)\b.{0,40}"
            r"\b(?:count|counts|number|numbers|total|sum|average|avg|quantity|amount)\b",
            q,
        )
    )
    asks_location = bool(re.match(r"\s*where\b", str(query or "").lower()))
    slots: List[Dict[str, Any]] = []

    identifier_terms = requested_identifier_terms(query)
    for term in identifier_terms:
        if term == "id":
            cues = ["id", "ids"]
            _add_slot(slots, "id", required=True, extra_cues=cues)
        else:
            cues = [f"{term} id", f"{term}_id", term, "id"]
            _add_slot(slots, f"{term}_id", required=True, extra_cues=cues)
        slots[-1]["kind"] = "id"

    explicit_names = [
        "email",
        "address",
        "phone",
        "time",
        "room",
        "detail",
        "department",
        "institution",
        "city",
        "country",
        "state",
        "title",
    ]
    for name in explicit_names:
        template = SLOT_TEMPLATE_BY_NAME[name]
        if name == "time" and re.search(r"\bhow many times\b", q):
            continue
        if name == "address" and re.search(r"\b(?:email|e-mail|mail)\s+address(?:es)?\b", focus):
            continue
        if any(_text_contains_cue(focus, cue) for cue in template["cues"]):
            _add_slot(slots, name)

    if answer_type == "date" or re.search(r"\b(?:what|which|when|date|year)\b.{0,20}\b(?:date|year|when)\b", q):
        if "opened from year" not in q and "from year" not in q:
            _add_slot(slots, "date")

    if any(_text_contains_cue(focus, cue) for cue in ["name", "names", "first name", "last name"]):
        _add_slot(slots, "name")

    subject_terms = {
        "person",
        "people",
        "player",
        "student",
        "coach",
        "member",
        "employee",
        "staff",
        "customer",
        "candidate",
        "asset",
        "company",
        "medicine",
        "medicines",
        "drug",
        "drugs",
    }
    if answer_type != "number" and not asks_location and not identifier_terms and (
        re.search(r"\bwho\b", q)
        or re.search(r"\bwhich\s+[a-z0-9_ -]+", q)
        or any(_text_contains_cue(focus, term) for term in subject_terms)
    ):
        if not any(_slot_key(slot) in {"name", "department", "institution", "city", "country", "state", "title", "room"} for slot in slots):
            _add_slot(slots, "name", extra_cues=target_terms)

    if asks_location and not any(slot.get("required", True) for slot in slots):
        _add_slot(slots, "detail", extra_cues=["location", "located", "address", "place"])

    if answer_type == "email":
        _add_slot(slots, "email")
    elif answer_type == "number" and not (operation == "rank" and slots and not numeric_output_requested):
        slot_name = "count" if operation == "count" else "metric"
        _add_slot(slots, slot_name, extra_cues=["value"])

    if operation == "rank" and answer_type != "number":
        if any(_text_contains_cue(q, cue) for cue in SLOT_TEMPLATE_BY_NAME["count"]["cues"]):
            _add_slot(slots, "count", required=False, extra_cues=["rank metric"])
        if any(_text_contains_cue(q, cue) for cue in SLOT_TEMPLATE_BY_NAME["metric"]["cues"]):
            _add_slot(slots, "metric", required=False, extra_cues=["rank metric"])

    required_slots = [slot for slot in slots if slot.get("required", True)]
    if not required_slots:
        if answer_type in {"text", "email", "date"}:
            focus_terms = [token for token in split_words(focus) if token not in CELL_EVIDENCE_STOPWORDS]
            _add_slot(slots, "detail", extra_cues=focus_terms[:4] + list(target_terms))
        elif answer_type == "number":
            _add_slot(slots, "metric", extra_cues=["value"])
    order_text = f"{focus} {q}"
    original_positions = {id(slot): idx for idx, slot in enumerate(slots)}

    def slot_position(slot: Mapping[str, Any]) -> Tuple[int, int]:
        positions = []
        for cue in [slot.get("name", ""), *(slot.get("cues") or [])]:
            cue_text = str(cue).lower().strip()
            if cue_text:
                pos = order_text.find(cue_text)
                if pos >= 0:
                    positions.append(pos)
        return (min(positions) if positions else 10_000, original_positions[id(slot)])

    slots.sort(key=slot_position)
    return slots


def dedupe_keep_order(values: Iterable[Any]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        text = normalize_answer_text(value)
        if not text or text.lower() in NULL_TEXT:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def extract_quoted_phrases(text: str) -> List[str]:
    phrases = []
    for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", str(text)):
        phrase = match.group(1) if match.group(1) is not None else match.group(2)
        phrase = normalize_answer_text(phrase)
        if phrase:
            phrases.append(phrase)
    return dedupe_keep_order(phrases)


def content_tokens(text: str) -> List[str]:
    tokens = []
    for token in split_words(str(text)):
        if token in CELL_EVIDENCE_STOPWORDS:
            continue
        if len(token) > 1 or token.isdigit():
            tokens.append(token)
    return tokens


def truncate_text(value: Any, limit: int = 80) -> str:
    text = normalize_answer_text(value)
    limit = max(8, int(limit))
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def truth_status(values: Mapping[str, Any], keys: Sequence[str]) -> str:
    failures = [key for key in keys if values.get(key) is False]
    if failures:
        return "FAIL"
    warnings = [key for key in keys if values.get(key) is None]
    return "WARN" if warnings else "PASS"


@dataclass
class AnswerContract:
    answer_type: str = "text"
    target_terms: List[str] = field(default_factory=list)
    required_output_slots: List[Dict[str, Any]] = field(default_factory=list)
    operation: str = "lookup"
    requires_single: bool = False
    multi_field: bool = False
    max_steps: int = 4

    def to_prompt_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepObservation:
    step_id: int
    action: str
    thought: str = ""
    code: str = ""
    status: str = "ERROR"
    error: str = ""
    stdout: str = ""
    structured_result: Dict[str, Any] = field(default_factory=dict)
    contract_status: str = "FAIL"
    contract_error: str = ""

    def to_log(self) -> Dict[str, Any]:
        def json_safe(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                safe = {}
                for k, v in value.items():
                    key = str(k)
                    if key == "projection_rows":
                        safe["projection_row_count"] = len(v) if isinstance(v, list) else 0
                        continue
                    safe[key] = json_safe(v)
                return safe
            if isinstance(value, (list, tuple, set)):
                return [json_safe(v) for v in value]
            return str(value)

        return {
            "step_id": self.step_id,
            "action": self.action,
            "thought": self.thought,
            "code": self.code,
            "status": self.status,
            "error": self.error,
            "stdout": self.stdout,
            "structured_result": json_safe(self.structured_result),
            "contract_status": self.contract_status,
            "contract_error": self.contract_error,
        }


class MinimalReActAgent:
    """A small MMQA ReAct agent with deterministic execution checks.

    The LLM proposes only the next local pandas action. Planning, contract
    checks, execution, and final projection are local and traceable.
    """

    def __init__(
        self,
        model_name: str,
        retrieve_mode: str = "bm25",
        embed_model_name: Optional[str] = None,
        task: str = "mmqa",
        agent_type: str = "MinimalReAct",
        top_k: int = 5,
        sr: int = 0,
        max_encode_cell: int = 1000,
        load_exist: bool = False,
        log_dir: Optional[str] = None,
        db_dir: Optional[str] = None,
        verbose: bool = False,
        max_steps_two: int = 4,
        max_steps_three: int = 6,
        max_tokens: int = 1200,
        temperature: float = 0.0,
        top_p: float = 1.0,
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
        run_config_hash: Optional[str] = None,
        **_: Any,
    ):
        del retrieve_mode, embed_model_name, task, top_k, sr, max_encode_cell, db_dir
        self.model_name = model_name
        self.agent_type = agent_type
        self.load_exist = load_exist
        self.log_dir = log_dir or "output/react_minimal"
        self.verbose = verbose
        self.max_steps_two = max_steps_two
        self.max_steps_three = max_steps_three
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_relation_hints = as_bool(enable_relation_hints)
        self.relation_hints_min_tables = max(2, int(relation_hints_min_tables))
        self.enable_cell_evidence = as_bool(enable_cell_evidence)
        self.cell_evidence_top_k = max(0, int(cell_evidence_top_k))
        self.cell_evidence_max_value_len = max(16, int(cell_evidence_max_value_len))
        self.cell_evidence_include_rows = as_bool(cell_evidence_include_rows)
        self.enable_faithfulness_report = as_bool(enable_faithfulness_report)
        self.faithfulness_mode = str(faithfulness_mode or "warn_only")
        self.enable_projection_verifier = as_bool(enable_projection_verifier)
        self.projection_verifier_feedback = as_bool(projection_verifier_feedback)
        self.enable_strict_recovery = as_bool(enable_strict_recovery)
        self.enable_one_shot_repair = as_bool(enable_one_shot_repair)
        self.enable_safe_merge = as_bool(enable_safe_merge)
        self.enable_static_checks = as_bool(enable_static_checks)
        self.enable_answer_contracts = as_bool(enable_answer_contracts)
        self.enable_schema_pruning = as_bool(enable_schema_pruning)
        self.schema_pruning_max_columns = max(1, int(schema_pruning_max_columns))
        self.schema_pruning_head_rows = max(1, int(schema_pruning_head_rows))
        self.enable_longtablebench_diagnostic_rules = as_bool(enable_longtablebench_diagnostic_rules)
        self.enable_candidate_selector = as_bool(enable_candidate_selector)
        self.candidate_selector_samples = max(1, int(candidate_selector_samples))
        self.candidate_selector_min_frac = max(0.0, min(1.0, float(candidate_selector_min_frac)))
        self.candidate_selector_max_candidates = max(1, int(candidate_selector_max_candidates))
        self.candidate_selector_model_name = str(candidate_selector_model_name or model_name)
        self._candidate_selector_model: Optional[Model] = None
        self.execution_timeout_seconds = max(0, int(execution_timeout_seconds))
        self.run_config_hash = str(run_config_hash or "")
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self._schema_pruning_diagnostics: Dict[str, Any] = {}
        self.model = Model(model_name)

    def _projection_feedback_enabled(self) -> bool:
        return bool(self.enable_projection_verifier and self.projection_verifier_feedback)

    def _query_agent(self, prompt: str) -> str:
        input_tokens = self.model.get_token_count(prompt)
        self.total_input_token_count += input_tokens
        if input_tokens > self.model.context_limit:
            return json.dumps({
                "thought": "Prompt is too long.",
                "action": "finish",
                "answer": "",
            })
        raw = self.model.query(
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=None,
            max_tokens=self.max_tokens,
        )
        self.total_output_token_count += self.model.get_token_count(raw)
        return raw

    @staticmethod
    def extract_query(data: Mapping[str, Any]) -> str:
        for key in ("Question", "question", "statement", "query"):
            value = data.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def extract_label(data: Mapping[str, Any]) -> str:
        for key in ("label", "answer", "gold_answer"):
            if key in data:
                return str(data.get(key, ""))
        return ""

    def _schema_cues(self, query: str, contract: Optional[AnswerContract]) -> List[str]:
        cues = content_tokens(query)
        if contract is not None:
            cues.extend(content_tokens(" ".join(contract.target_terms)))
            cues.append(str(contract.answer_type))
            cues.append(str(contract.operation))
            for slot in contract.required_output_slots:
                cues.extend(content_tokens(slot.get("name", "")))
                cues.extend(content_tokens(" ".join(str(cue) for cue in slot.get("cues", []))))
                cues.extend(content_tokens(slot.get("kind", "")))
        return dedupe_keep_order(cues)

    @staticmethod
    def _anchor_columns_for_df(
        df_name: str,
        relation_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
        cell_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, str]:
        columns: Dict[str, str] = {}
        for anchor in cell_anchors or []:
            if anchor.get("df") == df_name and anchor.get("column"):
                columns[str(anchor["column"])] = "cell-evidence"
        for anchor in relation_anchors or []:
            if anchor.get("left_df") == df_name and anchor.get("left_col"):
                columns[str(anchor["left_col"])] = "relation"
            if anchor.get("right_df") == df_name and anchor.get("right_col"):
                columns[str(anchor["right_col"])] = "relation"
        return columns

    def _score_schema_column(
        self,
        column: str,
        table_name: str,
        query_cues: Sequence[str],
        contract: Optional[AnswerContract],
        anchor_reason: str = "",
    ) -> Tuple[float, List[str]]:
        col_l = str(column).lower()
        compact_col = compact_name(column)
        col_tokens = set(content_tokens(column))
        table_tokens = set(content_tokens(table_name))
        score = 0.0
        reasons: List[str] = []

        if anchor_reason:
            score += 80.0 if anchor_reason == "cell-evidence" else 55.0
            reasons.append(anchor_reason)

        cue_hits = []
        for cue in query_cues:
            cue_l = str(cue).lower()
            cue_compact = compact_name(cue_l)
            if not cue_l:
                continue
            if cue_l.isdigit():
                matched = cue_l in col_tokens
            else:
                matched = cue_l in col_tokens or _text_contains_cue(col_l, cue_l) or (cue_compact and cue_compact in compact_col)
            if matched:
                cue_hits.append(cue_l)
        if cue_hits:
            score += 9.0 * len(set(cue_hits))
            reasons.append("query/contract:" + ",".join(sorted(set(cue_hits))[:4]))

        if table_tokens & col_tokens:
            score += 2.0 * len(table_tokens & col_tokens)
            reasons.append("table-token")
        if is_id_like_column(column):
            score += 7.0
            reasons.append("id-like")
        if column_kind(column):
            score += 4.0
            reasons.append("key-like")

        if contract is not None:
            answer_type = str(contract.answer_type)
            operation = str(contract.operation)
            if answer_type in {"email", "phone", "date", "time"} and answer_type in compact_col:
                score += 14.0
                reasons.append(f"answer-type:{answer_type}")
            if "name" in contract.target_terms and any(cue in compact_col for cue in ["name", "title", "detail"]):
                score += 10.0
                reasons.append("answer-name")
            if operation in {"rank", "avg", "sum", "count"} and any(
                cue in compact_col for cue in ["date", "year", "count", "total", "amount", "score", "rank", "value", "number"]
            ):
                score += 6.0
                reasons.append(f"operation:{operation}")

        return score, reasons

    def _select_schema_columns(
        self,
        dfs: Mapping[str, pd.DataFrame],
        table_names: Sequence[str],
        query: str = "",
        contract: Optional[AnswerContract] = None,
        relation_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
        cell_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
        query_cues = self._schema_cues(query, contract)
        selected_by_df: Dict[str, List[str]] = {}
        diagnostics: List[Dict[str, Any]] = []
        for idx, (df_name, df) in enumerate(dfs.items()):
            table_name = table_names[idx] if idx < len(table_names) else df_name
            columns = [str(col) for col in df.columns]
            max_columns = self.schema_pruning_max_columns
            pruning_needed = self.enable_schema_pruning and len(columns) > max_columns
            anchor_columns = self._anchor_columns_for_df(df_name, relation_anchors, cell_anchors)

            if not pruning_needed:
                selected = columns
                reasons_by_col = {col: ["all-columns"] for col in selected}
            else:
                scored = []
                reasons_by_col = {}
                for pos, col in enumerate(columns):
                    score, reasons = self._score_schema_column(
                        col,
                        table_name,
                        query_cues,
                        contract,
                        anchor_reason=anchor_columns.get(col, ""),
                    )
                    reasons_by_col[col] = reasons
                    scored.append((score, pos, col))
                scored.sort(key=lambda item: (-item[0], item[1]))
                selected = [col for _, _, col in scored[:max_columns]]
                if not selected:
                    selected = columns[:max_columns]

            reveal_candidates: List[str] = []
            if pruning_needed:
                selected_set = set(selected)
                hidden_scored = []
                for pos, col in enumerate(columns):
                    if col in selected_set:
                        continue
                    score, _ = self._score_schema_column(
                        col,
                        table_name,
                        query_cues,
                        contract,
                        anchor_reason=anchor_columns.get(col, ""),
                    )
                    if score > 0:
                        hidden_scored.append((score, pos, col))
                hidden_scored.sort(key=lambda item: (-item[0], item[1]))
                reveal_candidates = [col for _, _, col in hidden_scored[:8]]

            selected_by_df[df_name] = selected
            diagnostics.append({
                "df": df_name,
                "table": table_name,
                "total_columns": len(columns),
                "selected_columns": selected,
                "selected_column_count": len(selected),
                "hidden_column_count": max(0, len(columns) - len(selected)),
                "revealed_hidden_columns": reveal_candidates,
                "pruned": bool(pruning_needed),
                "reasons": {col: reasons_by_col.get(col, []) for col in selected[:max_columns]},
            })
        return selected_by_df, diagnostics

    def _render_schema_text(
        self,
        dfs: Mapping[str, pd.DataFrame],
        table_names: Sequence[str],
        query: str = "",
        contract: Optional[AnswerContract] = None,
        relation_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
        cell_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        selected_by_df, diagnostics = self._select_schema_columns(
            dfs=dfs,
            table_names=table_names,
            query=query,
            contract=contract,
            relation_anchors=relation_anchors,
            cell_anchors=cell_anchors,
        )
        schema_lines = []
        for idx, (var, df) in enumerate(dfs.items()):
            name = table_names[idx] if idx < len(table_names) else f"Table_{idx}"
            selected = selected_by_df[var]
            hidden = max(0, len(df.columns) - len(selected))
            header = f"- {name} ({var}): {selected}"
            if hidden:
                header += f" [schema-pruned: {hidden} hidden columns available in full DataFrame]"
            schema_lines.append(header)
            revealed = diagnostics[idx].get("revealed_hidden_columns", []) if idx < len(diagnostics) else []
            if revealed:
                schema_lines.append(f"  Query-relevant hidden columns: {revealed}")
            sample = df.loc[:, selected].head(self.schema_pruning_head_rows).astype(object).where(
                pd.notnull(df.loc[:, selected].head(self.schema_pruning_head_rows)),
                None,
            )
            schema_lines.append(sample.to_markdown(index=False))
        schema_text = "\n".join(schema_lines)
        self._schema_pruning_diagnostics = {
            "enabled": bool(self.enable_schema_pruning),
            "max_columns": int(self.schema_pruning_max_columns),
            "head_rows": int(self.schema_pruning_head_rows),
            "schema_text_chars": len(schema_text),
            "tables": diagnostics,
        }
        return schema_text

    def _build_failure_schema_reveal(
        self,
        query: str,
        dfs: Mapping[str, pd.DataFrame],
        table_names: Sequence[str],
        contract: AnswerContract,
        observations: Sequence[StepObservation],
    ) -> str:
        if not observations or not self.enable_schema_pruning:
            return ""
        last = observations[-1]
        reason = " ".join(str(x or "") for x in [last.error, last.contract_error]).lower()
        if not any(cue in reason for cue in ["missing", "not in columns", "empty", "no plausible", "projection", "ranking", "contract"]):
            return ""

        diagnostic_tables = {
            str(table.get("df")): table
            for table in (self._schema_pruning_diagnostics.get("tables") or [])
            if isinstance(table, Mapping)
        }
        slot_text = " ".join(
            " ".join([str(slot.get("name", "")), *[str(c) for c in slot.get("cues", [])]])
            for slot in contract.required_output_slots
        )
        cues = set(content_tokens(f"{query} {reason} {slot_text} {' '.join(contract.target_terms)}"))
        if not cues:
            return ""

        lines = ["Additional hidden columns revealed after failed step:"]
        any_revealed = False
        for idx, (df_name, df) in enumerate(dfs.items()):
            diag = diagnostic_tables.get(df_name, {})
            selected = set(str(col) for col in diag.get("selected_columns", []))
            hidden = [str(col) for col in df.columns if str(col) not in selected]
            scored: List[Tuple[float, int, str]] = []
            for pos, col in enumerate(hidden):
                col_tokens = set(content_tokens(col))
                overlap = cues & col_tokens
                compact_col = compact_name(col)
                compact_hits = [cue for cue in cues if cue and cue in compact_col]
                score = len(overlap) * 3.0 + len(compact_hits) * 1.5
                if score > 0:
                    scored.append((score, pos, col))
            scored.sort(key=lambda item: (-item[0], item[1]))
            revealed = [col for _, _, col in scored[:12]]
            if revealed:
                table_name = table_names[idx] if idx < len(table_names) else df_name
                lines.append(f"- {table_name} ({df_name}): {revealed}")
                any_revealed = True
        return "\n".join(lines) if any_revealed else ""

    @staticmethod
    def _append_schema_reveal(schema_text: str, reveal_text: str) -> str:
        if not reveal_text:
            return schema_text
        if reveal_text in schema_text:
            return schema_text
        return f"{schema_text}\n\n{reveal_text}"

    @staticmethod
    def _project_prompt_dfs(
        dfs: Mapping[str, pd.DataFrame],
        selected_by_df: Mapping[str, Sequence[str]],
    ) -> Dict[str, pd.DataFrame]:
        projected = {}
        for df_name, df in dfs.items():
            selected = [col for col in selected_by_df.get(df_name, []) if col in df.columns]
            projected[df_name] = df.loc[:, selected] if selected else df.iloc[:, :0]
        return projected

    def _build_dfs(
        self,
        data: Mapping[str, Any],
        query: str = "",
        contract: Optional[AnswerContract] = None,
        relation_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
        cell_anchors: Optional[Sequence[Mapping[str, Any]]] = None,
        build_schema: bool = True,
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
        dfs = {}
        for i, table in enumerate(data.get("tables", [])):
            df = pd.DataFrame(table["table_content"], columns=table["table_columns"])
            df = infer_dtype(df)
            var = f"df_{i}"
            dfs[var] = df
        if not build_schema:
            self._schema_pruning_diagnostics = {}
            return dfs, ""
        schema_text = self._render_schema_text(
            dfs=dfs,
            table_names=data.get("table_names", []),
            query=query,
            contract=contract,
            relation_anchors=relation_anchors,
            cell_anchors=cell_anchors,
        )
        return dfs, schema_text

    @staticmethod
    def _sample_values(series: pd.Series, limit: int = 200) -> set:
        values = set()
        if limit <= 0:
            return values
        if len(series) <= limit:
            sample = series
        else:
            edge = max(1, limit // 4)
            middle_count = max(0, limit - (edge * 2))
            indices = list(range(edge))
            if middle_count:
                middle = np.linspace(edge, len(series) - edge - 1, num=middle_count, dtype=int)
                indices.extend(int(i) for i in middle)
            indices.extend(range(max(edge, len(series) - edge), len(series)))
            sample = series.iloc[sorted(set(indices))[:limit]]
        for value in sample.tolist():
            if is_empty_value(value):
                continue
            text = normalize_join_value(value)
            if text:
                values.add(text)
        return values

    def _build_relation_evidence(
        self,
        dfs: Mapping[str, pd.DataFrame],
        table_names: Sequence[str],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.enable_relation_hints or len(dfs) < 2:
            return "Relation hints disabled.", []
        render_hints = len(dfs) >= self.relation_hints_min_tables

        names = list(dfs.keys())
        display = {
            name: table_names[i] if i < len(table_names) else name
            for i, name in enumerate(names)
        }
        pair_hints: List[Tuple[float, str, str, str, str, Dict[str, Any]]] = []
        best_pair_edge: Dict[Tuple[str, str], Tuple[float, str, Dict[str, Any]]] = {}

        for left_idx, left_name in enumerate(names):
            left_df = dfs[left_name]
            left_cols = [str(c) for c in left_df.columns]
            left_compact = {compact_name(c): c for c in left_cols}
            for right_name in names[left_idx + 1:]:
                right_df = dfs[right_name]
                right_cols = [str(c) for c in right_df.columns]
                candidate_pairs: List[Tuple[str, str, str]] = []

                right_compact = {compact_name(c): c for c in right_cols}
                for compact, left_col in left_compact.items():
                    if compact and compact in right_compact:
                        candidate_pairs.append((left_col, right_compact[compact], "same-column"))

                for left_col in left_cols:
                    left_kind = column_kind(left_col)
                    if not left_kind:
                        continue
                    for right_col in right_cols:
                        right_kind = column_kind(right_col)
                        if left_kind and left_kind == right_kind:
                            candidate_pairs.append((left_col, right_col, f"{left_kind}-key"))

                for left_col in left_cols:
                    left_values = self._sample_values(left_df[left_col])
                    if not left_values:
                        continue
                    left_kind = column_kind(left_col)
                    for right_col in right_cols:
                        right_values = self._sample_values(right_df[right_col])
                        if not right_values:
                            continue
                        overlap = len(left_values & right_values)
                        if not overlap:
                            continue
                        denom = max(1, min(len(left_values), len(right_values)))
                        ratio = overlap / denom
                        right_kind = column_kind(right_col)
                        if overlap >= 2 or ratio >= 0.25 or left_kind or right_kind:
                            candidate_pairs.append((left_col, right_col, "value-overlap"))

                seen_pairs = set()
                for left_col, right_col, reason in candidate_pairs:
                    key = (left_col, right_col)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    left_values = self._sample_values(left_df[left_col])
                    right_values = self._sample_values(right_df[right_col])
                    overlap = len(left_values & right_values)
                    denom = max(1, min(len(left_values), len(right_values)))
                    ratio = overlap / denom
                    name_bonus = 1.0 if compact_name(left_col) == compact_name(right_col) else 0.0
                    kind_bonus = 0.5 if reason.endswith("-key") else 0.0
                    score = overlap * 2.0 + ratio * 3.0 + name_bonus + kind_bonus
                    if score <= 0 and reason != "same-column":
                        continue
                    line = (
                        f"{left_name}.{left_col} <-> {right_name}.{right_col} "
                        f"({reason}, sample_overlap={overlap}, overlap_ratio={ratio:.2f})"
                    )
                    anchor = {
                        "type": "relation",
                        "left_df": left_name,
                        "left_table": display.get(left_name, left_name),
                        "left_col": str(left_col),
                        "right_df": right_name,
                        "right_table": display.get(right_name, right_name),
                        "right_col": str(right_col),
                        "reason": reason,
                        "sample_overlap": int(overlap),
                        "overlap_ratio": float(ratio),
                        "score": float(score),
                    }
                    pair_hints.append((score, left_name, right_name, line, reason, anchor))
                    edge_key = (left_name, right_name)
                    if edge_key not in best_pair_edge or score > best_pair_edge[edge_key][0]:
                        best_pair_edge[edge_key] = (score, line, anchor)

        pair_hints.sort(key=lambda item: item[0], reverse=True)
        anchors = []
        for idx, item in enumerate(pair_hints[:12], start=1):
            anchor = dict(item[5])
            anchor["edge_id"] = f"r{idx}"
            anchors.append(anchor)

        if not render_hints:
            return f"Relation hints skipped for {len(dfs)} tables.", anchors
        if not pair_hints:
            return "No strong relation hints found from column names or sample overlaps.", []

        lines = ["Likely table relations. Prefer these join keys before guessing:"]
        for _, _, _, line, _, _ in pair_hints[:12]:
            lines.append(f"- {line}")

        if len(names) >= 3:
            bridge_lines = []
            for mid in names:
                others = [name for name in names if name != mid]
                for i, left in enumerate(others):
                    for right in others[i + 1:]:
                        edge1 = best_pair_edge.get((left, mid)) or best_pair_edge.get((mid, left))
                        edge2 = best_pair_edge.get((mid, right)) or best_pair_edge.get((right, mid))
                        if edge1 and edge2:
                            score = edge1[0] + edge2[0]
                            bridge_lines.append((score, f"{left} -> {mid} -> {right}"))
            bridge_lines.sort(reverse=True)
            for _, path in bridge_lines[:3]:
                pretty = " -> ".join(display.get(part, part) for part in path.split(" -> "))
                lines.append(f"- possible bridge path: {path} ({pretty})")

        return "\n".join(lines), anchors

    def _build_relation_hints(self, dfs: Mapping[str, pd.DataFrame], table_names: Sequence[str]) -> str:
        relation_text, _ = self._build_relation_evidence(dfs, table_names)
        return relation_text

    def _cell_evidence_row_context(
        self,
        df: pd.DataFrame,
        row_index: Any,
        hit_col: str,
        q_tokens: Sequence[str],
    ) -> Dict[str, str]:
        if not self.cell_evidence_include_rows:
            return {}
        try:
            row = df.loc[row_index]
        except Exception:
            return {}
        if isinstance(row, pd.DataFrame):
            if row.empty:
                return {}
            row = row.iloc[0]

        q_token_set = set(q_tokens)
        ranked_cols = []
        for pos, col in enumerate(df.columns):
            col_s = str(col)
            try:
                value = row[col]
            except Exception:
                continue
            if is_empty_value(value):
                continue

            col_score = 0.0
            compact_col = compact_name(col_s)
            if col_s == hit_col:
                col_score += 10.0
            if q_token_set & set(content_tokens(col_s)):
                col_score += 4.0
            if is_id_like_column(col_s):
                col_score += 3.0
            if any(cue in compact_col for cue in ["name", "title", "detail", "email", "phone"]):
                col_score += 2.0
            ranked_cols.append((col_score, pos, col_s, value))

        if not ranked_cols:
            return {}
        ranked_cols.sort(key=lambda item: (-item[0], item[1]))
        context = {}
        for _, _, col_s, value in ranked_cols[:6]:
            col_text = truncate_text(col_s, 32)
            value_text = truncate_text(value, min(48, self.cell_evidence_max_value_len))
            context[col_text] = value_text
        return context

    def _format_cell_evidence_row_context(
        self,
        df: pd.DataFrame,
        row_index: Any,
        hit_col: str,
        q_tokens: Sequence[str],
    ) -> str:
        context = self._cell_evidence_row_context(df, row_index, hit_col, q_tokens)
        if not context:
            return ""
        return "{" + ", ".join(f"{key}: {value}" for key, value in context.items()) + "}"

    def _build_cell_evidence_anchors(
        self,
        query: str,
        dfs: Mapping[str, pd.DataFrame],
        table_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not self.enable_cell_evidence or self.cell_evidence_top_k <= 0:
            return []

        phrases = extract_quoted_phrases(query)
        q_tokens = content_tokens(query)
        q_token_set = set(q_tokens)
        if not phrases and not q_token_set:
            return []

        display = {
            name: table_names[i] if i < len(table_names) else name
            for i, name in enumerate(dfs.keys())
        }
        candidates: Dict[Tuple[str, str, str], Tuple[float, str, str, str, int, Any, str, List[str], Dict[str, str]]] = {}

        for df_name, df in dfs.items():
            table_display = truncate_text(display.get(df_name, df_name), 60)
            table_token_overlap = q_token_set & set(content_tokens(table_display))
            for col in df.columns:
                col_s = str(col)
                col_token_overlap = sorted(q_token_set & set(content_tokens(col_s)))
                for row_pos, (row_index, value) in enumerate(df[col].items()):
                    if is_empty_value(value):
                        continue
                    value_text = normalize_answer_text(value)
                    if not value_text:
                        continue

                    value_l = value_text.lower()
                    score = 0.0
                    reasons: List[str] = []
                    value_match = False
                    phrase_hit = False

                    for phrase in phrases:
                        phrase_l = phrase.lower()
                        if not phrase_l:
                            continue
                        if value_l == phrase_l:
                            score += 30.0
                            reasons.append(f"quoted_exact:{truncate_text(phrase, 32)}")
                            value_match = True
                            phrase_hit = True
                        elif phrase_l in value_l:
                            score += 22.0
                            reasons.append(f"quoted_contains:{truncate_text(phrase, 32)}")
                            value_match = True
                            phrase_hit = True
                        elif value_l in phrase_l and len(value_l) >= 3:
                            score += 12.0
                            reasons.append(f"cell_in_quote:{truncate_text(value_text, 32)}")
                            value_match = True
                            phrase_hit = True

                    value_token_overlap = sorted(q_token_set & set(split_words(value_text)))
                    if value_token_overlap:
                        score += 4.0 * len(value_token_overlap)
                        reasons.append(f"tokens:{','.join(value_token_overlap[:5])}")
                        value_match = True

                    if not value_match:
                        continue

                    if col_token_overlap:
                        score += 1.25 * len(col_token_overlap)
                        reasons.append(f"column:{','.join(col_token_overlap[:4])}")
                    if table_token_overlap:
                        score += 0.75 * len(table_token_overlap)
                        reasons.append(f"table:{','.join(sorted(table_token_overlap)[:4])}")
                    if is_number_like_value(value_text):
                        numeric_overlap = [token for token in value_token_overlap if token.isdigit()]
                        if numeric_overlap:
                            score += 3.0
                        elif not phrase_hit:
                            score -= 2.0
                    if len(value_text) <= 1 and not phrase_hit:
                        score -= 3.0
                    if score <= 0:
                        continue

                    dedupe_key = (df_name, col_s, normalize_join_value(value_text))
                    candidate = (
                        score,
                        df_name,
                        table_display,
                        col_s,
                        row_pos,
                        row_index,
                        value_text,
                        reasons[:5],
                        self._cell_evidence_row_context(df, row_index, col_s, q_tokens),
                    )
                    if dedupe_key not in candidates or score > candidates[dedupe_key][0]:
                        candidates[dedupe_key] = candidate

        ranked = sorted(candidates.values(), key=lambda item: (-item[0], item[1], item[3], item[4]))
        anchors = []
        for idx, (score, df_name, table_display, col_s, row_pos, row_index, value_text, reasons, row_context) in enumerate(ranked[: self.cell_evidence_top_k], start=1):
            anchors.append({
                "anchor_id": f"c{idx}",
                "type": "cell",
                "df": df_name,
                "table": table_display,
                "column": col_s,
                "row": int(row_pos),
                "row_index": normalize_answer_text(row_index),
                "value": truncate_text(value_text, self.cell_evidence_max_value_len),
                "score": float(score),
                "matched": list(reasons),
                "row_context": row_context,
            })
        return anchors

    def _render_cell_evidence(self, anchors: Sequence[Mapping[str, Any]]) -> str:
        if not self.enable_cell_evidence or self.cell_evidence_top_k <= 0:
            return "Cell evidence disabled."
        if not anchors:
            return "No relevant cell evidence found."

        lines = ["Top cell evidence scanned from full tables. Treat as hints, not final answers:"]
        for anchor in anchors:
            value_render = json.dumps(
                anchor.get("value", ""),
                ensure_ascii=False,
            )
            line = (
                f"- {anchor.get('df')}.{anchor.get('column')} row={anchor.get('row')} value={value_render} "
                f"(table={anchor.get('table')}, score={float(anchor.get('score', 0.0)):.1f}, "
                f"matched={';'.join(str(x) for x in anchor.get('matched', []))})"
            )
            row_context = anchor.get("row_context") or {}
            if row_context:
                rendered_context = "{" + ", ".join(f"{key}: {value}" for key, value in row_context.items()) + "}"
                line += f" row_context={rendered_context}"
            lines.append(line)
        return "\n".join(lines)

    def _build_cell_evidence(
        self,
        query: str,
        dfs: Mapping[str, pd.DataFrame],
        table_names: Sequence[str],
    ) -> str:
        anchors = self._build_cell_evidence_anchors(query, dfs, table_names)
        return self._render_cell_evidence(anchors)

    @staticmethod
    def _is_support_classification_query(query: str) -> bool:
        q = str(query or "").lower()
        return (
            "supports the above statement" in q
            or "output 'supported'" in q
            or 'output "supported"' in q
            or "output supported" in q
        ) and "unsupported" in q

    @staticmethod
    def _is_list_completion_query(query: str) -> bool:
        q = str(query or "").lower()
        return bool(
            re.search(r"\b(list|all|different|distinct)\b", q)
            or re.search(r"\b(what|which|who)\s+are\b", q)
            or re.search(r"\bwhich\s+(?:states|countries|cities|schools|universities|institutions|colleges|players|students|people|customers|companies|departments|names|titles|items|products|makers|manufacturers)\b", q)
            or re.search(r"\bwhat\s+(?:states|countries|cities|schools|universities|institutions|colleges|players|students|people|customers|companies|departments|names|titles|items|products|makers|manufacturers)\b", q)
            or "names of" in q
        )

    def infer_contract(self, query: str, table_count: int) -> AnswerContract:
        q = f" {query.lower()} "
        terms: List[str] = []
        answer_type = "text"
        operation = "lookup"
        longtablebench_rules = bool(getattr(self, "enable_longtablebench_diagnostic_rules", False))
        support_classification = (
            longtablebench_rules
            and self._is_support_classification_query(query)
        )
        list_completion = self._is_list_completion_query(query)

        target_map = [
            ("email", ["email", "e-mail"]),
            ("address", ["address", "street address"]),
            ("phone", ["phone", "cell", "mobile", "telephone"]),
            ("time", ["time", "hour"]),
            ("room", ["room", "classroom"]),
            ("date", ["date", "when", "year"]),
            ("number", ["how many", "number of", " count ", " counts "]),
            ("name", ["name", "who", "person", "player", "student", "coach", "member", "employee"]),
            ("title", ["title", "book", "movie", "film", "song", "album"]),
            ("company", ["company", "manufacturer", "maker"]),
            ("university", ["school", "university", "institution", "college"]),
            ("department", ["department", "dept", "division"]),
            ("city", ["city", "town"]),
            ("country", ["country", "nation"]),
            ("state", ["state", "province"]),
            ("detail", ["detail", "description"]),
            ("enzyme", ["enzyme"]),
        ]
        for term, cues in target_map:
            if any((cue in q if cue.strip() != cue or " " in cue.strip() else _text_contains_cue(q, cue)) for cue in cues):
                terms.append(term)
        phone_number_of = bool(re.search(r"\b(?:phone|telephone|mobile|cell)\s+number\s+of\b", q))
        has_count_cue = (
            "how many" in q
            or ("number of" in q and not phone_number_of)
            or bool(re.search(r"\bcounts?\b", q))
        )
        has_rank_cue = bool(
            re.search(r"\b(highest|lowest|largest|smallest|maximum|minimum|most|greatest|fewest)\b", q)
            or re.search(r"(?<!\bat\s)\bleast\b", q)
            or re.search(r"\b(shortest|longest|max|min)\b", q)
        )
        asks_rank_subject = _rank_subject_question(query)
        asks_numeric_extreme_value = _numeric_extreme_value_question(query)
        has_date_answer_cue = (
            "when" in q
            or "what date" in q
            or "which date" in q
            or "on which date" in q
            or bool(re.search(r"\bwhat year\b|\bwhich year\b", q))
        )
        if any(cue in q for cue in ["email", "e-mail"]):
            answer_type = "email"
        elif has_count_cue and not asks_rank_subject:
            answer_type = "number"
            operation = "count"
        elif asks_numeric_extreme_value:
            answer_type = "number"
            operation = "rank"
        elif any(cue in q for cue in ["average", " avg "]):
            answer_type = "number"
            operation = "avg"
        elif any(cue in q for cue in ["total", "sum "]):
            answer_type = "number"
            operation = "sum"
        elif has_date_answer_cue:
            answer_type = "date"

        if has_rank_cue:
            operation = "rank"
            if asks_rank_subject and answer_type == "number":
                answer_type = "text"
        if any(cue in q for cue in [" both ", " among those ", " and among those ", " also "]):
            operation = "intersection"

        focus_for_field_count = re.sub(r"\b(?:email|e-mail|mail)\s+address(?:es)?\b", "email", _answer_focus_text(query))
        multi_field = bool(
            re.search(r"\b(and what|and how many|along with|together with|as well as|name and)\b", q)
            or len(re.findall(r"\b(?:name|names|email|e-mail|address|addresses|phone|telephone|time|room|date|department|city|cities|country|state|province|title|description|descriptions|detail|details|id|ids)\b", focus_for_field_count)) >= 2
        )
        if multi_field:
            # Keep the generic terms because multi-field projection needs all requested fields.
            for cue in ["headquarters", "sales", "enrollment", "population"]:
                if cue in q and cue not in terms:
                    terms.append(cue)

        requires_single = any(cue in q for cue in [
            "what is the", "who is the", "which is the", "highest", "lowest",
            "largest", "smallest", "maximum", "minimum", "most ", "least ",
        ])
        if list_completion or any(cue in q for cue in ["list", "what are", "which are", "who are", "names of", "all "]):
            requires_single = False

        if not terms:
            terms = ["answer"]
        required_output_slots = infer_required_output_slots(
            query=query,
            answer_type=answer_type,
            operation=operation,
            target_terms=terms,
        )
        if support_classification:
            answer_type = "text"
            operation = "support_classification"
            requires_single = True
            multi_field = False
            terms = ["support_status"]
            required_output_slots = [{
                "name": "support_status",
                "kind": "text",
                "cues": ["support_status", "supported", "unsupported", "entailment", "statement"],
                "required": True,
            }]
        elif list_completion and answer_type == "text" and operation != "rank":
            operation = "list_completion"
            requires_single = False
        return AnswerContract(
            answer_type=answer_type,
            target_terms=dedupe_keep_order(terms),
            required_output_slots=required_output_slots,
            operation=operation,
            requires_single=requires_single,
            multi_field=multi_field or sum(1 for slot in required_output_slots if slot.get("required", True)) > 1,
            max_steps=self.max_steps_three if table_count >= 3 else self.max_steps_two,
        )

    def _schema_memory_summary(self, observations: Sequence[StepObservation]) -> str:
        if not observations:
            return "No previous steps."
        lines = []
        for obs in observations:
            struct = obs.structured_result or {}
            rows = struct.get("rows") or []
            lines.append(
                f"Step {obs.step_id}: status={obs.status}, contract={obs.contract_status}, "
                f"columns={struct.get('columns', [])}, rows_preview={rows[:5]}, error={obs.error or obs.contract_error}"
            )
        return "\n".join(lines[-4:])

    def _diagnostic_contract_rules(self, contract: AnswerContract) -> str:
        if not self.enable_longtablebench_diagnostic_rules:
            return ""
        rules = []
        if contract.operation == "support_classification":
            rules.append(
                "11. For support/unsupported questions, compute the statement truth value and make "
                "step_result a one-row table with column support_status containing exactly "
                "'Supported' or 'Unsupported'. Do not finish with a raw row, dict, entity, or timestamp."
            )
            rules.append(
                "12. When a date and a time are split across columns, compare the date column to the "
                "requested date and the time column to the requested HH:MM:SS string separately."
            )
        elif contract.operation == "list_completion":
            rules.append(
                "11. For list/different/distinct/all questions, keep every matching row and return all "
                "unique requested answer entities. Do not use .iloc[0], head(1), or a single exemplar "
                "unless the question explicitly asks for one row."
            )
            rules.append(
                "12. For maker/manufacturer/company lists, project the canonical maker/manufacturer/company "
                "name column after filtering the cars/products, then drop duplicates."
            )
        return "\n" + "\n".join(rules) if rules else ""

    def _build_prompt(
        self,
        query: str,
        schema_text: str,
        relation_text: str,
        cell_evidence_text: str,
        contract: AnswerContract,
        observations: Sequence[StepObservation],
        df_names: Optional[Sequence[str]] = None,
    ) -> str:
        allowed_dfs = ", ".join(df_names or ["df_0"])
        if self.enable_safe_merge:
            merge_rule = (
                f"1. Use only {allowed_dfs}, previous variables, pandas as pd, numpy as np, and safe_merge.\n"
                "2. Store the step output in a variable named step_result.\n"
                "3. Prefer safe_merge(left, right, left_on=..., right_on=..., how='inner') for joins, using the relation hints when applicable.\n"
                "4. Do not use pd.merge(...) or DataFrame.merge(...)."
            )
        else:
            merge_rule = (
                f"1. Use only {allowed_dfs}, previous variables, pandas as pd, and numpy as np.\n"
                "2. Store the step output in a variable named step_result.\n"
                "3. Use ordinary pandas joins or merges when needed.\n"
                "4. Join keys must still be derived from visible columns, not guessed."
            )
        contract_rule = (
            "9. If the previous result does not satisfy the answer contract, repair it with another pandas_code action."
            if self.enable_answer_contracts
            else "9. The answer-contract gate is disabled for this ablation; still return the requested final answer."
        )
        return f"""You are a minimal ReAct pandas agent for multi-table QA.
You must choose exactly one next local action. Do not make a global plan.

Question:
{query}

Answer contract inferred by the executor:
{json.dumps(contract.to_prompt_dict(), ensure_ascii=False)}

Available DataFrames and samples:
{schema_text}

Join and relation hints:
{relation_text}

Relevant cell evidence from full tables:
{cell_evidence_text}

Previous observations:
{self._schema_memory_summary(observations)}

Return valid JSON only. No markdown outside JSON.
JSON schema:
{{
  "thought": "one short reason for the next local operation",
  "action": "pandas_code" | "finish",
  "code": "python code when action is pandas_code",
  "answer_column": "optional final answer column when action is finish"
}}

Rules for pandas_code:
{merge_rule}
5. Keep only columns needed for the next step or final answer; do not return whole unrelated rows.
6. For ranking questions, sort and keep the top/bottom candidate rows before finishing.
7. For count/sum/avg questions, make step_result a scalar or a one-row table with the numeric answer.
8. Use cell evidence only as hints for locating rows/values; verify the answer by pandas operations over the full DataFrames.
{contract_rule}
10. When an entity/value filter returns multiple matching ids, keep all candidates and use .isin(...); do not use .iloc[0] unless the question explicitly asks for one specific row.
{self._diagnostic_contract_rules(contract)}

Rules for finish:
1. Use finish only when the latest structured result already contains the factual final answer.
2. Do not invent an answer in finish; final formatting is handled by the executor.
"""

    @staticmethod
    def _extract_json_action(raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
        if fence:
            text = fence.group(1).strip()
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(text[start:end + 1])
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        loose = MinimalReActAgent._extract_loose_json_action(text)
        if loose:
            return loose
        code = MinimalReActAgent._extract_code(text)
        if code:
            return {"thought": "Recovered python code from non-JSON response.", "action": "pandas_code", "code": code}
        return {"thought": "Could not parse model response.", "action": "finish", "answer": ""}

    @staticmethod
    def _extract_loose_json_action(text: str) -> Dict[str, Any]:
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text, flags=re.I)
        if not action_match:
            return {}
        action = action_match.group(1).strip()

        def loose_string(key: str) -> str:
            start_match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text, flags=re.I)
            if not start_match:
                return ""
            start = start_match.end()
            end_match = re.search(r'"\s*,\s*"[a-zA-Z_][a-zA-Z0-9_]*"\s*:', text[start:], flags=re.S)
            if end_match:
                return text[start:start + end_match.start()]
            end_match = re.search(r'"\s*}\s*$', text[start:], flags=re.S)
            if end_match:
                return text[start:start + end_match.start()]
            return text[start:].strip().strip('"')

        return {
            "thought": loose_string("thought"),
            "action": action,
            "code": loose_string("code"),
            "answer_column": loose_string("answer_column"),
        }

    @staticmethod
    def _extract_code(text: str) -> str:
        text = str(text or "")
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
        if match:
            return match.group(1).strip()
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("\n", 1)
            text = parts[1] if len(parts) == 2 else ""
            text = text.strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        if text.lower().startswith("python\n"):
            return text.split("\n", 1)[1].strip()
        return text

    def _canonicalize_model_code(self, code: str) -> str:
        code = self._extract_code(code)
        if self.enable_safe_merge:
            code = code.replace("pd.safe_merge(", "safe_merge(")
        if "\\n" in code and "\n" not in code:
            code = code.replace("\\n", "\n")
        if self.enable_safe_merge:
            code = re.sub(r"\bpd\.merge\s*\(", "safe_merge(", code)
        kept = []
        skip_safe_merge_def = False
        for line in code.splitlines():
            stripped = line.strip()
            if (
                re.fullmatch(r"import\s+(pandas|numpy)\s+as\s+(pd|np)", stripped)
                or re.fullmatch(r"from\s+safe_merge\s+import\s+safe_merge", stripped)
            ):
                continue
            if re.match(r"def\s+safe_merge\s*\(", stripped):
                skip_safe_merge_def = True
                continue
            if skip_safe_merge_def:
                if line.startswith((" ", "\t")) or not stripped:
                    continue
                skip_safe_merge_def = False
            kept.append(line)
        return "\n".join(kept).strip()

    @staticmethod
    def _static_code_check(
        code: str,
        dfs: Mapping[str, pd.DataFrame],
        memory: Mapping[str, Any],
        allow_raw_merge: bool = False,
        enable_column_checks: bool = True,
    ) -> Tuple[bool, str]:
        if not code.strip():
            return False, "empty code"
        normalized = code.replace("pd.safe_merge(", "safe_merge(")
        if not allow_raw_merge and (re.search(r"\bpd\.merge\s*\(", normalized) or re.search(r"\.merge\s*\(", normalized)):
            return False, "raw pandas merge is not allowed; use safe_merge"
        try:
            tree = ast.parse(normalized)
        except SyntaxError as exc:
            return False, f"syntax error: {exc}"

        known: Dict[str, set] = {}
        for name, value in {**dfs, **memory}.items():
            if isinstance(value, pd.DataFrame):
                known[name] = {str(c) for c in value.columns}

        errors = []

        def closest(col: str, columns: Iterable[str]) -> str:
            all_columns = [str(c) for c in columns]
            matches = difflib.get_close_matches(str(col), all_columns, n=3, cutoff=0.55)
            compact_col = compact_name(col)
            for candidate in all_columns:
                compact_candidate = compact_name(candidate)
                if (
                    compact_col
                    and candidate not in matches
                    and (compact_candidate.endswith(compact_col) or compact_col in compact_candidate)
                ):
                    matches.append(candidate)
                if len(matches) >= 3:
                    break
            return f"; closest={matches}" if matches else ""

        def missing_msg(var: str, col: str) -> str:
            return f"{var}[{col!r}] missing from columns{closest(col, known.get(var, []))}"

        def literal_strings(node: ast.AST) -> List[str]:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return [node.value]
            if isinstance(node, (ast.List, ast.Tuple)):
                return [
                    elt.value for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
            return []

        def dict_string_keys(node: ast.AST) -> List[str]:
            if not isinstance(node, ast.Dict):
                return []
            out = []
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    out.append(key.value)
            return out

        def keyword_value(node: ast.Call, name: str) -> Optional[ast.AST]:
            for kw in node.keywords:
                if kw.arg == name:
                    return kw.value
            return None

        def literal_pair(node: Optional[ast.AST]) -> Optional[Tuple[str, str]]:
            if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 2:
                values = literal_strings(node)
                if len(values) == 2:
                    return values[0], values[1]
            return None

        def expr_label(node: Optional[ast.AST]) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                return node.value.id
            return "input"

        def infer_safe_merge_columns(node: ast.Call) -> set:
            left_node = node.args[0] if len(node.args) >= 1 else keyword_value(node, "left_df")
            right_node = node.args[1] if len(node.args) >= 2 else keyword_value(node, "right_df")
            left_cols = infer_columns(left_node) or set()
            right_cols = infer_columns(right_node) or set()
            on_keys = literal_strings(keyword_value(node, "on"))
            left_keys = literal_strings(keyword_value(node, "left_on"))
            right_keys = literal_strings(keyword_value(node, "right_on"))
            suffixes = literal_pair(keyword_value(node, "suffixes")) or ("_x", "_y")

            if on_keys:
                left_keys = right_keys = on_keys
            right_drop_keys = {
                right_key for left_key, right_key in zip(left_keys, right_keys)
                if left_key == right_key
            }
            overlap = (set(left_cols) & set(right_cols)) - right_drop_keys
            out = set()
            for col in left_cols:
                out.add(f"{col}{suffixes[0]}" if col in overlap else col)
            for col in right_cols:
                if col in right_drop_keys:
                    continue
                out.add(f"{col}{suffixes[1]}" if col in overlap else col)
            return out

        def infer_columns(node: Optional[ast.AST]) -> Optional[set]:
            if node is None:
                return None
            if isinstance(node, ast.Name) and node.id in known:
                return set(known[node.id])
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in known:
                    cols = literal_strings(node.slice)
                    return set(cols) if cols else set(known[node.value.id])
                if (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr in {"loc", "at"}
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id in known
                    and isinstance(node.slice, ast.Tuple)
                    and len(node.slice.elts) >= 2
                ):
                    cols = literal_strings(node.slice.elts[1])
                    return set(cols) if cols else set(known[node.value.value.id])
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "safe_merge":
                    return infer_safe_merge_columns(node)
                if isinstance(func, ast.Attribute):
                    base_cols = infer_columns(func.value)
                    if base_cols is None:
                        return None
                    if func.attr in {"copy", "head", "tail", "drop_duplicates", "dropna", "fillna", "sort_values", "reset_index"}:
                        return set(base_cols)
                    if func.attr == "rename":
                        renamed = set(base_cols)
                        columns_arg = keyword_value(node, "columns")
                        if isinstance(columns_arg, ast.Dict):
                            for old, new in zip(columns_arg.keys, columns_arg.values):
                                if isinstance(old, ast.Constant) and isinstance(old.value, str):
                                    renamed.discard(old.value)
                                    if isinstance(new, ast.Constant) and isinstance(new.value, str):
                                        renamed.add(new.value)
                        return renamed
                    if func.attr == "assign":
                        assigned = set(base_cols)
                        for kw in node.keywords:
                            if kw.arg:
                                assigned.add(str(kw.arg))
                        return assigned
                    if func.attr == "groupby":
                        group_cols = []
                        if node.args:
                            group_cols.extend(literal_strings(node.args[0]))
                        by_kw = keyword_value(node, "by")
                        if by_kw is not None:
                            group_cols.extend(literal_strings(by_kw))
                        return set(group_cols) or set(base_cols)
                    if func.attr in {"agg", "aggregate"}:
                        agg_cols = set(base_cols)
                        if node.args:
                            agg_cols.update(dict_string_keys(node.args[0]))
                        return agg_cols
            return None

        class SecurityChecker(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                errors.append("imports are not allowed in generated pandas code")

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                errors.append("imports are not allowed in generated pandas code")

            def visit_Name(self, node: ast.Name) -> None:
                if node.id in DANGEROUS_NAMES or node.id in DANGEROUS_MODULE_NAMES:
                    errors.append(f"unsafe name {node.id!r} is not allowed")

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if node.attr.startswith("__") or node.attr in DANGEROUS_ATTRS:
                    errors.append(f"unsafe attribute {node.attr!r} is not allowed")
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if isinstance(func, ast.Name) and func.id in DANGEROUS_NAMES:
                    errors.append(f"unsafe call {func.id!r} is not allowed")
                if isinstance(func, ast.Attribute) and (func.attr.startswith("__") or func.attr in DANGEROUS_ATTRS):
                    errors.append(f"unsafe call to attribute {func.attr!r} is not allowed")
                self.generic_visit(node)

        class Checker(ast.NodeVisitor):
            def _record_assigned_columns(self, target: ast.AST) -> None:
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name) and target.value.id in known:
                        known[target.value.id].update(str(col) for col in literal_strings(target.slice))
                    if (
                        isinstance(target.value, ast.Attribute)
                        and target.value.attr in {"loc", "at"}
                        and isinstance(target.value.value, ast.Name)
                        and target.value.value.id in known
                        and isinstance(target.slice, ast.Tuple)
                        and len(target.slice.elts) >= 2
                    ):
                        known[target.value.value.id].update(str(col) for col in literal_strings(target.slice.elts[1]))

            def visit_Assign(self, node: ast.Assign) -> None:
                self.visit(node.value)
                cols = infer_columns(node.value)
                if cols:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            known[target.id] = {str(c) for c in cols}
                for target in node.targets:
                    self._record_assigned_columns(target)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if node.value is not None:
                    self.visit(node.value)
                    cols = infer_columns(node.value)
                    if cols and isinstance(node.target, ast.Name):
                        known[node.target.id] = {str(c) for c in cols}
                self._record_assigned_columns(node.target)

            def visit_Subscript(self, node: ast.Subscript) -> None:
                if isinstance(node.value, ast.Name) and node.value.id in known:
                    for col in literal_strings(node.slice):
                        if col not in known[node.value.id]:
                            errors.append(missing_msg(node.value.id, col))
                if (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr in {"loc", "at"}
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id in known
                    and isinstance(node.slice, ast.Tuple)
                    and len(node.slice.elts) >= 2
                ):
                    var = node.value.value.id
                    for col in literal_strings(node.slice.elts[1]):
                        if col not in known[var]:
                            errors.append(missing_msg(var, col))
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if isinstance(func, ast.Name) and func.id == "safe_merge":
                    left_node = node.args[0] if len(node.args) >= 1 else keyword_value(node, "left_df")
                    right_node = node.args[1] if len(node.args) >= 2 else keyword_value(node, "right_df")
                    left_cols = infer_columns(left_node)
                    right_cols = infer_columns(right_node)
                    left_label = expr_label(left_node)
                    right_label = expr_label(right_node)
                    on_value = keyword_value(node, "on")
                    left_on_value = keyword_value(node, "left_on")
                    right_on_value = keyword_value(node, "right_on")
                    if on_value is not None:
                        for col in literal_strings(on_value):
                            if left_cols is not None and col not in left_cols:
                                errors.append(f"safe_merge on={col!r} missing from {left_label}{closest(col, left_cols)}")
                            if right_cols is not None and col not in right_cols:
                                errors.append(f"safe_merge on={col!r} missing from {right_label}{closest(col, right_cols)}")
                    if left_on_value is not None or right_on_value is not None:
                        left_keys = literal_strings(left_on_value) if left_on_value is not None else []
                        right_keys = literal_strings(right_on_value) if right_on_value is not None else []
                        if not left_keys or not right_keys:
                            errors.append("safe_merge requires literal left_on and right_on keys for static checking")
                        for col in left_keys:
                            if left_cols is not None and col not in left_cols:
                                errors.append(f"safe_merge left_on={col!r} missing from {left_label}{closest(col, left_cols)}")
                        for col in right_keys:
                            if right_cols is not None and col not in right_cols:
                                errors.append(f"safe_merge right_on={col!r} missing from {right_label}{closest(col, right_cols)}")
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in known:
                    if func.attr in {"sort_values", "drop_duplicates", "groupby", "set_index"}:
                        for arg in node.args[:1]:
                            for col in literal_strings(arg):
                                if col not in known[func.value.id]:
                                    errors.append(f"{func.value.id}.{func.attr}({col!r}) references missing column{closest(col, known[func.value.id])}")
                        for kw in node.keywords:
                            if kw.arg in {"by", "subset"}:
                                for col in literal_strings(kw.value):
                                    if col not in known[func.value.id]:
                                        errors.append(f"{func.value.id}.{func.attr}({col!r}) references missing column{closest(col, known[func.value.id])}")
                    if func.attr == "rename":
                        columns_arg = keyword_value(node, "columns")
                        for col in dict_string_keys(columns_arg) if columns_arg is not None else []:
                            if col not in known[func.value.id]:
                                errors.append(f"{func.value.id}.rename(columns={{ {col!r}: ... }}) references missing column{closest(col, known[func.value.id])}")
                    if func.attr == "query" and node.args:
                        for expr in literal_strings(node.args[0]):
                            for col in re.findall(r"`([^`]+)`", expr):
                                if col not in known[func.value.id]:
                                    errors.append(f"{func.value.id}.query references missing column {col!r}{closest(col, known[func.value.id])}")
                self.generic_visit(node)

        SecurityChecker().visit(tree)
        if enable_column_checks:
            Checker().visit(tree)
        if errors:
            return False, "; ".join(errors[:3])
        return True, ""

    @staticmethod
    def _preview_rows(rows: Sequence[Sequence[Any]], limit: int = 20) -> List[List[Any]]:
        return [list(row) for row in list(rows)[:limit]]

    @staticmethod
    def _projection_rows(structured: Mapping[str, Any]) -> List[Any]:
        rows = structured.get("projection_rows")
        if isinstance(rows, list):
            return rows
        return list(structured.get("rows") or [])

    @staticmethod
    def _projection_structured(structured: Mapping[str, Any]) -> Dict[str, Any]:
        projected = dict(structured)
        projection_rows = MinimalReActAgent._projection_rows(structured)
        projected["rows"] = projection_rows
        if projection_rows and not structured.get("projection_truncated"):
            projected["truncated"] = False
        return projected

    def _object_to_structured(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, pd.DataFrame):
            safe = obj.copy()
            safe = safe.replace({np.nan: None})
            projection_frame = safe.head(PROJECTION_ROW_LIMIT)
            projection_rows = projection_frame.astype(object).where(pd.notnull(projection_frame), None).values.tolist()
            preview_rows = self._preview_rows(projection_rows, 20)
            return {
                "kind": "dataframe",
                "columns": [str(c) for c in safe.columns],
                "rows": preview_rows,
                "projection_rows": projection_rows,
                "shape": [int(safe.shape[0]), int(safe.shape[1])],
                "truncated": int(safe.shape[0]) > 20,
                "projection_truncated": int(safe.shape[0]) > len(projection_rows),
            }
        if isinstance(obj, pd.Series):
            name = str(obj.name or "value")
            projection_series = obj.head(PROJECTION_ROW_LIMIT)
            values = projection_series.astype(object).where(pd.notnull(projection_series), None).tolist()
            projection_rows = [[v] for v in values[:PROJECTION_ROW_LIMIT]]
            return {
                "kind": "series",
                "columns": [name],
                "rows": projection_rows[:20],
                "projection_rows": projection_rows,
                "shape": [int(obj.shape[0]), 1],
                "truncated": int(obj.shape[0]) > 20,
                "projection_truncated": int(obj.shape[0]) > len(projection_rows),
            }
        if isinstance(obj, (list, tuple, set)):
            values = list(obj)
            projection_rows = [[v] for v in values[:PROJECTION_ROW_LIMIT]]
            return {
                "kind": "list",
                "columns": ["value"],
                "rows": projection_rows[:20],
                "projection_rows": projection_rows,
                "shape": [len(values), 1],
                "truncated": len(values) > 20,
                "projection_truncated": len(values) > len(projection_rows),
            }
        return {
            "kind": "scalar",
            "columns": ["value"],
            "rows": [[obj]],
            "projection_rows": [[obj]],
            "shape": [1, 1],
            "truncated": False,
            "projection_truncated": False,
        }

    def _execute_code(
        self,
        code: str,
        dfs: Mapping[str, pd.DataFrame],
        memory: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        code = self._canonicalize_model_code(code)
        ok, static_error = self._static_code_check(
            code,
            dfs,
            memory,
            allow_raw_merge=not self.enable_safe_merge,
            enable_column_checks=self.enable_static_checks,
        )
        if not ok:
            return "ERROR", static_error, {}, memory
        env: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "re": re,
            "math": math,
            "safe_merge": safe_merge,
            **dfs,
            **memory,
        }
        stdout_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                with execution_time_limit(getattr(self, "execution_timeout_seconds", 30)):
                    exec(compile(ast.parse(code), "<react_step>", "exec"), env, env)
            obj = env.get("step_result", env.get("ans"))
            if obj is None:
                return "ERROR", "code executed but did not assign step_result", {}, memory
            reserved = {"pd", "np", "re", "math", "safe_merge", "__builtins__", *dfs.keys()}
            memory.update({k: v for k, v in env.items() if k not in reserved})
            prior_step_count = sum(1 for k in memory if re.fullmatch(r"step_\d+", str(k)))
            memory[f"step_{prior_step_count + 1}"] = obj
            memory["step_result"] = obj
            return "PASS", stdout_buffer.getvalue(), self._object_to_structured(obj), memory
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                err += "\n" + traceback.format_exc()
            return "ERROR", err, {}, memory

    @staticmethod
    def _required_slots(contract: AnswerContract) -> List[Dict[str, Any]]:
        return [dict(slot) for slot in (contract.required_output_slots or []) if slot.get("required", True)]

    @staticmethod
    def _slot_display_name(slot: Mapping[str, Any]) -> str:
        return str(slot.get("name") or slot.get("kind") or "answer")

    @staticmethod
    def _looks_phone_like(values: Sequence[Any]) -> bool:
        samples = [str(v) for v in values if not is_empty_value(v)][:10]
        return any(re.search(r"(?:\+?\d[\d(). -]{6,}\d)", sample) for sample in samples)

    @staticmethod
    def _looks_time_like(values: Sequence[Any]) -> bool:
        samples = [str(v).strip().lower() for v in values if not is_empty_value(v)][:10]
        return any(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b", sample) for sample in samples)

    @staticmethod
    def _looks_date_like(values: Sequence[Any]) -> bool:
        samples = [str(v).strip().lower() for v in values if not is_empty_value(v)][:10]
        return any(
            re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", sample)
            or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", sample)
            for sample in samples
        )

    @staticmethod
    def _non_empty_values_for_column(columns: Sequence[str], rows: Sequence[Sequence[Any]], col: str) -> List[Any]:
        if col not in columns:
            return []
        idx = list(columns).index(col)
        return [row[idx] for row in rows if idx < len(row) and not is_empty_value(row[idx])]

    def _slot_column_score(
        self,
        query: str,
        contract: AnswerContract,
        slot: Mapping[str, Any],
        col: str,
        values: Sequence[Any],
    ) -> float:
        del contract
        col_l = str(col).lower()
        compact = compact_name(col_l)
        tokens = set(split_words(col_l))
        kind = str(slot.get("kind") or "text").lower()
        name = str(slot.get("name") or "").lower()
        cues = [str(cue).lower() for cue in (slot.get("cues") or []) if str(cue).strip()]
        score = 0.0

        if kind == "id":
            if is_id_like_column(col) and identifier_column_matches_query(query, col):
                return 20.0
            return -12.0 if is_id_like_column(col) else -4.0

        if is_id_like_column(col):
            score -= 8.0

        for cue in cues + [name]:
            cue_compact = compact_name(cue)
            if cue and (_text_contains_cue(col_l, cue) or cue_compact and cue_compact in compact):
                score += 5.0
            elif cue in tokens:
                score += 3.0

        if kind == "email":
            if "email" in compact or "mail" in compact:
                score += 8.0
            if any("@" in str(v) for v in values[:20]):
                score += 8.0
        elif kind == "address":
            if any(cue in compact for cue in ["address", "street", "location"]):
                score += 8.0
            if any(cue in compact for cue in ["line", "postal", "zipcode", "zip"]):
                score += 2.0
        elif kind == "phone":
            if any(cue in compact for cue in ["phone", "telephone", "mobile", "cell"]):
                score += 8.0
            if "number" in compact and any(cue in f" {query.lower()} " for cue in ["phone", "telephone", "mobile", "cell"]):
                score += 3.0
            if self._looks_phone_like(values):
                score += 4.0
        elif kind == "date":
            if any(cue in compact for cue in ["date", "year", "month", "day"]):
                score += 7.0
            if self._looks_date_like(values):
                score += 4.0
        elif kind == "time":
            if any(cue in compact for cue in ["time", "hour", "starttime", "endtime"]):
                score += 7.0
            if self._looks_time_like(values):
                score += 4.0
        elif kind == "room":
            if any(cue in compact for cue in ["room", "classroom"]):
                score += 8.0
        elif kind == "institution":
            if any(cue in compact for cue in ["institution", "university", "college", "school", "academy"]):
                score += 8.0
            if any(cue in compact for cue in ["name", "title", "detail"]):
                score += 2.0
        elif kind == "department":
            if any(cue in compact for cue in ["department", "dept", "division"]):
                score += 8.0
            if compact in {"name", "title"} or compact.endswith("name"):
                score += 1.0
        elif kind == "city":
            if "city" in compact or "town" in compact:
                score += 8.0
        elif kind == "country":
            if "country" in compact or "nation" in compact:
                score += 8.0
        elif kind == "state":
            if "state" in compact or "province" in compact:
                score += 8.0
        elif kind == "title":
            if "title" in compact:
                score += 8.0
            if any(cue in compact for cue in ["name", "detail"]):
                score += 2.0
        elif kind == "number":
            if any(cue in compact for cue in ["count", "number", "total", "sum", "avg", "average", "quantity", "amount", "population", "sales", "enrollment", "value"]):
                score += 6.0
            non_empty = [v for v in values if not is_empty_value(v)]
            if non_empty and all(is_number_like_value(v) for v in non_empty):
                score += 5.0
            if is_id_like_column(col):
                score -= 4.0
        else:
            generic_cues = ["detail", "description", "value"] if name == "detail" else ["name", "title", "detail", "description", "value"]
            if any(cue in compact for cue in generic_cues):
                score += 4.0

        non_empty = [v for v in values if not is_empty_value(v)]
        if non_empty:
            numeric_ratio = sum(is_number_like_value(v) for v in non_empty) / len(non_empty)
            if kind in {"text", "email", "address", "phone", "date", "time", "room", "institution", "department", "city", "country", "state", "title"}:
                score += 1.0 - numeric_ratio
                if numeric_ratio == 1.0 and kind not in {"phone", "date", "time"}:
                    score -= 4.0
            elif kind == "number":
                score += numeric_ratio
        return score

    def _select_columns_for_slots(
        self,
        query: str,
        contract: AnswerContract,
        structured: Mapping[str, Any],
    ) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
        columns = [str(c) for c in structured.get("columns", [])]
        rows = self._projection_rows(structured)
        selected: List[str] = []
        selected_by_slot: Dict[str, List[str]] = {}
        missing_slots: List[str] = []
        if not columns:
            return selected, selected_by_slot, [self._slot_display_name(slot) for slot in self._required_slots(contract)]

        for slot in self._required_slots(contract):
            slot_name = self._slot_display_name(slot)
            scored = []
            for idx, col in enumerate(columns):
                vals = [row[idx] for row in rows if idx < len(row)]
                score = self._slot_column_score(query, contract, slot, col, vals)
                if vals and all(is_empty_value(v) for v in vals):
                    score -= 10.0
                scored.append((score, -idx, col))
            scored.sort(reverse=True)
            threshold = 4.0 if str(slot.get("kind") or "") != "text" else 2.0
            matches = [col for score, _, col in scored if score >= threshold]
            if matches:
                best = matches[0]
                selected_by_slot[slot_name] = [best]
                if best not in selected:
                    selected.append(best)
            else:
                selected_by_slot[slot_name] = []
                missing_slots.append(slot_name)
        return selected, selected_by_slot, missing_slots

    @staticmethod
    def _is_helper_or_metric_column(col: str, values: Sequence[Any]) -> bool:
        col_l = str(col).lower()
        compact = compact_name(col_l)
        if is_id_like_column(col):
            return True
        if any(cue in compact for cue in ["count", "number", "total", "sum", "avg", "average", "quantity", "amount", "rank"]):
            return True
        non_empty = [v for v in values if not is_empty_value(v)]
        return bool(non_empty and all(is_number_like_value(v) for v in non_empty))

    def verify_projection(
        self,
        query: str,
        contract: AnswerContract,
        structured_result: Mapping[str, Any],
        selected_columns: Optional[Sequence[str]] = None,
        final_answer: str = "",
    ) -> Dict[str, Any]:
        columns = [str(c) for c in structured_result.get("columns", [])]
        rows = self._projection_rows(structured_result)
        selected_from_slots, selected_slot_columns, missing_slots = self._select_columns_for_slots(
            query, contract, structured_result
        )
        selected = [str(col) for col in (selected_columns or selected_from_slots)]

        required_slots = self._required_slots(contract)
        incompatible_columns: List[str] = []
        matched_columns = {col for cols in selected_slot_columns.values() for col in cols}
        for col in selected:
            vals = self._non_empty_values_for_column(columns, rows, col)
            if col not in matched_columns and self._is_helper_or_metric_column(col, vals):
                incompatible_columns.append(col)

        answer_shape_error = ""
        if len(required_slots) > 1:
            matched_required_count = sum(1 for slot in required_slots if selected_slot_columns.get(self._slot_display_name(slot)))
            if matched_required_count < len(required_slots):
                answer_shape_error = "multi_field_missing_required_columns"
            elif selected and len(set(selected)) < len(required_slots):
                answer_shape_error = "multi_field_projection_collapsed"
            elif final_answer:
                separators = sum(sep in final_answer for sep in [",", ";", "\t", "\n"])
                if len(required_slots) > 1 and separators == 0:
                    answer_shape_error = "multi_field_answer_rendered_as_single_value"

        if contract.answer_type in {"text", "email", "date"} and selected:
            selected_values = [
                value
                for col in selected
                for value in self._non_empty_values_for_column(columns, rows, col)
            ]
            if selected_values and all(is_number_like_value(value) for value in selected_values) and not query_asks_identifier(query):
                incompatible_columns.extend(col for col in selected if col not in incompatible_columns)
            if selected and all(is_id_like_column(col) for col in selected) and not query_asks_identifier(query):
                incompatible_columns.extend(col for col in selected if col not in incompatible_columns)

        status = "FAIL" if missing_slots or incompatible_columns or answer_shape_error else "PASS"
        return {
            "status": status,
            "missing_slots": dedupe_keep_order(missing_slots),
            "incompatible_columns": dedupe_keep_order(incompatible_columns),
            "selected_slot_columns": selected_slot_columns,
            "selected_columns": dedupe_keep_order(selected),
            "answer_shape_error": answer_shape_error,
        }

    def _column_score(self, query: str, contract: AnswerContract, col: str, values: Sequence[Any]) -> float:
        col_l = str(col).lower()
        tokens = set(split_words(col_l))
        score = 0.0
        for term in contract.target_terms:
            t = str(term).lower()
            if t in col_l or t in tokens:
                score += 4.0
            if t == "email" and "mail" in col_l:
                score += 5.0
            if t == "phone" and any(x in col_l for x in ["phone", "mobile", "cell"]):
                score += 5.0
            if t == "name" and any(x in col_l for x in ["name", "detail", "title"]):
                score += 3.0
            if t == "university" and any(x in col_l for x in ["school", "university", "institution", "college", "name"]):
                score += 4.0
            if t == "date" and ("date" in col_l or "year" in col_l):
                score += 4.0
        if contract.answer_type == "number":
            if any(x in col_l for x in ["count", "number", "total", "sum", "avg", "average"]):
                score += 4.0
        if contract.answer_type in {"text", "email", "date"} and is_id_like_column(col):
            score -= 5.0
        non_empty = [v for v in values if not is_empty_value(v)]
        if non_empty:
            text_ratio = sum(not isinstance(v, (int, float, np.integer, np.floating)) for v in non_empty) / len(non_empty)
            if contract.answer_type in {"text", "email", "date"}:
                score += text_ratio
            if contract.answer_type == "email" and any("@" in str(v) for v in non_empty[:10]):
                score += 6.0
        if str(col).lower() in query.lower():
            score += 2.0
        if is_id_like_column(col) and query_asks_identifier(query):
            if identifier_column_matches_query(query, col):
                score += 8.0
            else:
                score -= 8.0
        if contract.operation == "rank" and contract.answer_type in {"text", "email", "date"}:
            non_empty = [v for v in values if not is_empty_value(v)]
            if non_empty and all(isinstance(v, (int, float, np.integer, np.floating)) for v in non_empty):
                score -= 4.0
        return score

    def _select_answer_columns(self, query: str, contract: AnswerContract, structured: Mapping[str, Any]) -> List[str]:
        columns = [str(c) for c in structured.get("columns", [])]
        rows = self._projection_rows(structured)
        if not columns:
            return []
        if self._projection_feedback_enabled() and contract.required_output_slots:
            selected, _, _ = self._select_columns_for_slots(query, contract, structured)
            if selected:
                return selected
        if contract.multi_field:
            selected = []
            id_answer = query_asks_identifier(query)
            for col_idx, col in enumerate(columns):
                score = self._column_score(query, contract, col, [row[col_idx] for row in rows if col_idx < len(row)])
                if score <= 0:
                    continue
                if is_id_like_column(col) and contract.answer_type != "number":
                    if not (id_answer and identifier_column_matches_query(query, col)):
                        continue
                selected.append(col)
            return selected or columns[: min(3, len(columns))]
        if len(columns) == 1:
            if (
                contract.answer_type in {"text", "email", "date"}
                and is_id_like_column(columns[0])
                and query_asks_identifier(query)
                and not identifier_column_matches_query(query, columns[0])
            ):
                return []
            return columns
        scored = []
        for idx, col in enumerate(columns):
            vals = [row[idx] for row in rows if idx < len(row)]
            score = self._column_score(query, contract, col, vals)
            if vals and all(is_empty_value(v) for v in vals):
                score -= 10.0
            scored.append((score, idx, col))
        scored.sort(reverse=True)
        best = scored[0][2]
        if contract.answer_type in {"text", "email", "date"}:
            for score, _, col in scored:
                if score >= scored[0][0] - 1 and not is_id_like_column(col):
                    best = col
                    break
        return [best]

    def check_contract(self, query: str, contract: AnswerContract, structured: Mapping[str, Any]) -> Tuple[str, str]:
        rows = self._projection_rows(structured)
        columns = [str(c) for c in structured.get("columns", [])]
        shape = structured.get("shape") or [0, 0]
        if not self.enable_answer_contracts:
            if not rows or shape[0] == 0:
                return "FAIL", "empty structured result"
            if all(all(is_empty_value(v) for v in row) for row in rows):
                return "FAIL", "all result cells are empty"
            return "PASS", ""

        def projection_error() -> str:
            if not self._projection_feedback_enabled():
                return ""
            verdict = self.verify_projection(query, contract, structured)
            if verdict.get("status") == "PASS":
                return ""
            parts = []
            if verdict.get("missing_slots"):
                parts.append("missing_slot=" + ",".join(verdict["missing_slots"]))
            if verdict.get("incompatible_columns"):
                parts.append("incompatible_column=" + ",".join(verdict["incompatible_columns"]))
            if verdict.get("answer_shape_error"):
                parts.append("answer_shape=" + str(verdict["answer_shape_error"]))
            return "projection verifier failed: " + "; ".join(parts)

        if not rows or shape[0] == 0:
            return "FAIL", "empty structured result"
        if all(all(is_empty_value(v) for v in row) for row in rows):
            return "FAIL", "all result cells are empty"
        if structured.get("projection_truncated"):
            return "FAIL", "structured result is too large for full projection; narrow or aggregate before finishing"
        if structured.get("truncated") and not structured.get("projection_rows") and not (contract.operation == "rank" and contract.requires_single):
            return "FAIL", "structured result is truncated; narrow or aggregate before finishing"
        if contract.requires_single and shape[0] > 1 and contract.operation == "rank":
            return "FAIL", "ranking contract expects narrowed top/bottom candidates"
        if contract.answer_type == "number":
            if contract.operation in {"count", "sum", "avg"} and shape[0] != 1:
                return "FAIL", "aggregate numeric contract expects one scalar result"
            flat = [v for row in rows[:3] for v in row]
            if not any(str(v).strip().replace(".", "", 1).replace("-", "", 1).isdigit() for v in flat):
                return "FAIL", "numeric contract but result has no numeric value"
            p_error = projection_error()
            if p_error:
                return "FAIL", p_error
        if contract.operation == "support_classification":
            flat_text = [str(v).strip().lower() for row in rows[:3] for v in row]
            if not any(value in {"supported", "unsupported"} for value in flat_text):
                return "FAIL", "support classification contract expects Supported or Unsupported status"
            if shape[0] != 1:
                return "FAIL", "support classification contract expects one status row"
            p_error = projection_error()
            if p_error:
                return "FAIL", p_error
        if contract.answer_type in {"text", "email", "date"}:
            id_answer = query_asks_identifier(query)
            selected = self._select_answer_columns(query, contract, structured)
            if not selected:
                p_error = projection_error()
                if p_error:
                    return "FAIL", f"no plausible answer column; {p_error}"
                return "FAIL", "no plausible answer column"
            if all(is_id_like_column(col) for col in selected) and (
                not id_answer or not all(identifier_column_matches_query(query, col) for col in selected)
            ):
                return "FAIL", "selected answer columns look like ids"
            col_idx = [columns.index(c) for c in selected if c in columns]
            vals = [row[i] for row in rows for i in col_idx if i < len(row)]
            non_empty_vals = [v for v in vals if not is_empty_value(v)]
            if non_empty_vals and all(is_number_like_value(v) for v in non_empty_vals) and not id_answer:
                p_error = projection_error()
                if p_error:
                    return "FAIL", f"{p_error}; text answer contract but selected values are numeric/count-like"
                return "FAIL", "text answer contract but selected values are numeric/count-like"
            if any(str(v).strip().lower().startswith("no ") or "not found" in str(v).strip().lower() for v in non_empty_vals):
                return "FAIL", "text answer contract but result is an unsupported no-result string"
            if contract.answer_type == "email":
                if not any("@" in str(v) for v in vals):
                    return "FAIL", "email contract but no email-like value"
            p_error = projection_error()
            if p_error:
                return "FAIL", p_error
        return "PASS", ""

    def deterministic_project(self, query: str, contract: AnswerContract, structured: Mapping[str, Any]) -> str:
        rows = self._projection_rows(structured)
        columns = [str(c) for c in structured.get("columns", [])]
        if not rows:
            if contract.operation == "list_completion":
                return "[]"
            return ""
        if contract.operation == "rank" and contract.requires_single and len(rows) > 1:
            rows = rows[:1]
        if contract.operation == "support_classification":
            selected = self._select_answer_columns(query, contract, structured)
            scan_columns = selected or columns
            scan_idx = [columns.index(c) for c in scan_columns if c in columns]
            if not scan_idx:
                scan_idx = list(range(len(columns)))
            for row in rows[:3]:
                for idx in scan_idx:
                    if idx >= len(row):
                        continue
                    text = normalize_answer_text(row[idx]).lower()
                    if text == "supported":
                        return "Supported"
                    if text == "unsupported":
                        return "Unsupported"
                    if isinstance(row[idx], (bool, np.bool_)):
                        return "Supported" if bool(row[idx]) else "Unsupported"
        selected = self._select_answer_columns(query, contract, structured)
        if not selected:
            selected = columns[:1] or ["value"]
        selected_idx = [columns.index(c) for c in selected if c in columns]
        if not selected_idx and columns == ["value"]:
            selected_idx = [0]

        if len(selected_idx) > 1:
            rendered_rows = []
            for row in rows:
                parts = [normalize_answer_text(row[i]) for i in selected_idx if i < len(row) and not is_empty_value(row[i])]
                if parts:
                    rendered_rows.append(", ".join(parts))
            return "; ".join(dedupe_keep_order(rendered_rows))

        values = []
        idx = selected_idx[0] if selected_idx else 0
        for row in rows:
            if idx < len(row):
                values.append(row[idx])
        final_values = dedupe_keep_order(values)
        if contract.requires_single and final_values:
            return final_values[0]
        return ", ".join(final_values)

    @staticmethod
    def _candidate_answer_key(answer: Any) -> str:
        text = normalize_answer_text(answer).lower()
        if text in {"[]", "[ ]"}:
            return "[]"
        parts = [
            normalize_join_value(part)
            for part in re.split(r"[,;\n\t]+", text)
            if normalize_join_value(part)
        ]
        if len(parts) > 1:
            return " | ".join(sorted(parts))
        return normalize_join_value(text)

    @staticmethod
    def _answer_looks_error_like(answer: Any) -> bool:
        text = str(answer or "").strip().lower()
        if not text:
            return True
        return (
            text == "none"
            or text.startswith("error")
            or text.startswith("traceback")
            or "traceback" in text
            or "exception" in text
            or "failed to produce a structured result" in text
        )

    @staticmethod
    def _query_requests_full_table(query: str) -> bool:
        q = str(query or "").lower()
        return bool(
            re.search(r"\b(?:all information|all info|display all|show all|list all|entire row|full row|whole row)\b", q)
            or re.search(r"\ball\b.{0,30}\b(?:columns|fields|details|records|rows)\b", q)
        )

    @staticmethod
    def _render_projection_rows(rows: Sequence[Sequence[Any]], selected_idx: Sequence[int], requires_single: bool = False) -> str:
        if not rows:
            return "[]"
        if len(selected_idx) > 1:
            rendered_rows = []
            for row in rows:
                parts = [
                    normalize_answer_text(row[idx])
                    for idx in selected_idx
                    if idx < len(row) and not is_empty_value(row[idx])
                ]
                if parts:
                    rendered_rows.append(", ".join(parts))
            values = dedupe_keep_order(rendered_rows)
            if requires_single and values:
                return values[0]
            return "; ".join(values)

        values = []
        idx = selected_idx[0] if selected_idx else 0
        for row in rows:
            if idx < len(row) and not is_empty_value(row[idx]):
                values.append(row[idx])
        final_values = dedupe_keep_order(values)
        if requires_single and final_values:
            return normalize_answer_text(final_values[0])
        return ", ".join(normalize_answer_text(value) for value in final_values)

    @staticmethod
    def _render_labeled_projection_rows(
        rows: Sequence[Sequence[Any]],
        columns: Sequence[str],
        selected_idx: Sequence[int],
        requires_single: bool = False,
    ) -> str:
        if not rows:
            return "[]"
        rendered_rows = []
        for row in rows:
            parts = []
            for idx in selected_idx:
                if idx >= len(row) or idx >= len(columns) or is_empty_value(row[idx]):
                    continue
                parts.append(f"{columns[idx]}={normalize_answer_text(row[idx])}")
            if parts:
                rendered_rows.append("; ".join(parts))
        values = dedupe_keep_order(rendered_rows)
        if requires_single and values:
            return values[0]
        return " | ".join(values)

    def _rank_candidate_columns(
        self,
        query: str,
        contract: AnswerContract,
        structured: Mapping[str, Any],
        selected_columns: Sequence[str],
        slot_columns: Sequence[str],
    ) -> List[str]:
        columns = [str(c) for c in structured.get("columns", [])]
        rows = self._projection_rows(structured)
        if not columns:
            return []
        q_compact = compact_name(query)
        q_tokens = set(split_words(query))
        selected = set(selected_columns)
        slot_set = set(slot_columns)
        requested_id = query_asks_identifier(query)
        scored = []
        for idx, col in enumerate(columns):
            vals = [row[idx] for row in rows if idx < len(row)]
            non_empty = [v for v in vals if not is_empty_value(v)]
            col_compact = compact_name(col)
            score = 0.0
            if col in selected:
                score += 10.0
            if col in slot_set:
                score += 12.0
            try:
                score += max(0.0, self._column_score(query, contract, col, vals))
            except Exception:
                pass
            for term in contract.target_terms:
                if compact_name(term) and compact_name(term) in col_compact:
                    score += 5.0
            for slot in contract.required_output_slots or []:
                cues = [slot.get("name", ""), *(slot.get("cues") or [])]
                for cue in cues:
                    cue_compact = compact_name(cue)
                    if cue_compact and cue_compact in col_compact:
                        score += 4.0
            if col_compact and (col_compact in q_compact or any(token and token in col_compact for token in q_tokens)):
                score += 2.0
            if is_id_like_column(col):
                if requested_id and identifier_column_matches_query(query, col):
                    score += 14.0
                elif requested_id:
                    score += 3.0
                elif contract.answer_type != "number":
                    score -= 3.0
            if any(cue in col_compact for cue in ["name", "title", "surname", "firstname", "lastname", "color", "email", "address", "street"]):
                score += 4.0
            if any(cue in col_compact for cue in ["count", "total", "sum", "avg", "average", "price", "salary", "speed"]):
                score += 5.0 if contract.answer_type == "number" else 1.0
            if non_empty and all(is_number_like_value(v) for v in non_empty):
                score += 4.0 if contract.answer_type == "number" else -1.0
            if non_empty and all(is_empty_value(v) for v in vals):
                score -= 10.0
            scored.append((score, -idx, col))
        scored.sort(reverse=True)
        return [col for _, _, col in scored]

    def _candidate_column_combinations(
        self,
        query: str,
        contract: AnswerContract,
        structured: Mapping[str, Any],
        selected_columns: Sequence[str],
        slot_columns: Sequence[str],
    ) -> List[List[str]]:
        columns = [str(c) for c in structured.get("columns", [])]
        if len(columns) < 2:
            return []
        ranked = self._rank_candidate_columns(query, contract, structured, selected_columns, slot_columns)
        combos: List[List[str]] = []

        def add(cols: Sequence[str]) -> None:
            clean = [str(col) for col in cols if str(col) in columns]
            clean = dedupe_keep_order(clean)
            if len(clean) >= 2 and clean not in combos:
                combos.append(clean)

        add(selected_columns)
        add(slot_columns)
        requested_id = query_asks_identifier(query)
        id_cols = [col for col in ranked if is_id_like_column(col)]
        descriptor_cols = [
            col for col in ranked
            if col not in id_cols
            and any(cue in compact_name(col) for cue in ["name", "title", "surname", "firstname", "lastname", "color", "email", "address", "street"])
        ]
        if requested_id:
            for id_col in id_cols[:4]:
                for desc_col in descriptor_cols[:6]:
                    add([id_col, desc_col])
            for id_col in id_cols[:3]:
                for pair in combinations(descriptor_cols[:5], 2):
                    add([id_col, *pair])

        top = ranked[: min(8, len(ranked))]
        allow_generic_combos = (
            contract.multi_field
            or len(contract.required_output_slots or []) > 1
            or requested_id
            or int((structured.get("shape") or [0])[0] or 0) == 1
        )
        if allow_generic_combos:
            for width in range(2, min(3, len(top)) + 1):
                for combo in combinations(top, width):
                    add(combo)
                    if len(combos) >= 40:
                        return combos
        return combos

    def _candidate_from_columns(
        self,
        query: str,
        contract: AnswerContract,
        obs: StepObservation,
        selected_columns: Sequence[str],
        kind: str,
    ) -> Optional[Dict[str, Any]]:
        structured = obs.structured_result or {}
        columns = [str(c) for c in structured.get("columns", [])]
        rows = self._projection_rows(structured)
        selected = [str(col) for col in selected_columns if str(col) in columns]
        if not selected:
            return None
        selected_idx = [columns.index(col) for col in selected]
        answer = self._render_projection_rows(rows, selected_idx, requires_single=contract.requires_single)
        if self._answer_looks_error_like(answer):
            return None
        display_answer = answer
        if len(selected_idx) > 1:
            display_answer = self._render_labeled_projection_rows(
                rows, columns, selected_idx, requires_single=contract.requires_single
            ) or answer
        projected_rows = [
            [row[idx] if idx < len(row) else None for idx in selected_idx]
            for row in rows
        ]
        projected_structured = {
            "kind": "candidate_projection",
            "columns": selected,
            "rows": projected_rows[:20],
            "projection_rows": projected_rows,
            "shape": [len(projected_rows), len(selected)],
            "truncated": len(projected_rows) > 20,
            "projection_truncated": False,
            "candidate_selector": {
                "source_step_id": obs.step_id,
                "source_action": obs.action,
                "selected_columns": selected,
                "candidate_kind": kind,
            },
        }
        return {
            "id": "",
            "answer": answer,
            "display_answer": display_answer,
            "source_step_id": obs.step_id,
            "source_action": obs.action,
            "selected_columns": selected,
            "kind": kind,
            "structured_result": projected_structured,
        }

    def _add_candidate(
        self,
        candidates: List[Dict[str, Any]],
        seen: set,
        candidate: Optional[Dict[str, Any]],
    ) -> None:
        if not candidate:
            return
        key = self._candidate_answer_key(candidate.get("answer", ""))
        if not key or key in seen:
            return
        candidate["id"] = f"C{len(candidates)}"
        seen.add(key)
        candidates.append(candidate)

    def _build_candidate_selector_candidates(
        self,
        query: str,
        contract: AnswerContract,
        observations: Sequence[StepObservation],
        final_answer: str,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen: set = set()
        if final_answer and not self._answer_looks_error_like(final_answer):
            self._add_candidate(candidates, seen, {
                "id": "",
                "answer": final_answer,
                "source_step_id": None,
                "source_action": "current_answer",
                "selected_columns": [],
                "kind": "current_answer",
                "structured_result": {},
            })

        max_candidates = int(getattr(self, "candidate_selector_max_candidates", 24))
        full_table_requested = self._query_requests_full_table(query)
        for obs in observations:
            if obs.status != "PASS" or not obs.structured_result:
                continue
            structured = obs.structured_result
            columns = [str(c) for c in structured.get("columns", [])]
            rows = self._projection_rows(structured)
            if not columns:
                continue
            if not rows and contract.operation == "list_completion":
                self._add_candidate(candidates, seen, {
                    "id": "",
                    "answer": "[]",
                    "source_step_id": obs.step_id,
                    "source_action": obs.action,
                    "selected_columns": columns,
                    "kind": "empty_list",
                    "structured_result": {
                        "kind": "candidate_projection",
                        "columns": columns,
                        "rows": [],
                        "projection_rows": [],
                        "shape": [0, len(columns)],
                        "truncated": False,
                        "projection_truncated": False,
                        "candidate_selector": {
                            "source_step_id": obs.step_id,
                            "source_action": obs.action,
                            "selected_columns": columns,
                            "candidate_kind": "empty_list",
                        },
                    },
                })
                continue

            selected = self._select_answer_columns(query, contract, structured)
            self._add_candidate(candidates, seen, self._candidate_from_columns(
                query, contract, obs, selected, "contract_projection"
            ))
            if full_table_requested:
                self._add_candidate(candidates, seen, self._candidate_from_columns(
                    query, contract, obs, columns, "full_table_projection"
                ))

            if contract.required_output_slots:
                slot_columns = []
                _, selected_by_slot, _ = self._select_columns_for_slots(query, contract, structured)
                for cols in selected_by_slot.values():
                    slot_columns.extend(cols)
                self._add_candidate(candidates, seen, self._candidate_from_columns(
                    query, contract, obs, dedupe_keep_order(slot_columns), "slot_projection"
                ))
            else:
                slot_columns = []

            for combo in self._candidate_column_combinations(query, contract, structured, selected, slot_columns):
                self._add_candidate(candidates, seen, self._candidate_from_columns(
                    query, contract, obs, combo, "column_combo"
                ))
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

            for col in self._rank_candidate_columns(query, contract, structured, selected, slot_columns):
                self._add_candidate(candidates, seen, self._candidate_from_columns(
                    query, contract, obs, [col], "single_column"
                ))
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

        return candidates[:max_candidates]

    def _build_candidate_selector_prompt(self, query: str, candidates: Sequence[Mapping[str, Any]]) -> str:
        lines = []
        for candidate in candidates:
            answer = truncate_text(str(candidate.get("display_answer") or candidate.get("answer", "")), 1200)
            lines.append(f"{candidate.get('id')}: {answer}")
        return f"""Select the candidate answer that best answers the question.
Return exactly one minified JSON object and nothing else. Do not explain. Do not repeat the question or candidates.
Use only the question and candidate answers. If no candidate answers the question, select NONE.
Prefer the most exact answer shape: scalar counts for "how many" questions, all requested fields for multi-field questions, and candidates containing requested ids when the question asks for ids.

Question:
{query}

Candidate answers:
{chr(10).join(lines)}

Return JSON only:
{{"selection": "C0"}}
or
{{"selection": "NONE"}}
"""

    def _query_candidate_selector(self, prompt: str) -> str:
        model = self._candidate_selector_model
        if model is None:
            model = Model(self.candidate_selector_model_name)
            self._candidate_selector_model = model
        input_tokens = model.get_token_count(prompt)
        self.total_input_token_count += input_tokens
        if input_tokens > model.context_limit:
            return '{"selection": "NONE"}'
        raw = model.query(
            prompt=prompt,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            max_tokens=96,
        )
        self.total_output_token_count += model.get_token_count(raw)
        return raw

    @staticmethod
    def _parse_candidate_selector_choice(raw: Any) -> str:
        text = str(raw or "").strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, Mapping):
                for key in ["selection", "candidate", "candidate_id", "answer"]:
                    value = str(parsed.get(key, "")).strip().upper()
                    if value:
                        return value
        except Exception:
            pass
        upper = text.upper()
        explicit_candidate = re.search(
            r"(?:SELECTION|SELECT|CHOOSE|CHOICE|ANSWER|BEST(?:\s+CANDIDATE)?)\s*(?:IS|:|=)?\s*(C\d+)\b",
            upper,
        )
        if explicit_candidate:
            return explicit_candidate.group(1)
        conclusion_candidate = re.search(
            r"(?:SO|THEREFORE|THUS|HENCE|FINAL(?:\s+ANSWER)?|CONCLUSION)\s*,?\s*"
            r"(C\d+)\b\s*(?:IS|SEEMS|PROVIDES|CONTAINS|GIVES|ANSWERS|MATCHES)\b",
            upper,
        )
        if conclusion_candidate:
            return conclusion_candidate.group(1)
        explicit_none = re.search(
            r"(?:SELECTION|SELECT|CHOOSE|CHOICE|ANSWER|BEST(?:\s+CANDIDATE)?)\s*(?:IS|:|=)?\s*NONE\b",
            upper,
        )
        if explicit_none or re.fullmatch(r"\W*NONE\W*", upper):
            return "NONE"
        ids = re.findall(r"\bC\d+\b", upper)
        unique_ids = dedupe_keep_order(ids)
        if len(unique_ids) == 1:
            return unique_ids[0]
        return "NONE"

    def _candidate_selector_vote_threshold(self) -> int:
        samples = max(1, int(getattr(self, "candidate_selector_samples", 3)))
        frac = float(getattr(self, "candidate_selector_min_frac", 0.67))
        majority = samples // 2 + 1
        frac_threshold = int(round(samples * frac))
        return max(1, min(samples, max(majority if samples > 1 else 1, frac_threshold)))

    def _apply_candidate_selector(
        self,
        query: str,
        contract: AnswerContract,
        observations: List[StepObservation],
        final_answer: str,
        finish_reason: str,
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.enable_candidate_selector:
            return final_answer, finish_reason, {"enabled": False}
        candidates = self._build_candidate_selector_candidates(query, contract, observations, final_answer)
        diagnostics: Dict[str, Any] = {
            "enabled": True,
            "candidate_count": len(candidates),
            "samples": int(getattr(self, "candidate_selector_samples", 3)),
            "threshold": self._candidate_selector_vote_threshold(),
            "votes": {},
            "selected_candidate_id": "",
            "accepted": False,
        }
        if len(candidates) <= 1:
            diagnostics["reason"] = "not_enough_candidates"
            return final_answer, finish_reason, diagnostics

        by_id = {str(candidate["id"]).upper(): candidate for candidate in candidates}
        prompt = self._build_candidate_selector_prompt(query, candidates)
        votes: Counter = Counter()
        raw_choices = []
        for _ in range(int(getattr(self, "candidate_selector_samples", 3))):
            raw = self._query_candidate_selector(prompt)
            choice = self._parse_candidate_selector_choice(raw)
            raw_choices.append({"raw": str(raw), "choice": choice})
            if choice in by_id:
                votes.update([choice])
            elif choice == "NONE":
                votes.update(["NONE"])

        diagnostics["votes"] = dict(votes)
        diagnostics["raw_choices"] = raw_choices
        if not votes:
            diagnostics["reason"] = "no_valid_votes"
            return final_answer, finish_reason, diagnostics
        selected_id, vote_count = votes.most_common(1)[0]
        diagnostics["selected_candidate_id"] = selected_id
        diagnostics["selected_vote_count"] = vote_count
        if selected_id == "NONE":
            diagnostics["reason"] = "selector_chose_none"
            return final_answer, finish_reason, diagnostics
        if vote_count < diagnostics["threshold"]:
            diagnostics["reason"] = "below_vote_threshold"
            return final_answer, finish_reason, diagnostics

        selected = by_id.get(selected_id)
        selected_answer = str((selected or {}).get("answer", "")).strip()
        if not selected or self._answer_looks_error_like(selected_answer):
            diagnostics["reason"] = "selected_answer_rejected"
            return final_answer, finish_reason, diagnostics
        if self._candidate_answer_key(selected_answer) == self._candidate_answer_key(final_answer):
            diagnostics["reason"] = "same_as_current_answer"
            return final_answer, finish_reason, diagnostics

        selector_structured = dict(selected.get("structured_result") or {})
        selector_meta = dict(selector_structured.get("candidate_selector") or {})
        selector_meta.update({
            "candidate_id": selected_id,
            "votes": dict(votes),
            "vote_count": vote_count,
            "threshold": diagnostics["threshold"],
            "original_answer": final_answer,
            "selected_answer": selected_answer,
        })
        selector_structured["candidate_selector"] = selector_meta
        observations.append(StepObservation(
            step_id=len(observations) + 1,
            action="candidate_selector",
            thought="LLM selector chose a final projection from executed candidates.",
            status="PASS",
            structured_result=selector_structured,
            contract_status="PASS",
            contract_error="",
        ))
        diagnostics["accepted"] = True
        diagnostics["selected_answer"] = selected_answer
        diagnostics["selected_columns"] = selected.get("selected_columns", [])
        diagnostics["source_step_id"] = selected.get("source_step_id")
        return selected_answer, f"candidate_selector_from_step_{selected.get('source_step_id')}", diagnostics

    @staticmethod
    def _safe_merge_calls_from_code(code: str) -> List[Dict[str, Any]]:
        normalized = MinimalReActAgent._extract_code(code).replace("pd.safe_merge(", "safe_merge(")
        try:
            tree = ast.parse(normalized)
        except Exception:
            return []

        def literal_strings(node: Optional[ast.AST]) -> List[str]:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return [node.value]
            if isinstance(node, (ast.List, ast.Tuple)):
                return [
                    elt.value for elt in node.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
            return []

        def keyword_value(node: ast.Call, name: str) -> Optional[ast.AST]:
            for kw in node.keywords:
                if kw.arg == name:
                    return kw.value
            return None

        def expr_label(node: Optional[ast.AST]) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                return node.value.id
            return ""

        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "safe_merge":
                continue
            left_node = node.args[0] if len(node.args) >= 1 else keyword_value(node, "left_df")
            right_node = node.args[1] if len(node.args) >= 2 else keyword_value(node, "right_df")
            on_keys = literal_strings(keyword_value(node, "on"))
            left_keys = literal_strings(keyword_value(node, "left_on"))
            right_keys = literal_strings(keyword_value(node, "right_on"))
            if on_keys:
                left_keys = right_keys = on_keys
            calls.append({
                "left_input": expr_label(left_node),
                "right_input": expr_label(right_node),
                "left_on": left_keys,
                "right_on": right_keys,
            })
        return calls

    def _answer_source_observation(
        self,
        query: str,
        contract: AnswerContract,
        final_answer: str,
        observations: Sequence[StepObservation],
    ) -> Tuple[Optional[StepObservation], str, List[str]]:
        if not final_answer or final_answer.lower().startswith("error"):
            return None, "", []
        final_key = normalize_join_value(final_answer)
        for obs in reversed(observations):
            if obs.status != "PASS" or not obs.structured_result:
                continue
            status, _ = self.check_contract(query, contract, obs.structured_result)
            if status != "PASS":
                continue
            projected = self.deterministic_project(query, contract, obs.structured_result)
            if normalize_join_value(projected) == final_key:
                selected = self._select_answer_columns(query, contract, obs.structured_result)
                return obs, projected, selected
        return None, "", []

    @staticmethod
    def _cell_anchor_supports_phrase(anchor: Mapping[str, Any], phrase: str) -> bool:
        phrase_l = phrase.lower()
        value_l = str(anchor.get("value", "")).lower()
        if phrase_l and (phrase_l == value_l or phrase_l in value_l or value_l in phrase_l):
            return True
        matched = ";".join(str(item).lower() for item in anchor.get("matched", []))
        return phrase_l in matched

    @staticmethod
    def _relation_anchor_supports_merge(call: Mapping[str, Any], anchor: Mapping[str, Any]) -> bool:
        left_keys = [str(x) for x in call.get("left_on", [])]
        right_keys = [str(x) for x in call.get("right_on", [])]
        if not left_keys or not right_keys:
            return False
        pairs = set(zip(left_keys, right_keys))
        reverse_pairs = set(zip(right_keys, left_keys))
        anchor_pair = (str(anchor.get("left_col", "")), str(anchor.get("right_col", "")))
        anchor_reverse = (str(anchor.get("right_col", "")), str(anchor.get("left_col", "")))
        return anchor_pair in pairs or anchor_pair in reverse_pairs or anchor_reverse in pairs or anchor_reverse in reverse_pairs

    @staticmethod
    def _same_key_merge(call: Mapping[str, Any]) -> bool:
        left_keys = [compact_name(x) for x in call.get("left_on", [])]
        right_keys = [compact_name(x) for x in call.get("right_on", [])]
        return bool(left_keys and right_keys and left_keys == right_keys)

    def _verify_faithfulness(
        self,
        query: str,
        contract: AnswerContract,
        final_answer: str,
        observations: Sequence[StepObservation],
        cell_anchors: Sequence[Mapping[str, Any]],
        relation_anchors: Sequence[Mapping[str, Any]],
        finish_reason: str,
    ) -> Dict[str, Any]:
        if not self.enable_faithfulness_report:
            return {"status": "DISABLED", "mode": self.faithfulness_mode}

        source_obs, projected, selected_columns = self._answer_source_observation(query, contract, final_answer, observations)
        answer_from_execution = source_obs is not None
        reasons = []
        if not answer_from_execution:
            reasons.append("final_answer_not_recovered_from_pass_structured_result")

        phrases = extract_quoted_phrases(query)
        supported_phrases = []
        missing_phrases = []
        for phrase in phrases:
            if any(self._cell_anchor_supports_phrase(anchor, phrase) for anchor in cell_anchors):
                supported_phrases.append(phrase)
            else:
                missing_phrases.append(phrase)
        entity_anchor_supported = not missing_phrases
        if missing_phrases:
            reasons.append(f"missing_cell_anchor_for_quoted_phrase:{','.join(missing_phrases[:3])}")

        merge_calls = []
        supported_merge_calls = []
        unsupported_merge_calls = []
        for obs in observations:
            if not obs.code:
                continue
            for call in self._safe_merge_calls_from_code(obs.code):
                merge_calls.append(call)
                if self._same_key_merge(call) or any(
                    self._relation_anchor_supports_merge(call, anchor)
                    for anchor in relation_anchors
                ):
                    supported_merge_calls.append(call)
                else:
                    unsupported_merge_calls.append(call)
        join_path_supported = not unsupported_merge_calls
        if unsupported_merge_calls:
            reasons.append("safe_merge_join_key_not_supported_by_relation_anchor")

        projection_verified = False
        projection_complete = False
        projection_verdict = {
            "missing_slots": [],
            "incompatible_columns": [],
            "selected_slot_columns": {},
            "answer_shape_error": "",
        }
        operation_verified = False
        if source_obs is not None:
            status, contract_error = self.check_contract(query, contract, source_obs.structured_result)
            projection_verdict = self.verify_projection(
                query=query,
                contract=contract,
                structured_result=source_obs.structured_result,
                selected_columns=selected_columns,
                final_answer=final_answer,
            ) if getattr(self, "enable_projection_verifier", True) else {
                "status": "PASS",
                "missing_slots": [],
                "incompatible_columns": [],
                "selected_slot_columns": {},
                "answer_shape_error": "",
            }
            projection_complete = projection_verdict.get("status") == "PASS"
            projection_verified = status == "PASS" and bool(selected_columns) and projection_complete
            if not projection_verified:
                reasons.append(contract_error or "projection_contract_failed")
            if projection_verdict.get("missing_slots"):
                reasons.append("missing_target_slots:" + ",".join(projection_verdict["missing_slots"][:3]))
            if projection_verdict.get("incompatible_columns"):
                reasons.append("incompatible_answer_columns:" + ",".join(projection_verdict["incompatible_columns"][:3]))
            if projection_verdict.get("answer_shape_error"):
                reasons.append("answer_shape_error:" + str(projection_verdict["answer_shape_error"]))
            operation_verified = status == "PASS"
            if contract.operation in {"count", "sum", "avg"}:
                flat = [v for row in (source_obs.structured_result.get("rows") or [])[:3] for v in row]
                operation_verified = any(is_number_like_value(v) for v in flat)
            elif contract.operation == "rank" and contract.requires_single:
                shape = source_obs.structured_result.get("shape") or [0, 0]
                operation_verified = int(shape[0]) <= 1 or finish_reason.startswith("strict_final_recovery")
            if not operation_verified:
                reasons.append(f"operation_not_verified:{contract.operation}")

        checks = {
            "answer_from_execution": answer_from_execution,
            "entity_anchor_supported": entity_anchor_supported,
            "join_path_supported": join_path_supported,
            "projection_verified": projection_verified,
            "projection_complete": projection_complete,
            "operation_verified": operation_verified,
        }
        status = truth_status(checks, list(checks))
        return {
            "status": status,
            "mode": self.faithfulness_mode,
            **checks,
            "reasons": reasons,
            "source_step_id": source_obs.step_id if source_obs else None,
            "source_action": source_obs.action if source_obs else None,
            "projected_answer": projected,
            "selected_columns": selected_columns,
            "selected_slot_columns": projection_verdict.get("selected_slot_columns", {}),
            "missing_target_slots": projection_verdict.get("missing_slots", []),
            "incompatible_answer_columns": projection_verdict.get("incompatible_columns", []),
            "answer_shape_error": projection_verdict.get("answer_shape_error", ""),
            "supported_phrases": supported_phrases,
            "missing_phrases": missing_phrases,
            "merge_calls": merge_calls,
            "supported_merge_calls": supported_merge_calls,
            "unsupported_merge_calls": unsupported_merge_calls,
            "used_cell_anchors": [
                anchor.get("anchor_id")
                for anchor in cell_anchors
                if any(self._cell_anchor_supports_phrase(anchor, phrase) for phrase in phrases)
            ],
            "used_relation_anchors": [
                anchor.get("edge_id")
                for anchor in relation_anchors
                if any(self._relation_anchor_supports_merge(call, anchor) for call in supported_merge_calls)
            ],
        }

    def _strict_final_recovery(
        self,
        query: str,
        contract: AnswerContract,
        observations: Sequence[StepObservation],
    ) -> Tuple[str, str]:
        empty_list_candidate = ""
        empty_list_reason = ""
        for obs in reversed(observations):
            if obs.status != "PASS" or not obs.structured_result:
                continue
            structured = obs.structured_result
            rows = self._projection_rows(structured)
            shape = structured.get("shape") or [0, 0]
            if contract.operation == "list_completion" and (not rows or int(shape[0] or 0) == 0):
                empty_list_candidate = empty_list_candidate or "[]"
                empty_list_reason = empty_list_reason or f"strict_final_recovery_empty_list_from_step_{obs.step_id}"
                continue
            status, error = self.check_contract(query, contract, obs.structured_result)
            if status != "PASS":
                continue
            answer = self.deterministic_project(query, contract, obs.structured_result)
            if answer and not answer.lower().startswith("error"):
                return answer, f"strict_final_recovery_from_step_{obs.step_id}"
            if error:
                continue
        if empty_list_candidate:
            return empty_list_candidate, empty_list_reason
        return "", ""

    @staticmethod
    def _should_attempt_repair(observations: Sequence[StepObservation]) -> bool:
        if not observations:
            return True
        last = observations[-1]
        if last.status == "ERROR":
            return True
        if last.contract_status == "FAIL":
            reason = (last.error or last.contract_error or "").lower()
            return any(
                cue in reason
                for cue in [
                    "empty",
                    "missing",
                    "not in columns",
                    "contract",
                    "numeric",
                    "ids",
                    "no plausible",
                    "no-result",
                    "ranking",
                    "projection",
                    "missing_slot",
                    "incompatible_column",
                    "answer_shape",
                ]
            )
        return False

    def _build_repair_prompt(
        self,
        query: str,
        schema_text: str,
        relation_text: str,
        cell_evidence_text: str,
        contract: AnswerContract,
        observations: Sequence[StepObservation],
        df_names: Optional[Sequence[str]] = None,
    ) -> str:
        last = observations[-1] if observations else StepObservation(step_id=0, action="none")
        failure = last.error or last.contract_error or "no prior executable result"
        allowed_dfs = ", ".join(df_names or ["df_0"])
        if self.enable_safe_merge:
            repair_merge_rule = (
                "2. Use safe_merge for joins and keep all required final answer column(s), not just helper ids, counts, or rank metrics."
            )
            repair_allowed = (
                f"1. Use only {allowed_dfs}, previous variables, pandas as pd, numpy as np, and safe_merge.\n"
                "2. Assign the repaired result to step_result.\n"
                "3. Use safe_merge for joins; do not use pd.merge or DataFrame.merge."
            )
        else:
            repair_merge_rule = (
                "2. Use pandas joins or merges for joins and keep all required final answer column(s), not just helper ids, counts, or rank metrics."
            )
            repair_allowed = (
                f"1. Use only {allowed_dfs}, previous variables, pandas as pd, and numpy as np.\n"
                "2. Assign the repaired result to step_result.\n"
                "3. Use pandas merge/join operations with visible join keys."
            )
        projection_guidance = ""
        if self._projection_feedback_enabled() and last.structured_result:
            verdict = self.verify_projection(query, contract, last.structured_result)
            projection_guidance = json.dumps({
                "missing_slots": verdict.get("missing_slots", []),
                "incompatible_columns": verdict.get("incompatible_columns", []),
                "selected_slot_columns": verdict.get("selected_slot_columns", {}),
                "answer_shape_error": verdict.get("answer_shape_error", ""),
            }, ensure_ascii=False)
        return f"""You are repairing one failed pandas attempt for multi-table QA.
This is the only repair attempt. Fix only the necessary code.

Question:
{query}

Answer contract:
{json.dumps(contract.to_prompt_dict(), ensure_ascii=False)}

Available DataFrames and samples:
{schema_text}

Join and relation hints:
{relation_text}

Relevant cell evidence from full tables:
{cell_evidence_text}

Previous observations:
{self._schema_memory_summary(observations)}

Most recent failed code:
{last.code}

Failure to fix:
{failure}

Projection verifier guidance:
{projection_guidance or "No projection verifier details available."}

Repair rules:
1. Preserve all matching candidate ids with .isin(...) when a lookup value appears in multiple rows.
{repair_merge_rule}
3. If missing_slots is non-empty, add columns matching those requested target fields.
4. If incompatible_columns is non-empty, replace those output columns with columns matching the requested slots.

Return valid JSON only:
{{
  "thought": "one short reason for the repair",
  "action": "pandas_code",
  "code": "python code assigning step_result"
}}

Rules:
{repair_allowed}
4. Prefer relation-hint join keys when the previous result was empty or a join failed.
5. Use cell evidence only as row/value hints; verify with pandas operations over the full DataFrames.
6. Return only the final answer column(s) or a small table needed for deterministic projection.
"""

    def _attempt_one_shot_repair(
        self,
        query: str,
        schema_text: str,
        relation_text: str,
        cell_evidence_text: str,
        contract: AnswerContract,
        observations: List[StepObservation],
        dfs: Mapping[str, pd.DataFrame],
        memory: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not self.enable_one_shot_repair or not self._should_attempt_repair(observations):
            return "", "", memory
        prompt = self._build_repair_prompt(
            query,
            schema_text,
            relation_text,
            cell_evidence_text,
            contract,
            observations,
            df_names=list(dfs.keys()),
        )
        raw = self._query_agent(prompt)
        action = self._extract_json_action(raw)
        if str(action.get("action", "")).strip().lower() != "pandas_code":
            return "", "", memory

        code = str(action.get("code", ""))
        thought = str(action.get("thought", "one-shot repair")).strip()
        status, output_or_error, structured, memory = self._execute_code(code, dfs, memory)
        if status == "PASS":
            c_status, c_error = self.check_contract(query, contract, structured)
            stdout = output_or_error
            error = ""
        else:
            c_status, c_error = "FAIL", output_or_error
            stdout = ""
            error = output_or_error

        observations.append(StepObservation(
            step_id=len(observations) + 1,
            action="one_shot_repair",
            thought=thought,
            code=self._canonicalize_model_code(code),
            status=status,
            error=error,
            stdout=stdout,
            structured_result=structured,
            contract_status=c_status,
            contract_error=c_error,
        ))
        if status == "PASS" and c_status == "PASS":
            answer = self.deterministic_project(query, contract, structured)
            if answer:
                return answer, "one_shot_repair_contract_satisfied", memory
        return "", "", memory

    def _load_existing_result(self, log_path: str) -> Optional[Dict[str, Any]]:
        if not self.load_exist or not os.path.exists(log_path):
            return None
        with open(log_path) as fp:
            result = json.load(fp)
        if self.run_config_hash and result.get("run_config_hash") != self.run_config_hash:
            return None
        return result

    def run(self, data: Mapping[str, Any], sc_id: int = 0) -> Dict[str, Any]:
        os.makedirs(os.path.join(self.log_dir, "log"), exist_ok=True)
        log_path = os.path.join(self.log_dir, "log", f'{data["id"]}-{sc_id}.json')
        existing = self._load_existing_result(log_path)
        if existing is not None:
            return existing

        self.total_input_token_count = 0
        self.total_output_token_count = 0

        query = self.extract_query(data)
        label = self.extract_label(data)
        table_names = data.get("table_names", [])
        table_count = len(data.get("tables", []))
        contract = self.infer_contract(query, table_count=table_count)
        dfs, _ = self._build_dfs(data, build_schema=False)
        relation_text, relation_evidence_anchors = self._build_relation_evidence(dfs, table_names)
        cell_evidence_anchors = self._build_cell_evidence_anchors(query, dfs, data.get("table_names", []))
        cell_evidence_text = self._render_cell_evidence(cell_evidence_anchors)
        schema_text = self._render_schema_text(
            dfs=dfs,
            table_names=table_names,
            query=query,
            contract=contract,
            relation_anchors=relation_evidence_anchors,
            cell_anchors=cell_evidence_anchors,
        )
        memory: Dict[str, Any] = {}
        observations: List[StepObservation] = []
        final_answer = ""
        finish_reason = "max_steps"

        for step_id in range(1, contract.max_steps + 1):
            prompt = self._build_prompt(
                query,
                schema_text,
                relation_text,
                cell_evidence_text,
                contract,
                observations,
                df_names=list(dfs.keys()),
            )
            raw = self._query_agent(prompt)
            action = self._extract_json_action(raw)
            action_type = str(action.get("action", "pandas_code")).strip().lower()
            thought = str(action.get("thought", "")).strip()

            latest_structured = observations[-1].structured_result if observations else {}
            if action_type == "finish":
                status, error = self.check_contract(query, contract, latest_structured)
                obs = StepObservation(
                    step_id=step_id,
                    action="finish",
                    thought=thought,
                    status=status,
                    structured_result=latest_structured,
                    contract_status=status,
                    contract_error=error,
                )
                observations.append(obs)
                if status == "PASS":
                    final_answer = self.deterministic_project(query, contract, latest_structured)
                    finish_reason = "contract_satisfied"
                    break
                reveal = self._build_failure_schema_reveal(query, dfs, table_names, contract, observations)
                schema_text = self._append_schema_reveal(schema_text, reveal)
                continue

            code = str(action.get("code", ""))
            status, output_or_error, structured, memory = self._execute_code(code, dfs, memory)
            if status == "PASS":
                c_status, c_error = self.check_contract(query, contract, structured)
                stdout = output_or_error
                error = ""
            else:
                c_status, c_error = "FAIL", output_or_error
                stdout = ""
                error = output_or_error
            observations.append(StepObservation(
                step_id=step_id,
                action="pandas_code",
                thought=thought,
                code=self._canonicalize_model_code(code),
                status=status,
                error=error,
                stdout=stdout,
                structured_result=structured,
                contract_status=c_status,
                contract_error=c_error,
            ))
            if status != "PASS" or c_status != "PASS":
                reveal = self._build_failure_schema_reveal(query, dfs, table_names, contract, observations)
                schema_text = self._append_schema_reveal(schema_text, reveal)
            if status == "PASS" and c_status == "PASS":
                prompt = self._build_prompt(
                    query,
                    schema_text,
                    relation_text,
                    cell_evidence_text,
                    contract,
                    observations,
                    df_names=list(dfs.keys()),
                )
                raw_finish = self._query_agent(prompt)
                finish_action = self._extract_json_action(raw_finish)
                if str(finish_action.get("action", "")).strip().lower() == "finish":
                    final_answer = self.deterministic_project(query, contract, structured)
                    finish_reason = "contract_satisfied"
                    break

        if not final_answer:
            if self.enable_strict_recovery:
                final_answer, finish_reason = self._strict_final_recovery(query, contract, observations)
        if not final_answer:
            repair_answer, repair_reason, memory = self._attempt_one_shot_repair(
                query=query,
                schema_text=schema_text,
                relation_text=relation_text,
                cell_evidence_text=cell_evidence_text,
                contract=contract,
                observations=observations,
                dfs=dfs,
                memory=memory,
            )
            if repair_answer:
                final_answer = repair_answer
                finish_reason = repair_reason
        final_answer, finish_reason, candidate_selector = self._apply_candidate_selector(
            query=query,
            contract=contract,
            observations=observations,
            final_answer=final_answer,
            finish_reason=finish_reason,
        )
        if not final_answer:
            final_answer = "Error: Minimal ReAct failed to produce a structured result."

        faithfulness = self._verify_faithfulness(
            query=query,
            contract=contract,
            final_answer=final_answer,
            observations=observations,
            cell_anchors=cell_evidence_anchors,
            relation_anchors=relation_evidence_anchors,
            finish_reason=finish_reason,
        )

        result = {
            "id": data["id"],
            "sc_id": sc_id,
            "query": query,
            "table_names": data.get("table_names", []),
            "answer": final_answer,
            "label": label,
            "contract": contract.to_prompt_dict(),
            "relation_hints": relation_text,
            "cell_evidence": cell_evidence_text,
            "schema_pruning": self._schema_pruning_diagnostics,
            "cell_evidence_anchors": cell_evidence_anchors,
            "relation_evidence_anchors": relation_evidence_anchors,
            "candidate_selector": candidate_selector,
            "faithfulness": faithfulness,
            "faithfulness_status": faithfulness.get("status"),
            "trace": [obs.to_log() for obs in observations],
            "finish_reason": finish_reason,
            "n_iter": len(observations),
            "init_prompt_token_count": 0,
            "total_token_count": self.total_input_token_count + self.total_output_token_count,
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "run_config_hash": self.run_config_hash,
            "enable_relation_hints": self.enable_relation_hints,
            "relation_hints_min_tables": self.relation_hints_min_tables,
            "enable_cell_evidence": self.enable_cell_evidence,
            "cell_evidence_top_k": self.cell_evidence_top_k,
            "cell_evidence_max_value_len": self.cell_evidence_max_value_len,
            "cell_evidence_include_rows": self.cell_evidence_include_rows,
            "enable_faithfulness_report": self.enable_faithfulness_report,
            "faithfulness_mode": self.faithfulness_mode,
            "enable_projection_verifier": self.enable_projection_verifier,
            "projection_verifier_feedback": self.projection_verifier_feedback,
            "enable_strict_recovery": self.enable_strict_recovery,
            "enable_one_shot_repair": self.enable_one_shot_repair,
            "enable_safe_merge": self.enable_safe_merge,
            "enable_static_checks": self.enable_static_checks,
            "enable_answer_contracts": self.enable_answer_contracts,
            "enable_schema_pruning": self.enable_schema_pruning,
            "schema_pruning_max_columns": self.schema_pruning_max_columns,
            "schema_pruning_head_rows": self.schema_pruning_head_rows,
            "enable_longtablebench_diagnostic_rules": self.enable_longtablebench_diagnostic_rules,
            "enable_candidate_selector": self.enable_candidate_selector,
            "candidate_selector_samples": self.candidate_selector_samples,
            "candidate_selector_min_frac": self.candidate_selector_min_frac,
            "candidate_selector_max_candidates": self.candidate_selector_max_candidates,
            "candidate_selector_model_name": self.candidate_selector_model_name,
            "execution_timeout_seconds": self.execution_timeout_seconds,
        }
        if "orig_id" in data:
            result["orig_id"] = data["orig_id"]
        elif "id_" in data:
            result["orig_id"] = data["id_"]

        with open(log_path, "w") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2, default=str)
        with open(log_path.replace(".json", ".txt"), "w") as fp:
            fp.write(json.dumps({
                "query": query,
                "contract": contract.to_prompt_dict(),
                "relation_hints": relation_text,
                "cell_evidence": cell_evidence_text,
                "cell_evidence_anchors": cell_evidence_anchors,
                "relation_evidence_anchors": relation_evidence_anchors,
                "candidate_selector": candidate_selector,
                "faithfulness": faithfulness,
                "trace": [obs.to_log() for obs in observations],
                "answer": final_answer,
            }, ensure_ascii=False, indent=2, default=str))
        return result
