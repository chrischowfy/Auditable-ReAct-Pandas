#!/usr/bin/env python3
"""Shared MMQA evaluation and data helpers.

This module is intentionally dependency-light so every runner, judge, and
summarizer can use exactly the same exact-match semantics. In particular it
flattens SQL-result shaped labels such as ``{"columns": ..., "data": ...}``
before normalization.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sqlite3
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def normalize_atom(value: Any) -> str:
    text = str(value).strip()
    numeric = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", numeric):
        try:
            number = float(numeric)
            if math.isfinite(number) and number.is_integer():
                return str(int(number))
            return f"{number:.12g}"
        except Exception:
            pass
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_literal_text(text: str) -> Any:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return None


def answer_items(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(answer_items(item))
        return out
    if isinstance(value, Mapping) and "data" in value:
        out = []
        for row in value.get("data") or []:
            out.extend(answer_items(row))
        return out
    if isinstance(value, Mapping):
        out = []
        for item in value.values():
            out.extend(answer_items(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    if text[:1] in "{[(" and text[-1:] in "}])":
        parsed = _parse_literal_text(text)
        if isinstance(parsed, (dict, list, tuple, set)):
            return answer_items(parsed)
    separator = ";" if ";" in text else "," if "," in text else ""
    if separator:
        return [part for part in text.split(separator)]
    return [text]


def normalize_answer(value: Any) -> str:
    parts = [normalize_atom(item) for item in answer_items(value)]
    parts = [part for part in parts if part]
    if not parts:
        return normalize_atom(value)
    return " | ".join(sorted(parts))


def exact_match(prediction: Any, label: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(label)


def render_answer_for_judge(value: Any) -> str:
    parts = [str(item).strip() for item in answer_items(value)]
    parts = [part for part in parts if part]
    if not parts:
        return str(value)
    return ", ".join(parts)


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "samples", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON record container: {path}")


def get_sample_id(item: Mapping[str, Any], idx: int = 0) -> str:
    return str(item.get("id") or item.get("id_") or item.get("qid") or f"sample-{idx}")


def get_question(item: Mapping[str, Any]) -> str:
    return str(item.get("Question") or item.get("question") or item.get("query") or "")


def get_label(item: Mapping[str, Any]) -> Any:
    for key in ("answer", "label", "gold", "gold_answer"):
        if key in item:
            return item[key]
    return ""


def get_tables(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    tables = item.get("tables") or []
    return list(tables) if isinstance(tables, list) else []


def get_table_names(item: Mapping[str, Any]) -> list[str]:
    tables = get_tables(item)
    names = item.get("table_names") or []
    if isinstance(names, list) and len(names) == len(tables):
        return [str(name) for name in names]
    return [f"table_{i}" for i in range(len(tables))]


def table_columns(table: Mapping[str, Any]) -> list[str]:
    for key in ("table_columns", "columns", "header"):
        value = table.get(key)
        if isinstance(value, list):
            return [str(col) for col in value]
    rows = table.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        return [str(col) for col in rows[0]]
    return []


def table_rows(table: Mapping[str, Any]) -> list[list[Any]]:
    if isinstance(table.get("table_content"), list):
        return [list(row) for row in table.get("table_content") or []]
    if isinstance(table.get("rows"), list):
        rows = table.get("rows") or []
        if rows and isinstance(rows[0], list) and table.get("header") is None and table.get("columns") is None:
            return [list(row) for row in rows[1:]]
        return [list(row) for row in rows]
    return []


def make_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {
        "table_columns": [str(col) for col in columns],
        "table_content": [list(row) for row in rows],
    }


def sample_size_stats(item: Mapping[str, Any]) -> dict[str, Any]:
    tables = get_tables(item)
    rows = [len(table_rows(table)) for table in tables]
    cols = [len(table_columns(table)) for table in tables]
    return {
        "tables": len(tables),
        "rows_total": sum(rows),
        "columns_total": sum(cols),
        "rows_per_table": rows,
        "columns_per_table": cols,
        "approx_context_tokens": max(1, sum(r * max(1, c) for r, c in zip(rows, cols)) // 3),
    }


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def infer_sqlite_type(values: Iterable[Any]) -> str:
    checked = 0
    numeric = 0
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        checked += 1
        try:
            float(text.replace(",", ""))
            numeric += 1
        except Exception:
            pass
        if checked >= 50:
            break
    if checked and numeric / checked >= 0.85:
        return "NUMERIC"
    return "TEXT"


def materialize_sqlite(conn: sqlite3.Connection, item: Mapping[str, Any]) -> None:
    for name, table in zip(get_table_names(item), get_tables(item)):
        cols = table_columns(table)
        rows = table_rows(table)
        if not cols:
            continue
        col_defs = []
        for idx, col in enumerate(cols):
            values = [row[idx] for row in rows if idx < len(row)]
            col_defs.append(f"{quote_ident(col)} {infer_sqlite_type(values)}")
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(name)}")
        conn.execute(f"CREATE TABLE {quote_ident(name)} ({', '.join(col_defs)})")
        if rows:
            placeholders = ", ".join(["?"] * len(cols))
            conn.executemany(
                f"INSERT INTO {quote_ident(name)} VALUES ({placeholders})",
                [tuple(row[: len(cols)] + [None] * max(0, len(cols) - len(row))) for row in rows],
            )
    conn.commit()


@dataclass
class SQLExecutionResult:
    ok: bool
    rows: list[list[Any]]
    columns: list[str]
    normalized: str
    error: str = ""

    def as_label_object(self) -> dict[str, Any]:
        return {"columns": self.columns, "data": self.rows}


def execute_sql(item: Mapping[str, Any], sql: str | None = None) -> SQLExecutionResult:
    query = str(sql if sql is not None else item.get("SQL") or "").strip()
    if not query:
        return SQLExecutionResult(False, [], [], "", "missing SQL")
    conn = sqlite3.connect(":memory:")
    try:
        materialize_sqlite(conn, item)
        cursor = conn.execute(query)
        rows = [list(row) for row in cursor.fetchall()]
        columns = [desc[0] for desc in (cursor.description or [])]
        label_obj = {"columns": columns, "data": rows}
        return SQLExecutionResult(True, rows, columns, normalize_answer(label_obj))
    except Exception as exc:
        return SQLExecutionResult(False, [], [], "", f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
