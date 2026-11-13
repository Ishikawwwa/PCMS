"""Reversible JSON tables for exact constraint-frontier records.

Row order, missing versus null fields, nested keys, values and multiplicity are
preserved. Symbols abbreviate only repeated stable references and are defined
in the same visible context. This changes presentation, never stored memory.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import re


REFERENCE_PREFIXES = ("decision:", "constraint:", "definition:", "event:", "task-contract:")


def pack_sections(sections,reference_values=()):
    counts = Counter()
    references=set(reference_values)
    if any(not isinstance(value,str) for value in references):
        raise ValueError("Declared opaque references must be strings")

    def count(value):
        if isinstance(value, str):
            if value.startswith(REFERENCE_PREFIXES) or value in references:
                counts[value] += 1
        elif isinstance(value, list):
            for item in value: count(item)
        elif isinstance(value, dict):
            if any(not isinstance(key, str) for key in value): raise ValueError("Nested keys must be strings")
            for key, item in value.items(): count(key); count(item)
        elif value is not None and type(value) not in (bool, int, float):
            raise TypeError("Unsupported protected JSON value")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite protected JSON value")

    if not isinstance(sections, dict) or any(not isinstance(key, str) or not isinstance(rows, list)
                                           for key, rows in sections.items()):
        raise ValueError("Sections must map names to record lists")
    for rows in sections.values():
        for row in rows:
            if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
                raise ValueError("Each record must be an object with string keys")
            for value in row.values(): count(value)
    symbols = {}

    def string(value):
        if counts[value] > 1:
            symbols.setdefault(value, f"~{len(symbols)}")
            return symbols[value]
        return "~" + value if value.startswith("~") else value

    def nested(value):
        if isinstance(value, str): return string(value)
        if isinstance(value, list): return [nested(item) for item in value]
        if isinstance(value, dict): return {string(key): nested(item) for key, item in value.items()}
        return value

    tables = {}
    for section, rows in sections.items():
        schemas, index, encoded = [], {}, []
        for row in rows:
            columns = tuple(sorted(row))
            if columns not in index:
                index[columns] = len(schemas)
                schemas.append(list(columns))
            encoded.append([index[columns], [nested(row[key]) for key in columns]])
        tables[section] = {"columns": schemas, "rows": encoded}
    result = {"version": 1, "symbols": {symbol: value for value, symbol in symbols.items()}, "tables": tables}
    # Unsupported/nonfinite values cannot silently become strings or disappear.
    json.dumps(result, allow_nan=False)
    return result


def unpack_sections(packed):
    if not isinstance(packed, dict) or type(packed.get("version")) is not int or packed["version"] != 1:
        raise ValueError("Unsupported binding-table version")
    symbols, tables = packed.get("symbols"), packed.get("tables")
    if not isinstance(symbols, dict) or not isinstance(tables, dict):
        raise ValueError("Binding tables require symbols and tables")
    if any(not isinstance(key, str) or not re.fullmatch(r"~\d+", key) or not isinstance(value, str)
           for key, value in symbols.items()):
        raise ValueError("Invalid symbol dictionary")

    def string(value):
        if value.startswith("~~"): return value[1:]
        if value.startswith("~"):
            if value not in symbols: raise ValueError("Unknown binding-table symbol")
            return symbols[value]
        return value

    def nested(value):
        if isinstance(value, str): return string(value)
        if isinstance(value, list): return [nested(item) for item in value]
        if isinstance(value, dict): return {string(key): nested(item) for key, item in value.items()}
        return value

    result = {}
    for section, table in tables.items():
        if not isinstance(section, str) or not isinstance(table, dict): raise ValueError("Invalid section table")
        columns, rows = table.get("columns"), table.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list): raise ValueError("Invalid schemas/rows")
        if any(not isinstance(schema, list) or any(not isinstance(key, str) for key in schema)
               or len(set(schema)) != len(schema) for schema in columns): raise ValueError("Invalid column schema")
        restored = []
        for row in rows:
            if (not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int
                    or not 0 <= row[0] < len(columns) or not isinstance(row[1], list)
                    or len(row[1]) != len(columns[row[0]])):
                raise ValueError("Invalid table row/schema reference")
            restored.append(dict(zip(columns[row[0]], [nested(value) for value in row[1]], strict=True)))
        result[section] = restored
    return result


TABLE_GUIDANCE = ("EXACT BINDING TABLES: each section declares columns schemas. Every row is [schema_index, values]; "
    "match value positions to that schema's columns. Expand ~N using symbols in values and nested object keys; "
    "~~ escapes a literal ~. Missing columns differ from null. Preserve every exact value, unit, version, source "
    "and unresolved question. These facts do not prove prior requirements were satisfied.\n")


def serialize_sections(sections,reference_values=()):
    packed = pack_sections(sections,reference_values)
    if unpack_sections(packed) != sections:
        raise ValueError("Binding-table roundtrip changed protected records")
    return TABLE_GUIDANCE + json.dumps(packed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
