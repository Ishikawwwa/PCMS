"""Explicit experiment schema, separate from the upstream CSV's literal facts.

The public numeric price columns do not certify their currency. This optional
protocol interprets them in USD for every condition; it does not change values.
"""
from __future__ import annotations

import copy

PRICE_SCHEMA_CONTRACT_TEXT = (
    "Declared property schema for this experiment: numeric price values returned "
    "by the search tools are interpreted in USD. This currency is a shared "
    "experiment declaration; copy the returned numeric values without conversion."
)


def declared_price_schema(unit):
    if unit != "USD":
        raise ValueError("The travel price schema must explicitly declare USD")
    return {"source_kind": "DECLARED_PROPERTY_SCHEMA", "price_unit": unit,
            "value_conversion": "none", "currency_verified_from_upstream": False}


def annotate_price_facts(attributes, schema_event):
    result = copy.deepcopy(attributes)
    if schema_event is None:
        return result
    schema = schema_event["content"]
    if (schema_event["type"] != "property_schema" or
            schema != declared_price_schema(schema.get("price_unit"))):
        raise ValueError("Price annotation requires the observed declared schema")
    source_ref = "event:" + schema_event["event_id"]
    for record in result.get("decisions", []):
        if record.get("attribute") != "price":
            continue
        record.update(unit=schema["price_unit"], unit_source_kind=schema["source_kind"],
                      unit_source_refs=[source_ref])
        record["source_refs"] = list(dict.fromkeys([*record.get("source_refs", []), source_ref]))
    return result
