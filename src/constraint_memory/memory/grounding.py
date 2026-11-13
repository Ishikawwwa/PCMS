"""Deterministic literal and selected-row checks for the E4 evidence boundary."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any


NUMBER = re.compile(r"(?<![\w.,+\-])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?!\w|\.\d|,\d)")


# Only recognized US state annotations qualify for the benchmark's city alias.
# A generic parenthesis (airport, neighborhood, or arbitrary label) is not a state.
US_STATE_ANNOTATIONS = frozenset("""
Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|
Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|
Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|
New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|
Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|
Washington|West Virginia|Wisconsin|Wyoming|District of Columbia|
AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|
NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC
""".replace("\n", "").casefold().split("|"))


def city_match_rank(selected: Any, observed: Any) -> int:
    """Prefer literal city identity; allow only one trailing known state suffix.

    Different annotated states never alias. Bare names must match literally;
    names of restaurants/hotels use the separate conservative name matcher.
    Rank 2 is exact, 1 is the state-annotation alias, and 0 is no match.
    """
    if not isinstance(selected, str) or not isinstance(observed, str) or not selected or not observed:
        return 0
    if selected == observed:
        return 2

    def state_suffix(city: str) -> tuple[str, str | None]:
        match = re.fullmatch(r"([^()]+?)\s*\(([^()]+)\)", city)
        if match and match[2].strip().casefold() in US_STATE_ANNOTATIONS:
            return match[1].rstrip(), match[2].strip().casefold()
        return city, None

    selected_base, selected_state = state_suffix(selected)
    observed_base, observed_state = state_suffix(observed)
    if bool(selected_state) == bool(observed_state):
        return 0
    return int(selected_base == observed_base)


def entity_name_match_rank(selected: Any, observed: Any, *, allow_whitespace: bool = True) -> int:
    """Prefer literal names; permit only ASCII horizontal-whitespace aliases.

    Case, punctuation, and every non-space/tab character remain exact. This is
    only a candidate identity match: a unique complete visible row is required
    before any property can be grounded. Flight identifiers remain literal.
    """
    if not isinstance(selected, str) or not isinstance(observed, str) or not selected.strip(" \t") or not observed.strip(" \t"):
        return 0
    if selected == observed:
        return 2
    if not allow_whitespace:
        return 0
    normalize = lambda text: re.sub(r"[ \t]+", " ", text).strip(" \t")
    selected_name, observed_name = normalize(selected), normalize(observed)
    return int(bool(selected_name) and selected_name == observed_name)


def matching_entity_rows(entity: str, city: str | None, rows: list[dict], *,
                         name_key: str = "name", city_key: str | None = "city",
                         allow_name_whitespace: bool = True) -> tuple[list[int], int, int]:
    """Return unique matching row indices and their name/city identity ranks.

    Literal names take precedence over aliases. Within that name tier, retain
    the existing literal-city/state-annotation precedence. Complete raw rows
    are deduplicated only when identical; two differing names or any differing
    field remain ambiguous even if the requested property value agrees.
    Callers keep observation metadata separate and preserve the raw evidence.
    """
    ranked = []
    for index, raw in enumerate(rows):
        row = {str(key).casefold(): value for key, value in raw.items()}
        name_rank = entity_name_match_rank(entity, row.get(name_key.casefold(), ""),
                                           allow_whitespace=allow_name_whitespace)
        if not name_rank:
            continue
        city_rank = 2 if city_key is None or city is None else city_match_rank(city, row.get(city_key.casefold(), ""))
        if name_rank and city_rank:
            ranked.append((name_rank, city_rank, index))
    best = max(((name_rank, city_rank) for name_rank, city_rank, _ in ranked), default=(0, 0))
    unique = {json.dumps(rows[index], sort_keys=True, ensure_ascii=False, default=str): index
              for name_rank, city_rank, index in ranked if (name_rank, city_rank) == best}
    return list(unique.values()), *best


def decimal_value(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value).replace(",", "").replace("\u2212", "-"))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def numeric_literal_present(value: Any, text: str) -> bool:
    wanted = decimal_value(value)
    if wanted is None:
        return False
    normalized = text.replace("\u2212", "-")
    return any(decimal_value(match.group()) == wanted for match in NUMBER.finditer(normalized))


ATTRIBUTE_COLUMNS = {
    "price": ("price", "average cost"),
    "rating": ("rating", "aggregate rating", "review rate number"),
    "cuisine": ("cuisine", "cuisines"),
    "room_type": ("room_type", "room type"),
    "house_rules": ("house_rules", "house rules"),
    "minimum_nights": ("minimum_nights", "minimum nights"),
    "maximum_occupancy": ("maximum_occupancy", "maximum occupancy"),
}
TABLE_COLUMNS = {
    "name", "city", "flight number", "price", "average cost", "cuisines", "aggregate rating",
    "room type", "house_rules", "minimum nights", "maximum occupancy", "review rate number",
    "deptime", "arrtime", "actualelapsedtime", "distance", "origincityname", "destcityname", "flightdate",
}


def _dict_rows(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [row for item in value for row in _dict_rows(item)]
    if isinstance(value, dict):
        if any(str(key).casefold() in {"name", "flight number"} for key in value):
            return [value]
        return [row for item in value.values() for row in _dict_rows(item)]
    return []


def observed_rows(content: Any) -> list[dict]:
    """Read rows from JSON or the official right-aligned DataFrame display.

    Unrecognized formats fail closed. The fixed-width parser accepts only a
    complete known header; it does not guess fields from an entity/value pair
    that happens to appear somewhere in a larger result.
    """
    if not isinstance(content, str):
        return _dict_rows(content)
    try:
        rows = _dict_rows(json.loads(content))
        if rows:
            return rows
    except (ValueError, TypeError):
        pass
    lines = content.splitlines()
    for index, line in enumerate(lines):
        matches = []
        for column in sorted(TABLE_COLUMNS, key=len, reverse=True):
            for match in re.finditer(r"(?<!\w)" + re.escape(column) + r"(?!\w)", line, re.I):
                if not any(match.start() < end and start < match.end() for start, end, _ in matches):
                    matches.append((match.start(), match.end(), column))
        matches.sort()
        names = {column for _, _, column in matches}
        if not ({"name", "flight number"} & names) or not (set(ATTRIBUTE_COLUMNS["price"]) & names):
            continue
        cursor = 0
        complete = True
        for start, end, _ in matches:
            if line[cursor:start].strip():
                complete = False
            cursor = end
        if line[cursor:].strip():
            complete = False
        if not complete:
            continue
        rows = []
        for data_line in lines[index + 1:]:
            if not data_line.strip():
                continue
            cells = {}
            cursor = 0
            for _, end, column in matches:
                cells[column] = data_line[cursor:end].strip()
                cursor = end
            if data_line[cursor:].strip():
                # A wrapped/truncated/extra-column display cannot certify a row.
                return []
            if cells.get("name", cells.get("flight number", "")):
                rows.append(cells)
        return rows
    # The official dict formatter emits one complete object as key:value lines.
    pairs = {}
    for line in lines:
        match = re.fullmatch(r"\s*([^:]+):\s*(.*?)\s*", line)
        if not match or match[1].strip().casefold() in pairs:
            return []
        pairs[match[1].strip().casefold()] = match[2]
    return [pairs] if pairs and ({"name", "flight number"} & set(pairs)) else []


def property_row_proof(selected_value: Any, slot: str, attribute: str, value: Any, contents: list[Any]) -> tuple[bool, str]:
    if not isinstance(selected_value, str) or selected_value == "-":
        return False, "PROPERTY_WITHOUT_SUBMITTED_ENTITY"
    if slot == "transportation":
        flight = re.search(r"Flight Number:\s*(F\d+)\b", selected_value)
        if not flight:
            return False, "PROPERTY_UNVERIFIABLE_ENTITY"
        entity, city, name_key = flight[1], None, "flight number"
    else:
        parts = selected_value.rsplit(",", 1)
        entity = parts[0].strip()
        city = parts[1].strip() if len(parts) == 2 else None
        name_key = "name"
    columns = ATTRIBUTE_COLUMNS.get(attribute)
    if columns is None:
        return False, "PROPERTY_UNVERIFIABLE_ATTRIBUTE"
    rows = []
    any_rows = False
    for content in contents:
        visible_rows = observed_rows(content)
        any_rows |= bool(visible_rows)
        rows.extend(visible_rows)
    matching, _, _ = matching_entity_rows(entity, city, rows, name_key=name_key,
                                          city_key=None if city is None else "city",
                                          allow_name_whitespace=slot != "transportation")
    if not matching:
        return False, "PROPERTY_ENTITY_MISMATCH" if any_rows else "PROPERTY_UNVERIFIABLE_ROW"
    if len(matching) != 1:
        return False, "PROPERTY_AMBIGUOUS_ENTITY"
    row = {str(key).casefold(): item for key, item in rows[matching[0]].items()}
    present = [row[column] for column in columns if column in row]
    if len(present) != 1:
        return False, "PROPERTY_UNVERIFIABLE_ATTRIBUTE"
    observed = present[0]
    wanted_number = decimal_value(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    same = decimal_value(observed) == wanted_number if wanted_number is not None else str(observed).casefold() == str(value).casefold()
    return (True, "PROPERTY_ROW_CHECKED") if same else (False, "PROPERTY_VALUE_MISMATCH")
