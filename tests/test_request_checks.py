"""Synthetic evidence for a limited adapter, separate from generic arithmetic."""
import copy
import json

import pytest

from constraint_memory.memory.request_checks import compile_request_checks, validate_request_checks


def decision(person, day, slot, value, *, attribute="value", version=1, step=0, group="g", entity_version=1, unit=None):
    rid = f"decision:{person}:day{day}:{slot}" + ("" if attribute == "value" else ":" + attribute)
    record = {"group_id": group, "decision_id": rid, "traveler": person, "day": day, "slot": slot,
        "attribute": attribute, "value": value, "version": version, "created_at": step,
        "status": "COMMITTED", "source_refs": [f"event:{person}:{attribute}:{version}"]}
    if attribute != "value": record["entity_ref"] = f"decision:{person}:day{day}:{slot}@v{entity_version}"
    if unit is not None: record["unit"] = unit
    return record


def entity(person="Lea", day=4, slot="accommodation", price=900, rating=3, *, version=1, step=0, group="g", unit="USD"):
    return [decision(person, day, slot, "North Harbor House, Wellington", version=version, step=step, group=group),
        decision(person, day, slot, price, attribute="price", unit=unit, version=version, entity_version=version, step=step, group=group),
        decision(person, day, slot, rating, attribute="rating", version=version, entity_version=version, step=step, group=group)]


def request(query, *, person="Noah", prior=None, step=1, group="g"):
    return {"group_id": group, "traveler": person, "step": step, "query": query,
        "prior_travelers": prior or ["Lea"], "source_refs": [f"event:request:{group}:{step}"]}


def actual(price=800, rating=3, *, person="Noah", day=1, slot="accommodation", step=2, unit="USD"):
    return entity(person, day, slot, price, rating, step=step, unit=unit)


def test_dollar_window_and_rating_are_individually_certified_from_different_target_day():
    raw = request("For my first-day accommodation, I want a place priced within $100 of Lea's day-4 accommodation and rated at least as high.")
    known = entity()
    compiled = compile_request_checks(raw, known)
    assert len(compiled["predicates"]) == 2
    assert {p["subject"]["day"] for p in compiled["predicates"]} == {1}
    assert all("day4:accommodation" in p["target"]["ref"] for p in compiled["predicates"])
    assert not compiled["full_request_semantics_verified"]
    checked = validate_request_checks(compiled, [*known, *actual()], step=2)
    assert checked["status"] == "PASSED"
    assert checked["task_success"] is None
    assert checked["checked_predicate_count"] == 2
    json.dumps(compiled, allow_nan=False)
    json.dumps(checked, allow_nan=False)


def test_exact_decimal_boundary_and_independent_numeric_violation_evidence():
    raw = request("For my first-day accommodation, I want a place priced within $0.1 of Lea's day4 accommodation and rated at least as high.")
    known = entity(price=100, rating=3)
    compiled = compile_request_checks(raw, known)
    passed = validate_request_checks(compiled, [*known, *actual(price=99.9)], step=2)
    assert passed["status"] == "PASSED"
    failed = validate_request_checks(compiled, [*known, *actual(price=99.89, rating=2.9)], step=2)
    assert len(failed["violations"]) == 2
    violation = failed["violations"][0]
    assert violation["subject"]["traveler"] == "Noah"
    assert violation["current_entity_ref"] == "decision:Noah:day1:accommodation@v1"
    assert violation["actual_ref"] == "decision:Noah:day1:accommodation:price@v1"
    assert violation["target_ref"] == "decision:Lea:day4:accommodation:price@v1"
    assert violation["actual_source_refs"] and violation["target_source_refs"]
    assert violation["evidence_quote"] == raw["query"]


@pytest.mark.parametrize("text, expected_ops", [
    ("For accommodation on the second day, I want a place priced within 30% of Lea's first-day accommodation, with a rating within 0.5.", ["GTE", "LTE", "ABS_DIFF_LTE"]),
    ("For lunch on day4, I want a place priced within 10% of Lea's third-day dinner but rated lower.", ["GTE", "LTE", "LT"]),
    ("For breakfast on day5, I want a place priced within 10% of Lea's third-day dinner and rated higher.", ["GTE", "LTE", "GT"]),
])
def test_percentage_and_rating_cues_keep_exact_scales_and_directions(text, expected_ops):
    known = [*entity(day=1), *entity(day=3, slot="dinner")]
    compiled = compile_request_checks(request(text), known)
    assert [p["operator"] for p in compiled["predicates"]] == expected_ops
    from decimal import Decimal
    scales = [Decimal(p["target"]["scale"]) for p in compiled["predicates"][:2]]
    assert scales in ([Decimal("0.7"), Decimal("1.3")], [Decimal("0.9"), Decimal("1.1")])


@pytest.mark.parametrize("word, operator, offset", [("less", "LTE", "-15"), ("more", "GTE", "15")])
def test_directional_price_offset_and_rating_difference(word, operator, offset):
    raw = request(f"For dinner on the third day, I want a place rated within 0.3 of Lea's first-day lunch and priced at least $15 {word}.")
    compiled = compile_request_checks(raw, entity(day=1, slot="lunch", price=36, rating=3.5))
    price = next(p for p in compiled["predicates"] if p["subject"]["attribute"] == "price")
    assert price["operator"] == operator
    assert price["target"]["offset"] == offset
    assert price["subject"]["day"] == 3 and price["subject"]["slot"] == "dinner"
    assert "day1:lunch:price" in price["target"]["ref"]


@pytest.mark.parametrize("text, slot", [
    ("For lunch on day1, I'd like to join Lea.", "lunch"),
    ("For first-day accommodation, I'd like to stay with Lea.", "accommodation"),
    ("On day1, I want to share accommodation with Lea.", "accommodation"),
    ("On day1, I hope to join Lea for dinner.", "dinner"),
    ("For lunch on day1, I want to eat with Lea.", "lunch"),
])
def test_supported_complete_shared_choice_suffix(text, slot):
    known = entity(day=1, slot=slot)
    compiled = compile_request_checks(request(text), known)
    assert len(compiled["predicates"]) == 1
    assert compiled["predicates"][0]["operator"] == "EQUALS"
    assert compiled["predicates"][0]["subject"]["attribute"] == "value"
    choice = decision("Noah", 1, slot, "North Harbor House, Wellington", step=2)
    assert validate_request_checks(compiled, [*known, choice], step=2)["status"] == "PASSED"
    choice["value"] = "Other Place, Dublin"
    assert validate_request_checks(compiled, [*known, choice], step=2)["status"] == "VIOLATED"


@pytest.mark.parametrize("query", [
    "For day1 accommodation, I do not want a place priced within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I never want a place priced within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I don't want a place priced within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I want a place that doesn't have a price within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I want a place priced within $100 unless Lea's day4 accommodation changes.",
    "For day1 accommodation, I'm considering a place priced within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I want to avoid a place priced within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I wonder whether a place is priced within $100 of Lea's day4 accommodation.",
    "For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation and another within $100 of Sam's day3 accommodation.",
    "For day1 accommodation, I want a place priced within $100 of Future's day4 accommodation.",
    "For day1 accommodation, I want a place priced within $100 of Lea's.",
    "For day1 accommodation, I want a place priced within $100 of her accommodation.",
    "For day1 accommodation, I want a place near Lea's day4 hotel and dinner at another venue.",
    "For day1 accommodation, I want it cheaper.",
])
def test_negation_unknown_ambiguity_anaphora_and_unsupported_wording_are_never_hardchecked(query):
    compiled = compile_request_checks(request(query, prior=["Lea", "Sam"]), [*entity(), *entity("Sam", day=3)])
    assert compiled["predicates"] == []
    assert compiled["unverified"] or compiled["scope_uncertainty"]
    assert validate_request_checks(compiled, [], step=2)["status"] == "UNVERIFIED"


def test_synonyms_paragraph_split_unrelated_person_and_city_remain_data():
    text = "I am Mina.\nI'm traveling with Omar.\nFor dinner on day3, I want a place costing at least $12 more than Omar's breakfast on day2.\n\nFor accommodation on day4, I'd like to stay with Omar."
    known = [*entity("Omar", day=2, slot="breakfast", price=40), *entity("Omar", day=4)]
    for record in known:
        if record["attribute"] == "value": record["value"] = "Copper Harbor Inn, Reykjavik"
    raw = request(text, person="Mina", prior=["Omar"])
    compiled = compile_request_checks(raw, known)
    assert len(compiled["predicates"]) == 2
    assert {p["subject"]["traveler"] for p in compiled["predicates"]} == {"Mina"}
    assert {p["target"]["ref"].split(":")[1] for p in compiled["predicates"]} == {"Omar"}
    for predicate in compiled["predicates"]:
        evidence = predicate["predicate_evidence"]
        assert text[evidence["start"]:evidence["end"]] == evidence["quote"]


@pytest.mark.parametrize("change", ["missing_property", "future", "group", "missing_group", "ambiguous_version", "entity_version", "missing_source", "unknown_unit"])
def test_unverifiable_reference_records_do_not_enable_a_price_check(change):
    known = entity()
    if change == "missing_property": known = [r for r in known if r["attribute"] != "price"]
    elif change == "future":
        for record in known: record["created_at"] = 9
    elif change == "group":
        for record in known: record["group_id"] = "other"
    elif change == "missing_group":
        for record in known: record.pop("group_id")
    elif change == "ambiguous_version": known.append({**known[1], "value": 901})
    elif change == "entity_version": known[1]["entity_ref"] = "decision:Lea:day4:accommodation@v9"
    elif change == "missing_source": known[1]["source_refs"] = []
    else: known[1].pop("unit")
    compiled = compile_request_checks(request("For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation."), known)
    assert compiled["predicates"] == []
    assert compiled["unverified"]


def test_explicit_evaluation_step_allows_actual_after_compilation_but_never_future_values():
    known = entity()
    compiled = compile_request_checks(request("For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation."), known)
    records = [*known, *actual(step=5)]
    assert validate_request_checks(compiled, records)["status"] == "UNVERIFIED"
    assert validate_request_checks(compiled, records, step=5)["status"] == "PASSED"
    with pytest.raises(ValueError): validate_request_checks(compiled, records, step=0)
    with pytest.raises(ValueError): validate_request_checks(compiled, records, step=True)


def test_changed_target_version_is_unverified_instead_of_stale_hard_comparison():
    old = entity()
    compiled = compile_request_checks(request("For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation."), old)
    new = entity(price=1000, version=2, step=2)
    checked = validate_request_checks(compiled, [*old, *new, *actual()], step=2)
    assert checked["violations"] == []
    assert checked["unverified"][0]["code"] == "DEPENDENCY_CHANGED"


def test_actual_unit_and_property_entity_binding_are_verified_and_inputs_unchanged():
    known = entity()
    raw = request("For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation.")
    compiled = compile_request_checks(raw, known)
    records = [*known, *actual(unit="EUR")]
    before = copy.deepcopy((raw, known, compiled, records))
    checked = validate_request_checks(compiled, records, step=2)
    assert checked["unverified"][0]["code"] == "UNITS_MISMATCH"
    assert (raw, known, compiled, records) == before
    records[-2]["unit"] = "USD"
    records[-2]["entity_ref"] = "decision:Noah:day1:accommodation@v9"
    assert validate_request_checks(compiled, records, step=2)["unverified"][0]["code"] == "PROPERTY_ENTITY_VERSION_UNVERIFIED"


def test_future_target_revision_does_not_shadow_visible_version_at_explicit_step():
    known = entity()
    compiled = compile_request_checks(request("For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation."), known)
    records = [*known, *entity(price=5000, version=2, step=9), *actual(step=2)]
    assert validate_request_checks(compiled, records, step=2)["status"] == "PASSED"


def test_unsupported_local_cuisine_policy_does_not_gain_fake_interpretation_completeness():
    compiled = compile_request_checks(request("For day1 dinner, I want a vegan restaurant with a quiet room."), entity())
    assert not compiled["full_request_semantics_verified"]
    assert compiled["predicates"] == []
    assert validate_request_checks(compiled, [], step=2)["task_success"] is None


@pytest.mark.parametrize("query", [
    "For dinner on day1, I want a place near Lea's day4 dinner and priced within $100 of the usual meal price.",
    "For dinner on day1, I want a place near Lea's day4 dinner and priced within 10% of the usual meal price.",
    "For dinner on day1, I want a place near Lea's day4 dinner and rated within 0.3 of the review average.",
    "For dinner on day1, I want a place near Lea's day4 dinner and priced at least $15 less than the typical venue.",
    "For dinner on day1, I want a place near Lea's day4 dinner and rated lower than the review average.",
])
def test_a_different_explicit_comparand_is_never_rebound_to_an_incidental_person(query):
    compiled = compile_request_checks(request(query), entity(slot="dinner"))
    assert compiled["predicates"] == []
    assert compiled["unverified"]


@pytest.mark.parametrize("number", ["0.3e2", "0.3%", "0.3stars", "-0.3"])
def test_unsupported_rating_number_spellings_are_not_silently_truncated(number):
    compiled = compile_request_checks(request(f"For day1 accommodation, I want a rating within {number} of Lea's day4 accommodation."), entity())
    assert compiled["predicates"] == []
    assert compiled["unverified"]


def test_nonfinite_record_data_cannot_escape_into_json_diagnostics():
    known = entity(price=float("nan"))
    compiled = compile_request_checks(request("For day1 accommodation, I want a place priced within $100 of Lea's day4 accommodation."), known)
    assert compiled["predicates"] == []
    json.dumps(compiled, allow_nan=False)


@pytest.mark.parametrize("query", [
    "For dinner on day1, I want a place costing at least $10 more in total, while Lea's day2 dinner is nearby.",
    "For dinner on day1, I want a place costing at least $10 less in total, while Lea's day2 dinner is nearby.",
    "For dinner on day1, I want a place rated within 0.3 on the review site, while Lea's day2 dinner is nearby.",
    "For dinner on day1, I want a place rated lower in reviews, while Lea's day2 dinner is nearby.",
    "For dinner on day1, I want a place rated higher overall, while Lea's day2 dinner is nearby.",
    "For dinner on day1, I want a place near Lea's day2 dinner and priced at least $10 more.",
    "For dinner on day1, I want a place near Lea's day2 dinner and priced at least $10 less.",
    "For dinner on day1, I want a place near Lea's day2 dinner and rated within 0.3.",
    "For dinner on day1, I want a place near Lea's day2 dinner and rated lower.",
    "For dinner on day1, I want a place near Lea's day2 dinner and rated higher.",
])
def test_incidental_earlier_or_later_person_is_not_a_numeric_comparand(query):
    compiled = compile_request_checks(request(query), entity(day=2, slot="dinner"))
    assert compiled["predicates"] == []
    assert compiled["unverified"]
    assert validate_request_checks(compiled, [], step=2)["status"] == "UNVERIFIED"


@pytest.mark.parametrize("tail", [
    "and rated lower in reviews.",
    "and rated higher overall.",
    "and priced at least $10 more in total.",
    "and priced at least $10 less on the meal deal.",
    "and has a rating within 0.3 according to the review site.",
])
def test_shared_anchor_ellipsis_rejects_an_independent_or_qualified_basis(tail):
    query = "For dinner on day1, I want a place priced within $100 of Lea's day2 dinner " + tail
    compiled = compile_request_checks(request(query), entity(day=2, slot="dinner"))
    assert len(compiled["predicates"]) == 1
    assert compiled["predicates"][0]["operator"] == "ABS_DIFF_LTE"
    assert compiled["predicates"][0]["subject"]["attribute"] == "price"


@pytest.mark.parametrize("query, expected", [
    ("For dinner on day1, I want a place priced at least $10 more than Lea's day2 dinner.", {"price": "GTE"}),
    ("For dinner on day1, I want a place priced at least $10 less than Lea's day2 dinner.", {"price": "LTE"}),
    ("For dinner on day1, I want a place rated within 0.3 of Lea's day2 dinner and priced at least $10 less.", {"price": "LTE", "rating": "ABS_DIFF_LTE"}),
    ("For dinner on day1, I want a place priced within $100 of Lea's day2 dinner and rated higher.", {"price": "ABS_DIFF_LTE", "rating": "GT"}),
    ("For dinner on day1, I want a place priced within 10% of Lea's day2 dinner, with a rating within 0.3.", {"price": "LTE", "rating": "ABS_DIFF_LTE"}),
])
def test_direct_comparands_and_narrow_shared_comparison_ellipsis_remain_supported(query, expected):
    compiled = compile_request_checks(request(query), entity(day=2, slot="dinner"))
    assert {p["subject"]["attribute"]: p["operator"] for p in compiled["predicates"]} == expected
