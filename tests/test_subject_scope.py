"""Current slot coverage never inherits a named prior target's day or slot."""
import copy

import pytest

from constraint_memory.memory.subject_scope import compile_scope_coverage


def request(query, traveler="Current", **extra):
    return {"group_id": "g", "traveler": traveler, "step": 4, "query": query, **extra}


def compile(query, prior=None, traveler="Current"):
    return compile_scope_coverage(request(query, traveler), prior or ["Base", "Benjamin"], ["event:g:4"])


def subjects(result):
    return {(row["subject"]["day"], row["subject"]["slot"]) for row in result["scopes"]}


def test_prior_target_day_and_slot_are_never_current_exceptions():
    result = compile("I am Current.\nI'm traveling with Base and Benjamin.\nFor dinner on the third day, I want a place rated within 0.3 of Benjamin's first-day lunch and priced at least $15 less.")
    assert result["coverage_complete"]
    assert subjects(result) == {(3, "dinner")}
    assert not result["relational_semantics_interpreted"]
    assert not result["policy_semantics_interpreted"]


@pytest.mark.parametrize("day", ["first day", "first-day", "day1", "day 1", "day-1"])
def test_supported_explicit_day_spellings(day):
    result = compile(f"For my {day} accommodation, I'd like to stay with Benjamin.")
    assert result["coverage_complete"]
    assert subjects(result) == {(1, "accommodation")}


@pytest.mark.parametrize("query, expected", [
    ("On the second day, for breakfast, I'm looking for a Tea restaurant rated 2.9 to 3.1.", (2, "breakfast")),
    ("On day4, I hope to join Benjamin for dinner.", (4, "dinner")),
    ("I'd like to have lunch on day 5 with Benjamin.", (5, "lunch")),
    ("On the second day, I'd like to stay somewhere priced within 30% of Benjamin's first-day accommodation.", (2, "accommodation")),
])
def test_current_scope_order_and_narrow_stay_cue(query, expected):
    result = compile(query)
    assert result["coverage_complete"]
    assert subjects(result) == {expected}


def test_multiple_sentences_decimal_and_literal_name_periods_keep_exact_spans():
    query = "I am Dr. Current.\nI'm traveling with Dr. Base and Ann.\nFor dinner on day 3, I want a rating within 0.3 of Dr. Base's day 1 lunch. On day 4, I want breakfast with Ann."
    result = compile(query, ["Dr. Base", "Ann"], "Dr. Current")
    assert result["coverage_complete"]
    assert subjects(result) == {(3, "dinner"), (4, "breakfast")}
    for row in result["scopes"]:
        span = row["evidence_span"]
        assert query[span["start"]:span["end"]] == row["evidence_quote"]
        for cue in row["scope_cues"]:
            assert query[cue["start"]:cue["end"]] == cue["quote"]
    assert result["source_refs"] == ["event:g:4"]
    assert all(row["source_kind"] == "REQUEST_SCOPE" for row in result["scopes"])


def test_repeated_same_scope_and_repeated_target_names_are_safe():
    result = compile("For day1 dinner, I want dinner on day1 priced below Benjamin's day2 lunch and rated above Benjamin's day4 breakfast.")
    assert result["coverage_complete"]
    assert subjects(result) == {(1, "dinner")}


@pytest.mark.parametrize("query", [
    "For dinner on day1 and breakfast on day2, I want cheap restaurants.",
    "For dinner on day1, I want a restaurant like Benjamin's day2 lunch and I want accommodation on day3.",
    "For dinner on day1, I want a restaurant like Benjamin's day2 lunch and accommodation on day3.",
    "For day4 dinner, I'd like to join Benjamin for breakfast on his first day.",
    "For dinner on day1, I want a restaurant and my day2 accommodation to be cheaper.",
    "For dinner on day1, I want a restaurant; every other breakfast must be vegetarian.",
    "For dinner on day1, I want a cheap place and book a flight tomorrow.",
    "For dinner on day1, I want a cheap place and every breakfast to be vegan.",
    "For dinner on the same day, I want a cheaper place.",
    "On day2, I want it to be cheaper.",
    "I want a cheaper restaurant.",
    "For Benjamin's first-day dinner, I want a cheaper place.",
    "For day1 dinner, we want a cheaper place.",
    "I am Current.\nI'm traveling with Future.\nFor dinner on day1, I want a cheap restaurant.",
    "For day1 dinner, I want to join Future.",
    "For day1 dinner, I want a place cheaper than Future's day2 lunch.",
    "I am Current.\nI'm joining Base for this trip.\nFor dinner on day1, I want a cheap place.",
    "On day2, I want to leave accommodation and take lunch with Benjamin.",
])
def test_ambiguous_anaphoric_unknown_or_unscoped_wording_disables_completeness(query):
    assert not compile(query)["coverage_complete"]


def test_metadata_must_be_entire_exact_line_with_known_companions():
    result = compile("I am Current.\nI'm going on this trip with Base, Benjamin.\nFor dinner on day1, I want a cheap place.")
    assert result["coverage_complete"]
    assert len(result["ignored_metadata"]) == 2
    bad = compile("I am Current.\nI'm traveling with Base and Benjamin and need cheaper meals.\nFor dinner on day1, I want a cheap place.")
    assert not bad["coverage_complete"]


def test_scoped_hints_survive_an_unscoped_requirement_without_claiming_completion():
    result = compile("For dinner on day1, I want a cheaper place.\nI need every other meal to be vegetarian.")
    assert not result["coverage_complete"]
    assert subjects(result) == {(1, "dinner")}
    assert result["scopes"][0]["independently_scoped"]


def test_future_and_other_group_descriptors_cannot_be_used_as_visible_people():
    raw = request("For dinner on day1, I want to join Future.")
    result = compile_scope_coverage(raw, [{"group_id": "g", "traveler": "Future", "created_at": 5}], ["event:g:4"])
    assert not result["coverage_complete"]
    assert any(row["code"] == "FUTURE_PRIOR_TRAVELER" for row in result["diagnostics"])
    with pytest.raises(ValueError, match="current group"):
        compile_scope_coverage(raw, [{"group_id": "other", "traveler": "Future", "created_at": 0}], ["event:g:4"])


def test_inputs_are_unchanged_and_privileged_data_is_rejected():
    raw = request("For day1 dinner, I want to join Base.")
    prior = [{"group_id": "g", "traveler": "Base", "created_at": 0}]
    before = copy.deepcopy((raw, prior))
    result = compile_scope_coverage(raw, prior, ["event:g:4"])
    assert result["coverage_complete"]
    assert (raw, prior) == before
    with pytest.raises(ValueError, match="Privileged"):
        compile_scope_coverage({**raw, "gold_plan": []}, prior, ["event:g:4"])


def test_no_requirements_or_model_coverage_flag_is_not_scope_evidence():
    raw = request("I am Current.\nI'm traveling with Base.", interpretation_complete=True)
    result = compile_scope_coverage(raw, ["Base"], ["event:g:4"])
    assert not result["coverage_complete"]
    assert result["scopes"] == []


@pytest.mark.parametrize("query", [
    "For breakfast on day1, I want a place near Benjamin's day2 hotel, and dinner at Pita Pit.",
    "For first-day accommodation, I'd like to stay with Benjamin, and also dinner with him.",
    "For first-day accommodation, I'd like to stay with Benjamin, with dinner at Pita Pit.",
    "For dinner on day1, I want a place like Benjamin's day2 lunch, plus supper at Burger Queen.",
    "For breakfast on day1, I want a place near Benjamin's day2 lunch, and a private room with him.",
    "For breakfast on day1, I want a place near Benjamin's day2 lunch, plus brunch the following morning.",
    "For lunch on day1, I want a place like Benjamin's day2 breakfast as well as dinner at Pita Pit.",
])
def test_bare_coordinated_scope_and_unrecognized_aliases_are_not_frozen_as_unchanged(query):
    result = compile(query)
    assert not result["coverage_complete"]
    assert result["diagnostics"]
