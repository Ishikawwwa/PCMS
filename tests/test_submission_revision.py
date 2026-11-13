from constraint_memory.benchmark_adapter.submissions import SubmissionLedger,named_sections


def packet(index,name):
    return {"group_id":"g","traveler_index":index,"name":name,"query":"visible request"}


def result(index,sections,source_round=None):
    raw="\n\n".join(f"=== {name}'s Plan ===\n{text}" for name,text in sections)
    accepted=[{"name":name,"submitted_text":text,"plan":[{"day":1,"dinner":text.split('Dinner: ',1)[1]}] if 'Dinner: ' in text else None}
              for name,text in named_sections(raw).items()]
    return {"group_id":"g","traveler_index":index,"source_round":source_round or index,"final_plan":raw,
            "accepted_submissions":accepted,"total_steps":1}


def test_later_section_replaces_whole_prior_plan_and_tracks_source_round():
    ledger=SubmissionLedger("g","Base")
    ledger.register_request(packet(1,"Bob"))
    ledger.accept(result(1,[("Bob","Day 1:\nDinner: Old\nDay 2:\nDinner: Old second day")]))
    ledger.register_request(packet(2,"Carol"))
    ledger.accept(result(2,[("Bob","Day 1:\nDinner: Revised"),("Carol","Day 1:\nDinner: Carol")]))
    final=ledger.predictions("E3","family")
    assert final[0]["final_plan"]=="=== Bob's Plan ===\nDay 1:\nDinner: Revised"
    assert "Day 2" not in final[0]["final_plan"]
    assert final[0]["source_round"]==2 and final[0]["submission_version"]==2
    assert "Old second day" in ledger.history[0]["final_plan"]


def test_empty_named_replacement_retires_old_plan_but_absence_preserves_it():
    ledger=SubmissionLedger("g","Base")
    ledger.register_request(packet(1,"Bob"))
    ledger.accept(result(1,[("Bob","Day 1:\nDinner: Old")]))
    ledger.register_request(packet(2,"Carol"))
    ledger.accept(result(2,[("Carol","Day 1:\nDinner: Carol")]))
    assert ledger.predictions()[0]["plan"] is not None
    ledger.accept(result(2,[("Bob","")]))
    assert ledger.predictions()[0]["plan"] is None
    assert ledger.predictions()[0]["submission_version"]==2


def test_last_same_name_section_wins_and_base_future_names_are_never_promoted():
    ledger=SubmissionLedger("g","Base")
    ledger.register_request(packet(1,"Bob"))
    accepted=ledger.accept(result(1,[("Bob","Day 1:\nDinner: First"),("Base","Day 1:\nDinner: Illegal"),
                                    ("Future","Day 1:\nDinner: Future secret"),("Bob","Day 1:\nDinner: Last")]))
    assert ledger.predictions()[0]["plan"][0]["dinner"]=="Last"
    assert {r["reason"] for r in accepted["ignored"]}=={"FIXED_BASE_REVISION","FUTURE_OR_UNKNOWN_TRAVELER"}
    ledger.register_request(packet(2,"Future"))
    assert ledger.predictions()[1]["plan"] is None


def test_repair_target_is_past_but_charged_to_current_visible_round():
    ledger=SubmissionLedger("g","Base")
    ledger.register_request(packet(1,"Bob"))
    ledger.accept(result(1,[("Bob","Day 1:\nDinner: Old")]))
    ledger.register_request(packet(2,"Carol"))
    ledger.accept(result(1,[("Bob","Day 1:\nDinner: Fixed")],source_round=2))
    assert ledger.predictions()[0]["source_round"]==2
    assert ledger.predictions()[1]["plan"] is None
