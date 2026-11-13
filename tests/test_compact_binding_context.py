import json
from pathlib import Path
from tokenizers import Tokenizer
from constraint_memory.memory import create_backend
from constraint_memory.memory.grounding import numeric_literal_present

ROOT=Path(__file__).resolve().parents[1]


def test_real_45_requirement_counterexample_fits_without_losing_bindings():
    fixture=json.loads((ROOT/"tests/fixtures/writer_development_request.json").read_text(encoding="utf-8"))
    memory=create_backend("E3",{"token_budget":4096}); memory.reset(fixture["group_id"])
    tokenizer=Tokenizer.from_file(str(ROOT/"external/qwen-tokenizer/tokenizer.json"))
    memory.set_token_counter(lambda text:len(tokenizer.encode(text).ids),name="pinned Qwen tokenizer")
    memory.observe({"event_id":"6:1","group_id":"6","step":1,"type":"request","content":fixture["query"]})
    context=memory.begin_subtask({"group_id":"6","traveler":fixture["traveler"],"step":1,"query":fixture["query"],
                                  "extraction":fixture["extraction"]})
    assert context["status"]=="OK" and context["token_count"]<=4096
    assert len(context["required_closure_ids"])==45 and len(context["closure_ids"])==45
    packed=[json.loads(line) for line in context["context"].splitlines() if line.startswith("{")]
    by_ref={row["record_ref"]:row for row in packed}
    for original in context["records"]:
        visible=by_ref[original["record_ref"]]
        for field in ("subject","operator","target","source_refs","status","kind"):
            assert visible[field]==original[field]


def test_sentence_periods_are_not_numeric_evidence_mismatches():
    assert numeric_literal_present(4.8,"rated between 4.4 and 4.8.")
    assert numeric_literal_present(3.2,"rated between 2.8 and 3.2.")
    assert not numeric_literal_present(4.8,"Version 4.8.1")
    assert not numeric_literal_present(4.8,"The threshold is -4.8.")
    assert not numeric_literal_present(4.8,"The threshold is 14.8.")
