import pytest

from .live_memory import ConceptMemory, Evidence, quoted_cents


def row(quote, status="confirmed"):
    return {"quote": quote, "concepts": ["Trip 1", "Emma"], "status": status,
            "table": "Trip 1", "row": "limit", "purchase_price_quote": None,
            "bundle_budget_quote": None, "purchase_slot": None}


def test_model_write_repair_is_journaled_as_one_replacement(tmp_path):
    requests = []

    def complete(operation, system, data, schema):
        requests.append(operation)
        if operation == "read":
            return {"concepts": ["Trip 1"], "table": "Trip 1", "rows": ["limit"], "next_purchase_slot": None}
        if data["event"]["id"] == "initial":
            return {"memories": [row("Emma confirmed $80.")]}
        if operation == "write":
            return {"memories": [row("Emma confirmed $100"), row("replacing $80.", "withdrawn")]}
        assert "Conflicting confirmations" in data["validation_error"]
        return {"memories": [row("Emma confirmed $100, replacing $80.")]}

    path = tmp_path / "journal.jsonl"
    memory = ConceptMemory(path, complete, len)
    initial = Evidence("initial", "user", "Emma confirmed $80.")
    replacement = Evidence("replacement", "user", "Emma confirmed $100, replacing $80.")
    memory.observe(initial)
    memory.observe(replacement)
    reloaded = ConceptMemory(path, complete, len)
    reloaded.observe(replacement)
    assert requests == ["write", "write", "write-repair"]
    assert len(path.read_text().splitlines()) == 2
    before = reloaded.render(*reloaded.route("limit", through=1), mode="hybrid", token_limit=1000)
    after = reloaded.render(*reloaded.route("limit"), mode="hybrid", token_limit=1000)
    assert before["delivered_ids"] == ["initial:0"]
    assert after["delivered_ids"] == ["replacement:0"]


def test_unconfirmed_action_and_invented_evidence_do_not_enter_memory(tmp_path):
    def complete(*_):
        return {"memories": [row("Purchase confirmed")]}

    memory = ConceptMemory(tmp_path / "invention.jsonl", complete, len)
    with pytest.raises(ValueError, match="Unsupported quote"):
        memory.observe(Evidence("attempt", "action", "Buy the item"))
    assert not memory.path.exists()
    assert memory.pending_path.exists()
    memory = ConceptMemory(tmp_path / "action.jsonl", complete, len)
    with pytest.raises(ValueError, match="action request cannot establish"):
        memory.observe(Evidence("attempt", "action", "Purchase confirmed"))
    assert not memory.path.exists()


def test_pending_evidence_and_scoped_aliases_survive_restart(tmp_path):
    path = tmp_path / "journal.jsonl"
    event = Evidence("initial", "user", "Em, short for Emma, confirmed $80.")

    def unavailable(*_):
        raise RuntimeError("provider unavailable")

    memory = ConceptMemory(path, unavailable, len)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        memory.observe(event)

    def complete(*_):
        return {"memories": [row(event.text)],
                "aliases": [{"concept": "Emma", "alias": "Em", "quote": event.text}]}

    reloaded = ConceptMemory(path, complete, len)
    with pytest.raises(ValueError, match="Resume unresolved evidence initial"):
        reloaded.observe(Evidence("later", "user", "Later update"))
    reloaded.observe(event)
    replay = ConceptMemory(path, unavailable, len)
    assert replay.catalog(replay.snapshot())["aliases"] == {"Emma": ["Em"]}
    assert replay.catalog(replay.snapshot(through=0))["aliases"] == {}
    assert not replay.pending_path.exists()


@pytest.mark.parametrize("partial", ["$14", "$1", "$14.9"])
def test_currency_parser_rejects_partial_amounts(partial):
    assert quoted_cents("$14.99", "Price: $14.99.") == 1499
    with pytest.raises(ValueError):
        quoted_cents(partial, "Price: $14.99.")
