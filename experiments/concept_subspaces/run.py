"""Reproduce the first retrieval ablations using the pinned Qwen tokenizer."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from constraint_memory.config import verify_tokenizer

from .memory import Memory, ReadScope, Snapshot, read_context


SHOPPING = (
    Memory("p1", 0, "paper:m0", ("bundle24", "slot1", "B00T6NA7PA"),
           "Product 1 purchased: gluten-free carrot cake mix (B00T6NA7PA), $14.99.",
           table="bundle24", row="slot1", cents=1499),
    Memory("p2", 24, "paper:m24", ("bundle24", "slot2", "B003GQP06O"),
           "Product 2 purchased: Betty Crocker ready-to-serve cream cheese frosting, 8 pack (B003GQP06O), $34.27.",
           table="bundle24", row="slot2", cents=3427),
    Memory("p3", 31, "paper:m31", ("bundle24", "slot3", "B071S91WRZ"),
           "Product 3 purchased: AmeriColor Tulip Red airbrush food color (B071S91WRZ), $16.00.",
           table="bundle24", row="slot3", cents=1600),
)

# A cross-system distractor from the paper's SimpleMem excerpt. Its position is
# assigned for this stress test. It was not observed in the published MAGE prefix.
MERMAID = Memory("mermaid", 30, "paper:sm2", ("bundle24", "B084JBT37C"),
                 "Mermaid cake cup (B084JBT37C), $6.99, was considered.", status="considered")

TRAVEL = (
    Memory("destination", 1, "brief:travel1", ("trip1", "day2", "Rockford"),
           "The Day 2 destination is Rockford.", table="trip1", row="destination"),
    Memory("access", 1, "brief:travel1", ("trip1", "Eric"),
           "Eric requires step-free transport.", table="trip1", row="access"),
    Memory("budget80", 2, "brief:travel2", ("trip1", "Emma1"),
           "Emma's transport limit is $80 per person.", table="trip1", row="budget"),
    Memory("proposal100", 3, "brief:travel3", ("trip1", "Emma1"),
           "Emma's transport limit of $100 per person was proposed.", status="proposed",
           table="trip1", row="budget"),
    Memory("budget100", 4, "brief:travel4", ("trip1", "Emma1"),
           "Emma's transport limit is $100 per person.", table="trip1", row="budget"),
    Memory("other-emma", 2, "synthetic:other-trip", ("trip2", "Emma2"),
           "The other Emma's transport limit is $150 per person.", table="trip2", row="budget"),
    Memory("hotel", 2, "synthetic:hotel", ("trip1", "day2"),
           "Day 2 hotel check-in is at 3 p.m.", table="trip1", row="hotel"),
)


def evaluate(count_tokens) -> dict:
    shopping = Snapshot(SHOPPING + (MERMAID,), through=31)
    shopping_scope = ReadScope("bundle24", ("slot3",), 10000, ("slot1", "slot2", "slot3"))
    travel_before = Snapshot(TRAVEL, through=3)
    travel_after = Snapshot(TRAVEL, through=4)
    travel_scope = ReadScope("trip1", ("destination", "access", "budget"))
    scenarios = (
        ("shopping-slot-resolved", shopping, shopping.recall(("bundle24", "slot3"), limit=2),
         shopping_scope, {"p3"}, "Remaining $34.74."),
        ("shopping-missed-predecessor", shopping, (SHOPPING[1], MERMAID),
         shopping_scope, {"p3"}, "Remaining $34.74."),
        ("shopping-no-confirmed-anchor", shopping, (MERMAID,),
         shopping_scope, {"p3"}, "Remaining $34.74."),
        ("travel-before-confirmation", travel_before, TRAVEL[:2],
         travel_scope, {"destination", "access", "budget80"}, None),
        ("travel-after-confirmation", travel_after, TRAVEL[:2],
         travel_scope, {"destination", "access", "budget100"}, None),
    )
    trials = []
    for case, snapshot, initial, scope, required, expected_total in scenarios:
        for token_limit in (48, 96, 192):
            for mode in ("subspaces", "subspaces+table", "subspaces+table-priority", "flat-table"):
                read = read_context(
                    snapshot, initial if mode != "flat-table" else (), scope,
                    mode=mode, token_limit=token_limit, count_tokens=count_tokens,
                )
                missing = required - set(read["delivered_ids"])
                total_present = None if expected_total is None else expected_total in read["context"]
                trials.append({
                    "case": case, "mode": mode, "token_limit": token_limit, **read,
                    "missing_required_ids": sorted(missing), "expected_total_present": total_present,
                    "complete_evidence": not missing and total_present is not False,
                })
    return {
        "kind": "pre-annotated offline retrieval diagnostic",
        "model_calls": 0,
        "shopping_source": "https://arxiv.org/pdf/2606.06090v1#page=15",
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    tokenizer_path, manifest = verify_tokenizer(root)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = evaluate(lambda value: len(tokenizer.encode(value, add_special_tokens=False).ids))
    result["tokenizer_sha256"] = manifest["sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
