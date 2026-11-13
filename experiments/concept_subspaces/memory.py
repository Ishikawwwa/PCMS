"""Deterministic retrieval over source-linked, pre-annotated memories."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Memory:
    id: str
    step: int
    source: str
    concepts: tuple[str, ...]
    text: str
    status: str = "confirmed"
    table: str | None = None
    row: str | None = None
    cents: int | None = None
    budget_cents: int | None = None
    purchase_slot: int | None = None

    def __post_init__(self):
        if not self.source or not self.concepts:
            raise ValueError("A memory needs evidence and at least one concept")
        if self.status not in {"confirmed", "considered", "proposed", "withdrawn"}:
            raise ValueError(f"Unknown memory status {self.status}")
        if (self.table is None) != (self.row is None):
            raise ValueError("A table reference needs both table and row IDs")
        if self.status == "withdrawn" and self.table is None:
            raise ValueError("A withdrawal needs a table row")

    def line(self) -> str:
        return f"[{self.status}] {self.text} (source: {self.source})"


class Snapshot:
    """Current rows and concept indexes at an inclusive evidence cutoff."""

    def __init__(self, memories: tuple[Memory, ...], *, through: int):
        self.through = through
        self.rows: dict[str, dict[str, Memory]] = defaultdict(dict)
        self.subspaces: dict[str, list[Memory]] = defaultdict(list)
        visible = sorted((m for m in memories if m.step <= through), key=lambda m: m.step)
        self.by_id = {m.id: m for m in visible}
        if len(self.by_id) != len(visible):
            raise ValueError("Memory IDs must be unique")
        for memory in visible:
            if memory.status in {"confirmed", "withdrawn"} and memory.table is not None:
                prior = self.rows[memory.table].get(memory.row)
                if prior is not None and prior.step == memory.step:
                    raise ValueError("Conflicting confirmations at the same evidence step")
                self.rows[memory.table][memory.row] = memory
            for concept in dict.fromkeys(memory.concepts):
                self.subspaces[concept].append(memory)

    def available(self, memory: Memory) -> bool:
        if self.by_id.get(memory.id) != memory:
            return False
        if memory.status in {"confirmed", "withdrawn"} and memory.table is not None:
            return self.rows[memory.table][memory.row] == memory
        return True

    def rank(self, concepts: tuple[str, ...]) -> list[tuple[Memory, int]]:
        matches: dict[Memory, int] = defaultdict(int)
        for concept in dict.fromkeys(concepts):
            for memory in self.subspaces.get(concept, ()):
                if self.available(memory):
                    matches[memory] += 1
        ordered = sorted(matches, key=lambda m: (-matches[m], m.status != "confirmed", -m.step, m.id))
        return [(memory, matches[memory]) for memory in ordered]

    def recall(self, concepts: tuple[str, ...], *, limit: int) -> tuple[Memory, ...]:
        return tuple(memory for memory, _ in self.rank(concepts)[:limit])


@dataclass(frozen=True)
class ReadScope:
    table: str
    required_rows: tuple[str, ...]
    budget_cents: int | None = None
    spend_rows: tuple[str, ...] = ()
    budget_source: str | None = None


def dollars(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    whole, fraction = divmod(abs(cents), 100)
    return f"{sign}${whole}.{fraction:02d}"


def read_context(
    snapshot: Snapshot,
    initial: tuple[Memory, ...],
    scope: ReadScope,
    *,
    mode: str,
    token_limit: int,
    count_tokens: Callable[[str], int],
    decisions: list[dict] | None = None,
) -> dict:
    if mode not in {"subspaces", "subspaces+table", "subspaces+table-priority", "hybrid", "flat-table"}:
        raise ValueError(f"Unknown read mode {mode}")
    if any(not snapshot.available(memory) for memory in initial):
        raise ValueError("Initial recall contains evidence outside the active snapshot")
    lines: list[str] = []
    delivered: list[str] = []
    balance_in_context = False

    def decision(candidate: str, status: str, **details) -> None:
        if decisions is not None:
            decisions.append({"candidate": candidate, "status": status, **details})

    def append(line: str, candidate: str) -> bool:
        tokens = count_tokens("\n".join([*lines, line]))
        decision(candidate, "over_budget" if tokens > token_limit else "included", packed_tokens_if_added=tokens)
        if tokens > token_limit:
            return False
        lines.append(line)
        return True

    # Use the frozen recall set as the anchor even when the prompt budget drops
    # some initial lines. Only an active confirmation may authorize expansion.
    table = snapshot.rows.get(scope.table, {})
    anchored = any(memory.status == "confirmed" and table.get(memory.row) == memory for memory in initial)

    def pack_initial() -> None:
        ordered = tuple(dict.fromkeys(initial))
        for index, memory in enumerate(ordered):
            if memory.id in delivered:
                decision(memory.id, "already_included", source="initial_recall")
                continue
            if balance_in_context and memory.budget_cents is not None and memory.source == scope.budget_source:
                decision(memory.id, "covered_by_balance")
                continue
            if not append(memory.line(), memory.id):
                for deferred in ordered[index + 1:]:
                    decision(deferred.id, "initial_recall_prefix_exhausted")
                break
            delivered.append(memory.id)

    if mode not in {"subspaces+table-priority", "hybrid"}:
        pack_initial()

    if mode in {"flat-table", "hybrid"} or (mode != "subspaces" and anchored):
        def pack_row(row):
            memory = table.get(row)
            if memory is None:
                decision(row, "row_missing")
            elif balance_in_context and memory.budget_cents is not None and memory.source == scope.budget_source:
                decision(memory.id, "covered_by_balance")
            elif memory.id in delivered:
                decision(memory.id, "already_included", source="table")
            elif append(memory.line(), memory.id):
                delivered.append(memory.id)

        rows = tuple(dict.fromkeys(scope.required_rows))
        prioritize_balance = mode in {"flat-table", "hybrid", "subspaces+table-priority"} and scope.spend_rows
        if prioritize_balance:
            predecessor = scope.spend_rows[-1]
            pack_row(predecessor)
            rows = tuple(row for row in rows if row != predecessor)
        else:
            for row in rows:
                pack_row(row)

        # State bundle totals only when every expected prior slot has evidence.
        if scope.budget_cents is not None and scope.spend_rows:
            purchases = [table.get(row) for row in dict.fromkeys(scope.spend_rows)]
            if all(purchase is not None and purchase.status == "confirmed" and purchase.cents is not None
                   for purchase in purchases):
                spent = sum(purchase.cents for purchase in purchases)
                remaining = scope.budget_cents - spent
                source_ids = [purchase.source for purchase in purchases]
                if scope.budget_source is not None:
                    source_ids.append(scope.budget_source)
                sources = ", ".join(dict.fromkeys(source_ids))
                balance_in_context = append(f"Spent {dollars(spent)}. Remaining {dollars(remaining)}. (sources: {sources})", "bundle_balance")
            else:
                decision("bundle_balance", "incomplete_confirmed_receipts")

        if prioritize_balance:
            for row in rows:
                pack_row(row)
    elif mode != "subspaces":
        decision(scope.table, "no_confirmed_anchor")

    if mode in {"subspaces+table-priority", "hybrid"}:
        pack_initial()

    context = "\n".join(lines)
    return {
        "initial_ids": [memory.id for memory in initial],
        "delivered_ids": delivered,
        "table_anchor": anchored,
        "tokens": count_tokens(context),
        "context": context,
    }
