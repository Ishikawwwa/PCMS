"""Causal, group-local memory backends for the experimental pipeline.

All contexts contain observations and actual agent actions only.  Record IDs are
stable across revisions; evidence and historical versions are append-only.
"""

from .backends import MemoryBackend, NoMemory, FullHistoryMemory, BM25Memory
from .ledger import LedgerMemory
from .frontier import FrontierMemory
from .table_frontier import TableFrontierMemory


def create_backend(variant: str, config: dict | None = None) -> MemoryBackend:
    """Construct an isolated matrix condition (B3 requires an external adapter)."""
    name = variant.upper()
    options = dict(config or {})
    if name == "B0":
        return NoMemory(options)
    if name == "B1":
        return FullHistoryMemory(options)
    if name == "B2":
        return BM25Memory(options)
    if name == "E3F":
        return FrontierMemory(options)
    if name == "E3T":
        return TableFrontierMemory(options)
    if name == "E3I":
        from .inherited_frontier import InheritedFrontierMemory
        return InheritedFrontierMemory(options)
    if name == "E3R":
        from .verified_frontier import VerifiedFrontierMemory
        return VerifiedFrontierMemory(options)
    if name in {f"E{i}" for i in range(7)}:
        return LedgerMemory(name, options)
    if name == "B3":
        raise NotImplementedError("B3 requires an authentic, separately installed MAGE adapter")
    raise ValueError(f"Unknown memory variant: {variant}")


__all__ = ["MemoryBackend", "LedgerMemory", "FrontierMemory", "TableFrontierMemory", "create_backend"]
