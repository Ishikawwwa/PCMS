"""Shared, opt-in public task semantics for the inheritance experiment."""
from constraint_memory.memory.inheritance import declare_inheritance_contract


INHERITANCE_CONTRACT_ID = "public-base-inheritance-v1"
INHERITANCE_CONTRACT_TEXT = (
    "DECLARED TASK CONTRACT (shared by every compared condition): For each added traveler, "
    "preserve the fixed public base itinerary in every day and slot unless that traveler's "
    "raw request explicitly asks to change that slot. A requested change exempts only its "
    "current-subject day and slot. A reference to another traveler's day or slot does not "
    "broaden that exception. Preserve cities, transportation, attractions, and other meals "
    "and accommodations unless requested. An exception permits a different choice; it does "
    "not require one if the base choice already satisfies the request."
)


def inheritance_contract(group_id, event):
    return declare_inheritance_contract(INHERITANCE_CONTRACT_ID, str(group_id),
        INHERITANCE_CONTRACT_TEXT, ["event:" + event["event_id"]], enabled=True,
        created_at=event["step"])
