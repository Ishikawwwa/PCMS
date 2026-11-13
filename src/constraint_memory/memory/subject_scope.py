"""Independent, deliberately narrow coverage of current request day/slots.

This recognizes explicit subject scope, not the meaning of relational targets,
policy predicates, or requested changes. It reads only its visible arguments;
valid model JSON is never evidence of complete request coverage. Unsupported
wording leaves useful scoped hints but disables a complete-coverage claim.
"""
from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from .backends import reject_privileged


SOURCE_KIND = "REQUEST_SCOPE"
_ORDINALS = {word: day for day, word in enumerate(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"), 1)}
_DAY = re.compile(r"\b(?:(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)[ -]+day|day[ -]*(?P<number>[1-9]\d*))\b", re.I)
_SLOT = re.compile(r"\b(?:breakfast|lunch|dinner|accommodation)\b", re.I)
_STAY = re.compile(r"\bstay\s+(?:somewhere|with|at|in)\b", re.I)
_CURRENT_ANAPHORA = re.compile(r"\b(?:(?:same|next|following|previous|this|that)\s+(?:day|meal|slot)|it|them|their|his|her)\b", re.I)
_CURRENT_INTENT = re.compile(r"\bI(?:['’](?:m|d|ll)|\s+(?:want|would|need|hope|prefer|am|will|like|seek))\b", re.I)


def _refs(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(ref, str) and bool(ref) for ref in value)


def _name_pattern(name: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")


def _visible_names(prior: list, group: str, step: int, current: str) -> tuple[list[str], list[dict]]:
    if not isinstance(prior, list):
        raise ValueError("Prior travelers must be a visible name/descriptor list")
    names, diagnostics = [], []
    for item in prior:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            reject_privileged(item)
            supplied_group = item.get("group_id", item.get("scope", {}).get("group") if isinstance(item.get("scope"), dict) else None)
            if supplied_group is None or str(supplied_group) != group:
                raise ValueError("Prior traveler descriptor must belong to the current group")
            created = item.get("created_at")
            if not isinstance(created, int) or isinstance(created, bool) or created < 0:
                raise ValueError("Prior traveler descriptor needs a nonnegative created_at")
            name = item.get("traveler", item.get("name"))
            if created > step:
                diagnostics.append({"code": "FUTURE_PRIOR_TRAVELER", "traveler": name})
                continue
        else:
            raise ValueError("Prior traveler must be a name or scoped visible descriptor")
        if not isinstance(name, str) or not name.strip() or name != name.strip() or "\n" in name or "\r" in name:
            raise ValueError("Prior traveler names must be nonempty single-line literals")
        if name != current and name not in names:
            names.append(name)
    return names, diagnostics


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _units(text: str, start: int, end: int, names: list[str]) -> list[tuple[int, int]]:
    """Split sentences while preserving decimal points and literal name periods."""
    protected = [match.span() for name in names for match in _name_pattern(name).finditer(text, start, end)]
    boundaries = [start]
    for pos in range(start, end):
        char = text[pos]
        if char not in ".?!;" or char != ";" and pos + 1 < end and not text[pos + 1].isspace():
            continue
        if any(left <= pos < right for left, right in protected):
            continue
        if char == "." and pos > start and pos + 1 < end and text[pos - 1].isdigit() and text[pos + 1].isdigit():
            continue
        boundaries.append(pos + 1)
    boundaries.append(end)
    return [span for left, right in zip(boundaries, boundaries[1:]) if (span := _trim_span(text, left, right))[0] < span[1]]


def _metadata(text: str, current: str, prior: list[str]) -> str | None:
    if re.fullmatch(r"I am " + re.escape(current) + r"\.?", text):
        return "CURRENT_IDENTITY"
    match = re.fullmatch(r"I['’]m (?:going on this trip|traveling) with (.+)", text)
    if not match or not prior:
        return None
    alternatives = "|".join(re.escape(name) for name in sorted(prior, key=len, reverse=True))
    name = "(?:" + alternatives + ")"
    if re.fullmatch(name + r"(?:(?:,\s*(?:and\s+)?|\s+and\s+)" + name + r")*\.?", match[1]):
        return "KNOWN_COMPANION_LIST"
    return None


def _evidence(text: str, start: int, end: int) -> dict:
    return {"start": start, "end": end, "quote": text[start:end]}


def compile_scope_coverage(request: dict, prior_travelers: list, source_refs: list[str]) -> dict:
    """Certify only explicit current-subject day/slot coverage in supported prose.

    ``request`` must contain group_id, traveler, step and raw query. String prior
    names are the caller's declaration of already visible travelers; descriptors
    additionally enforce group and causal step. A target's different day/slot
    never becomes a current-subject exception. All nonmetadata sentences must be
    independently covered. Unknown metadata, multiple scopes, current-subject
    anaphora and new current intents in a target suffix make coverage uncertain.
    """
    if not isinstance(request, dict):
        raise ValueError("Current request must be an object")
    reject_privileged(request)
    group, current, step, text = request.get("group_id"), request.get("traveler"), request.get("step"), request.get("query")
    if group is None or not str(group) or not isinstance(current, str) or not current.strip() or current != current.strip() or "\n" in current or "\r" in current:
        raise ValueError("Current request requires a group and literal traveler")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0 or not isinstance(text, str) or not text.strip():
        raise ValueError("Current request requires nonnegative step and nonempty raw query")
    if not _refs(source_refs):
        raise ValueError("Scope evidence requires nonempty literal source refs")
    group = str(group)
    prior, diagnostics = _visible_names(prior_travelers, group, step, current)
    patterns = [(name, _name_pattern(name)) for name in prior]
    scopes, ignored = [], []
    offset = 0
    for line in text.splitlines(keepends=True):
        left, right = _trim_span(text, offset, offset + len(line))
        offset += len(line)
        if left == right:
            continue
        metadata = _metadata(text[left:right], current, prior)
        if metadata:
            ignored.append({"kind": metadata, "evidence": _evidence(text, left, right), "source_refs": list(source_refs)})
            continue
        for start, end in _units(text, left, right, [current, *prior]):
            unit = text[start:end]
            mentions = [(match.start(), name) for name, pattern in patterns for match in pattern.finditer(unit)]
            target_start = min((pos for pos, _ in mentions), default=len(unit))
            prefix = unit[:target_start]
            day_matches, slot_matches = list(_DAY.finditer(prefix)), list(_SLOT.finditer(prefix))
            # "On day4, I hope to join Violet for dinner" names the other
            # participant before the shared meal. Accept only this complete,
            # narrow suffix; a following target day or extra clause stays
            # ambiguous instead of becoming the current person's subject.
            if target_start < len(unit) and re.search(r"\bjoin\s+$", prefix, re.I):
                for name, _ in patterns:
                    shared = re.fullmatch(re.escape(name) + r"\s+for\s+(breakfast|lunch|dinner)\.?", unit[target_start:], re.I)
                    if shared:
                        slot_matches.extend(_SLOT.finditer(unit, target_start + shared.start(1), target_start + shared.end(1)))
                        break
            stay_matches = list(_STAY.finditer(prefix))
            days = {int(match["number"]) if match["number"] else _ORDINALS[match["ordinal"].lower()] for match in day_matches}
            slots = {match[0].lower() for match in slot_matches}
            if stay_matches:
                slots.add("accommodation")
            codes = []
            if not re.match(r"^(?:For\b|On\b|I\b|I['’])", prefix, re.I) or not re.search(r"\b(?:I|my)\b", prefix, re.I):
                codes.append("CURRENT_SUBJECT_NOT_EXPLICIT")
            if len(days) != 1:
                codes.append("MISSING_OR_AMBIGUOUS_CURRENT_DAY")
            if len(slots) != 1:
                codes.append("MISSING_OR_AMBIGUOUS_CURRENT_SLOT")
            if _CURRENT_ANAPHORA.search(unit) or re.search(r"\b(?:we|our|us)\b", prefix, re.I):
                codes.append("ANAPHORIC_OR_MULTIPLE_CURRENT_SUBJECT")
            if len(list(_CURRENT_INTENT.finditer(unit))) > 1 or re.search(r"\b(?:and|also|but)\s+(?:for\s+my|my\s+(?:day|first|second|third|fourth|fifth)|I\b)", unit, re.I):
                codes.append("ADDITIONAL_CURRENT_REQUIREMENT")
            if re.search(r"\b(?:and|also|then|plus|but)\s+(?:book|take|change|replace|choose|visit|stay|eat|have|join|need|want|prefer|seek|go|travel|arrange|reserve|ensure|make)\b", unit, re.I):
                codes.append("ADDITIONAL_UNSCOPED_ACTION")
            if re.search(r"\b(?:and|also|then|plus|but|with|along\s+with|as\s+well\s+as|in\s+addition\s+to)\s+(?:(?:also|then|for|my|a|the|our|private|shared|entire|single|double|non-smoking)\s+)*(?:breakfast|lunch|dinner|supper|accommodation|hotel|lodging|room(?!\s+type)|bedroom|stay|flight|transportation|attraction)\b", unit, re.I):
                codes.append("BARE_COORDINATED_SCOPE_UNCERTAIN")
            if re.search(r"\b(?:supper|brunch|hotel|lodging|guesthouse|overnight)\b|\b(?:next|following|same|previous|that|this)\s+(?:morning|evening|night)\b", unit, re.I):
                codes.append("UNSUPPORTED_SCOPE_ALIAS")
            if re.search(r"\b(?:flight|flights|transportation|attraction|attractions|budget|throughout|tomorrow|tonight|later|afterwards)\b|\b(?:all|every|any|other)\s+(?:(?:the|my)\s+)?(?:other\s+)?(?:days?|meals?|breakfasts?|lunch(?:es)?|dinners?|accommodations?)\b", unit, re.I):
                codes.append("UNSUPPORTED_OR_UNSCOPED_CHANGE")
            # A second unnamed slot/day after a prior-person target could be a
            # second requested current change. Do not guess ownership across
            # coordination; a repeated named target starts a fresh segment.
            target_boundaries = sorted({pos for pos, _ in mentions}) + [len(unit)]
            for target_left, target_right in zip(target_boundaries, target_boundaries[1:]):
                target = unit[target_left:target_right]
                target_days = {int(match["number"]) if match["number"] else _ORDINALS[match["ordinal"].lower()] for match in _DAY.finditer(target)}
                target_slots = {match[0].lower() for match in _SLOT.finditer(target)}
                if len(target_days) > 1 or len(target_slots) > 1:
                    codes.append("AMBIGUOUS_COORDINATED_SUBJECT_SCOPE")
            # Unknown possessive people and unknown join/with people cannot be
            # silently treated as already visible relational targets.
            unknown = []
            for match in re.finditer(r"\b([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*)['’]s\b", unit):
                name = match[1]
                if name not in prior and name != current:
                    unknown.append(name)
            for match in re.finditer(r"\b(?:join|with)\s+([A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)*)", unit):
                literal = match[1]
                name = literal.rstrip(".")
                if literal not in prior and literal != current and name not in prior and name != current:
                    unknown.append(name)
            if unknown:
                codes.append("UNKNOWN_PERSON_REFERENCE")
            evidence = _evidence(text, start, end)
            if len(days) == len(slots) == 1:
                scope = {"subject": {"traveler": current, "day": next(iter(days)), "slot": next(iter(slots))},
                         "source_kind": SOURCE_KIND, "source_refs": list(source_refs),
                         "evidence_quote": unit, "evidence_span": {"start": start, "end": end},
                         "scope_cues": [_evidence(text, start + match.start(), start + match.end()) for match in [*day_matches, *slot_matches, *stay_matches]],
                         "independently_scoped": not codes}
                scopes.append(scope)
            if codes:
                diagnostics.append({"code": "REQUEST_SCOPE_UNCERTAIN", "reasons": list(dict.fromkeys(codes)),
                                    "evidence": evidence, "source_refs": list(source_refs),
                                    **({"unknown_people": sorted(set(unknown))} if unknown else {})})
    if not scopes:
        diagnostics.append({"code": "NO_INDEPENDENT_CURRENT_SCOPE"})
    complete = bool(scopes) and not diagnostics
    return {"schema_version": 1, "group_id": group, "traveler": current, "step": step,
            "source_kind": SOURCE_KIND, "source_refs": list(source_refs),
            "prior_travelers": copy.deepcopy(prior_travelers),
            "request_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "status": "COMPLETE" if complete else "UNCERTAIN", "coverage_complete": complete,
            "relational_semantics_interpreted": False, "policy_semantics_interpreted": False,
            "scopes": scopes, "ignored_metadata": ignored, "diagnostics": diagnostics}


__all__ = ["compile_scope_coverage"]
