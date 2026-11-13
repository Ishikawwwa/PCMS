"""Materialized binding frontier, a separate development candidate (E3F).

Current requirements consume actual submitted choices and properties established
by visible tool rows. The historical requirements that led to those choices stay
in the full archive; they are not recursively copied into every planner read.
Archive closure, invalidation, versions, and evidence checks remain unchanged.
"""
from __future__ import annotations

import copy
import re

from .ledger import LedgerMemory


_ORDINAL_DAYS = {word: index for index, word in enumerate(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"), 1)}
_DAY = re.compile(r"\b(?:day[\s-]+(\d+)|(\d+)(?:st|nd|rd|th)?[\s-]+day|(" +
    "|".join(_ORDINAL_DAYS) + r")[\s-]+day)\b", re.I)


def _literal_scope(text):
    days = {int(match[1] or match[2]) if match[1] or match[2] else _ORDINAL_DAYS[match[3].lower()]
            for match in _DAY.finditer(text)}
    slots = {slot for slot in ("breakfast", "lunch", "dinner", "attraction", "accommodation", "transportation")
             if re.search(r"\b" + slot + r"\b", text, re.I)}
    if re.search(r"\b(?:hotel|lodging|stay|room type)\b", text, re.I): slots.add("accommodation")
    if re.search(r"\bflight\b", text, re.I): slots.add("transportation")
    if re.search(r"\bcurrent[ _]city\b", text, re.I): slots.add("current_city")
    return days, slots


class FrontierMemory(LedgerMemory):
    def __init__(self, config=None):
        options = dict(config or {})
        for feature in ("typed_ledger", "dependency_closure", "protected_compression"):
            if options.get(feature, True) is not True:
                raise ValueError(f"E3F requires {feature}; use a separately named ablation")
        for feature in ("deterministic_validation", "targeted_recovery", "glossary_hops"):
            if options.get(feature, False) is not False:
                raise ValueError(f"E3F does not include {feature}; use a separately named ablation")
        super().__init__("E3", options)
        self.variant = "E3F"
        self.features.update(materialized_frontier=True, companion_list_filter=True)

    def _retrieval_dependencies(self, record):
        if "decision_id" in record:
            # _resolve has already checked commitment, causal visibility,
            # invalidation and property/entity identity. A property's exact
            # submitted entity remains required to interpret the observed fact.
            return [record["entity_ref"]] if record.get("entity_ref") else []
        return self._deps(record)

    def _requirement_lines(self, query):
        names = {record["traveler"] for history in self.decisions.values() for record in history}
        names.update(task.get("scope", {}).get("traveler", "") for task in self.tasks.values())
        alternatives = "(?:" + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True) if name) + ")"
        listing = alternatives + r"(?:(?:,\s*(?:and\s+)?|\s+and\s+)" + alternatives + r")*\.?"
        lines = []
        for line in query.splitlines():
            match = re.fullmatch(r"I['’]m (?:going on this trip|traveling) with (.+)", line.strip())
            if match and re.fullmatch(listing, match[1]):
                continue
            lines.append(line)
        # Other names retain the existing lexical fallback. Neither JSON parse
        # success nor an empty unresolved list proves semantic completeness.
        return "\n".join(lines)

    def _named_travelers(self, query):
        return super()._named_travelers(self._requirement_lines(query))

    def _seed_ids(self, request, step):
        # Reuse current typed-constraint selection without the parent policy's
        # person-wide lexical retrieval and covered-name suppression.
        seeds = super()._seed_ids({**request, "query": ""}, step)
        query = self._requirement_lines(str(request.get("query", request.get("request", ""))))
        seeds.extend(re.findall(r"(?:decision|constraint|definition):[\w:.-]+(?:@v\d+)?", query))
        anchors = set(request.get("public_anchor_travelers", []))
        # One correctly extracted relation must not suppress another relation
        # involving the same person. Literal scope is a retrieval hint only,
        # never a new constraint or a declaration of semantic completeness.
        for paragraph in re.split(r"\n\s*\n|\n", query):
            for name in super()._named_travelers(paragraph) - anchors:
                mentions = list(re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", paragraph, re.I))
                for mention in mentions:
                    if re.search(r"(?:decision|constraint|definition):$", paragraph[:mention.start()]):
                        continue
                    tail = paragraph[mention.end():]
                    possessive = re.match(r"['’]s\b", tail)
                    local = tail[possessive.end():] if possessive else paragraph
                    # Reference phrases precede the next comma or sentence end.
                    # When the phrase has no explicit day/slot, inherit the
                    # paragraph's literal scope conservatively.
                    local = re.split(r",|(?<!\d)\.(?!\d)", local, maxsplit=1)[0]
                    days, slots = _literal_scope(local)
                    paragraph_days, paragraph_slots = _literal_scope(paragraph)
                    days, slots = days or paragraph_days, slots or paragraph_slots
                    if days and slots:
                        seeds.extend(f"decision:{name}:day{day}:{slot}" for day in sorted(days) for slot in sorted(slots))
                    else:
                        # Ambiguous wording retains exact submitted entities,
                        # without bulk-loading every observed property.
                        for did in sorted(self.decisions):
                            record = self._resolve(did, step, diagnostics=False)
                            if record and record.get("traveler") == name and record.get("attribute", "value") == "value":
                                seeds.append(did)
        return list(dict.fromkeys(seeds))

    def _sections(self, records, definitions, outcomes, compact=False):
        # Exact choices represent past actions. Duplicating full outcome text
        # would spend the same cap twice without adding materialized state.
        sections = super()._sections(records, definitions, [], compact=True)
        for record in sections["KNOWN CONTEXT"]:
            record.pop("depends_on", None)
            if record.get("entity_ref"):
                record["depends_on"] = [record["entity_ref"]]
        return sections

    @staticmethod
    def _serialize(sections):
        return ("CURRENT BINDING FRONTIER: known facts are actual submitted choices and properties from observed tools. "
                "They do not prove earlier requirements were satisfied. Historical justification remains archived; "
                "the current active constraints still apply.\n\n" + LedgerMemory._serialize(sections))

    def begin_subtask(self, request):
        context = super().begin_subtask(request)
        context.update(closure_scope="materialized_frontier", historical_dependency_graph="retained_for_archive_traversal",
                       full_historical_closure_visible=False)
        self._last_context = copy.deepcopy(context)
        return context
