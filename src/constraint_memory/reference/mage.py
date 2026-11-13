"""Pure paper-reference kernel for MAGE Algorithm 1, with declared implementation choices.

This is not the authors' implementation, a runnable experimental comparator, or
an external-environment rollback mechanism. It makes no model or network calls.
Callers supply deterministic summarizer/validator callbacks and own action,
read-schedule, token-budget, environment, and submission-ledger policies.

Unresolved paper choices and explicit reference choices:
* Roots: immutable raw/summary sentinels; root is not a revision target. A public
  anchor is separate from nonroot compressed state.
* IDs: unique layer-prefixed node IDs, with a separate boundary_step/raw ID for
  the paper's exposed compression boundary. This avoids conflating identity
  with the boundary ID reused by Algorithm 1's summary constructor.
* Trace: the last already compressed raw boundary is excluded from the next
  covered segment. Empty compression is rejected.
* Equality: canonical JSON equality, only among children of the same parent.
  No global semantic merging or timestamps enter content equality.
* Compression trigger/prompt/threshold: none invented. Compress is explicit;
  when content is omitted, an injected summarizer receives instruction+trace.
* Validation prompt/criterion: caller supplied. Maintain returns a strict
  boolean verdict plus feedback and does not automatically revise.
* Hint Update: stable union of immutable child snapshots; Revise replaces the
  hint set with restored-boundary children and feedback. H is exploratory,
  never authoritative active state.
* Same-cover overwrite: node identity persists, version increments, old content
  remains in history; a failure note remains until successful revalidation.
* Rollback: only pb/pt/recent state changes. Nodes and history are never deleted;
  external submissions, bookings, and task visibility are untouched.

Primary source: https://arxiv.org/html/2606.06090v1, sections 3.2–3.3 and
Algorithm 1 (PDF pages 5–7). Integration must declare the missing choices and
provide matched action-level reads/controller privileges before comparison.
"""
from __future__ import annotations

import json

_UNSET = object()


def _clone(value):
    """Canonical, finite JSON data keeps snapshots and equality deterministic."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Reference kernel content must be finite JSON data") from exc


class MageReferenceKernel:
    def __init__(self, *, summarizer=None, validator=None, public_anchor=None):
        self.summarizer = summarizer
        self.validator = validator
        self._raw = {"r0": {"id": "r0", "step": 0, "parent": None, "children": [],
                            "content": {"action": None, "observation": _clone(public_anchor)}}}
        self._summaries = {"s0": {"id": "s0", "parent": None, "children": [], "content": _clone(public_anchor),
                                 "cover_nodes": ["r0"], "boundary_raw_id": "r0", "boundary_step": 0,
                                 "version": 0, "validation": "ROOT", "note": None}}
        self.pb = "r0"
        self.pt = "s0"
        self._next_raw_step = 1
        self._next_summary_id = 1
        self._hints = []
        self._history = []

    def _record(self, operation, **values):
        self._history.append({"operation_id": len(self._history) + 1, "operation": operation, **_clone(values)})

    def _summary_path(self):
        path = []
        current = self.pt
        while current is not None:
            path.append(current)
            current = self._summaries[current]["parent"]
        return list(reversed(path))

    def _boundary(self):
        return self._summaries[self.pt]["cover_nodes"][-1]

    def _trace(self, boundary, current):
        reverse_path = []
        while current != boundary:
            if current is None:
                raise ValueError("Current raw pointer is not descended from the declared boundary")
            reverse_path.append(current)
            current = self._raw[current]["parent"]
        return list(reversed(reverse_path))

    def _raw_trace(self, node_ids):
        return [{"id": node_id, "step": self._raw[node_id]["step"], **_clone(self._raw[node_id]["content"])}
                for node_id in node_ids]

    def _child_hints(self, *, raw_pointer=None, summary_pointer=None):
        hints = []
        if raw_pointer is not None:
            for identifier in self._raw[raw_pointer]["children"]:
                hints.append({"kind": "raw_continuation", "node_id": identifier, "active": False,
                              "content": _clone(self._raw[identifier]["content"])})
        if summary_pointer is not None:
            for identifier in self._summaries[summary_pointer]["children"]:
                node = self._summaries[identifier]
                hints.append({"kind": "summary_continuation", "node_id": identifier, "version": node["version"],
                              "active": False, "content": _clone(node["content"]), "validation": node["validation"],
                              "note": _clone(node["note"]), "cover_nodes": list(node["cover_nodes"])})
        return hints

    def _update_hints(self, values):
        known = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in self._hints}
        for value in values:
            encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if encoded not in known:
                self._hints.append(_clone(value))
                known.add(encoded)

    def grow(self, action, observation):
        """Advance to an identical sibling, or append one immutable raw node."""
        content = _clone({"action": action, "observation": observation})
        parent = self.pb
        identifier = next((child for child in self._raw[parent]["children"]
                           if self._raw[child]["content"] == content), None)
        reused = identifier is not None
        if identifier is None:
            step = self._next_raw_step
            self._next_raw_step += 1
            identifier = f"r{step}"
            self._raw[identifier] = {"id": identifier, "step": step, "parent": parent, "children": [], "content": content}
            self._raw[parent]["children"].append(identifier)
        self.pb = identifier
        self._update_hints(self._child_hints(raw_pointer=self.pb))
        self._record("Grow", raw_id=identifier, parent=parent, reused=reused, content=content)
        return _clone(self._raw[identifier])

    def compress(self, summary=_UNSET, *, task_instruction=None):
        """Cover the current uncompressed segment; reuse and overwrite same cover."""
        boundary = self._boundary()
        covered = self._trace(boundary, self.pb)
        if not covered:
            raise ValueError("Compress requires a nonempty uncompressed raw segment")
        if summary is _UNSET:
            if self.summarizer is None:
                raise ValueError("An explicit summary or injected summarizer is required")
            summary = self.summarizer(_clone(task_instruction), self._raw_trace(covered))
        content = _clone(summary)
        parent = self.pt
        identifier = next((child for child in self._summaries[parent]["children"]
                           if self._summaries[child]["cover_nodes"] == covered), None)
        reused = identifier is not None
        previous = None
        if identifier is None:
            identifier = f"s{self._next_summary_id}"
            self._next_summary_id += 1
            self._summaries[identifier] = {"id": identifier, "parent": parent, "children": [], "content": content,
                                          "cover_nodes": list(covered), "boundary_raw_id": boundary,
                                          "boundary_step": self._raw[boundary]["step"], "version": 1,
                                          "validation": "UNVALIDATED", "note": None}
            self._summaries[parent]["children"].append(identifier)
        else:
            previous = _clone(self._summaries[identifier])
            self._summaries[identifier]["content"] = content
            self._summaries[identifier]["version"] += 1
            self._summaries[identifier]["validation"] = "UNVALIDATED"
        self.pt = identifier
        self._update_hints(self._child_hints(summary_pointer=self.pt))
        self._record("Compress", summary_id=identifier, parent=parent, reused=reused,
                     cover_nodes=covered, content=content, previous=previous)
        return _clone(self._summaries[identifier])

    def maintain(self, task_instruction, *, validator=None):
        """Validate the just-compressed trace; return failure feedback/target."""
        if self.pt == "s0":
            raise ValueError("Maintain requires a nonroot compressed segment")
        callback = validator or self.validator
        if callback is None:
            raise ValueError("An injected validator is required")
        node = self._summaries[self.pt]
        verdict = callback(_clone(task_instruction), self._raw_trace(node["cover_nodes"]), _clone(node["content"]))
        if not isinstance(verdict, tuple) or len(verdict) != 2 or not isinstance(verdict[0], bool):
            raise ValueError("Validator must return a (bool, JSON feedback) tuple")
        passed, feedback = verdict[0], _clone(verdict[1])
        node["validation"] = "VALID" if passed else "FAILED"
        node["note"] = None if passed else feedback
        result = {"status": "PASS" if passed else "FAIL", "feedback": feedback,
                  "summary_id": self.pt, "boundary_raw_id": node["boundary_raw_id"],
                  "boundary_step": node["boundary_step"]}
        self._record("Maintain", **result, summary_version=node["version"])
        return _clone(result)

    def revise(self, feedback, target_summary_id):
        """Restore BEFORE an active target summary, preserving every old node."""
        path = self._summary_path()
        if target_summary_id == "s0" or target_summary_id not in path:
            raise ValueError("Revision target must be a nonroot summary on the active path")
        feedback = _clone(feedback)
        target_position = path.index(target_summary_id)
        excluded = path[target_position:]
        restored_summary = self._summaries[target_summary_id]["parent"]
        restored_raw = self._summaries[restored_summary]["cover_nodes"][-1]
        before = {"pb": self.pb, "pt": self.pt}
        self.pt, self.pb = restored_summary, restored_raw
        # Algorithm 1 resets hints to boundary children + feedback. Copy these
        # entries so later same-cover overwrites cannot rewrite the failure hint.
        self._hints = []
        self._update_hints(self._child_hints(raw_pointer=self.pb, summary_pointer=self.pt))
        self._update_hints([{"kind": "diagnostic", "active": False, "target_summary_id": target_summary_id,
                             "excluded_summary_ids": excluded, "feedback": feedback}])
        result = {"target_summary_id": target_summary_id, "excluded_summary_ids": excluded,
                  "pb": self.pb, "pt": self.pt, "feedback": feedback}
        self._record("Revise", before=before, **result)
        return _clone(result)

    def state(self):
        """Serializable active C/R plus explicitly nonactive exploratory H."""
        compressed = []
        for identifier in self._summary_path()[1:]:
            node = self._summaries[identifier]
            compressed.append({"id": identifier, "version": node["version"], "boundary_step": node["boundary_step"],
                               "boundary_raw_id": node["boundary_raw_id"], "content": _clone(node["content"]),
                               "validation": node["validation"], "note": _clone(node["note"])})
        return {"anchor": _clone(self._summaries["s0"]["content"]), "C": compressed,
                "R": self._raw_trace(self._trace(self._boundary(), self.pb)), "H": _clone(self._hints)}

    def snapshot(self):
        return _clone({"implementation": "MAGE paper-reference kernel; not an authentic implementation",
                       "pb": self.pb, "pt": self.pt, "next_raw_step": self._next_raw_step,
                       "next_summary_id": self._next_summary_id, "raw_nodes": self._raw,
                       "summary_nodes": self._summaries, "state": self.state(), "history": self._history})
