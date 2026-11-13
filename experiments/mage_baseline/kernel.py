"""Pinned experimental extension of the local MAGE paper-reference kernel.

This is a MAGE-paper-inspired kernel, not an authentic author implementation.
All tree transitions other than Grow identity are inherited from the unchanged
reference source.  The local identity policy compares finite canonical JSON
encodings of the actual action and actual observation, preserving bool/int and
int/float distinctions recursively.  Object-key order is irrelevant; array
order, string content, numeric representation (including signed float zero),
and value types are significant.  Equality applies only to children of the
current raw parent.  This policy is a declared local choice, not a recovered
author detail.

The controller must execute a tool before Grow even when an identical sibling
exists, and must independently checkpoint the actual resulting environment.
Node reuse neither performs execution nor refunds actions or model charges.
No network, model, dataset, scorer, or environment operation occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from constraint_memory.reference import mage as reference_mage


BASE_KERNEL_SHA256 = "a73f62452f4597831c9ec060b393013fbd78ed57c9d7704ff078b11952fdb55c"
IMPLEMENTATION_LABEL = "MAGE-paper-inspired experimental kernel; not an authentic implementation"
EQUALITY_POLICY = "finite-canonical-json-encoding; typed-values; same-raw-parent-only; v1"


def verify_base_kernel() -> str:
    """Fail closed when inherited transition code differs from the audited pin."""
    source = Path(reference_mage.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != BASE_KERNEL_SHA256:
        raise RuntimeError(
            "The MAGE experimental kernel requires its pinned reference source; "
            f"expected {BASE_KERNEL_SHA256}, found {digest}"
        )
    return digest


def _identity(content) -> str:
    # Content has already passed the reference kernel's canonical finite-JSON
    # clone.  Comparing JSON strings avoids Python's True == 1 == 1.0 equality.
    return json.dumps(content, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


class MageKernel(reference_mage.MageReferenceKernel):
    """Reference transitions with isolated, type-preserving raw sibling identity."""

    def __init__(self, *, summarizer=None, validator=None, public_anchor=None):
        verify_base_kernel()
        super().__init__(summarizer=summarizer, validator=validator,
                         public_anchor=public_anchor)

    def grow(self, action, observation):
        """Record actual action/result; reuse only an exactly identical sibling."""
        content = reference_mage._clone({"action": action, "observation": observation})
        identity = _identity(content)
        parent = self.pb
        identifier = next((child for child in self._raw[parent]["children"]
                           if _identity(self._raw[child]["content"]) == identity), None)
        reused = identifier is not None
        if identifier is None:
            step = self._next_raw_step
            self._next_raw_step += 1
            identifier = f"r{step}"
            self._raw[identifier] = {
                "id": identifier, "step": step, "parent": parent,
                "children": [], "content": content,
            }
            self._raw[parent]["children"].append(identifier)
        self.pb = identifier
        self._update_hints(self._child_hints(raw_pointer=self.pb))
        self._record("Grow", raw_id=identifier, parent=parent,
                     reused=reused, content=content)
        return reference_mage._clone(self._raw[identifier])

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot.update(implementation=IMPLEMENTATION_LABEL,
                        base_kernel_sha256=BASE_KERNEL_SHA256,
                        sibling_identity_policy=EQUALITY_POLICY)
        return snapshot
