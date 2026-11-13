"""Import only audited upstream modules, avoiding package initializer side effects."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

UPSTREAM_SHA = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
TRAVEL_PACKAGE = "env.env_systems.travel_planner_env"


def mount_upstream(root: Path) -> None:
    root = root.resolve()
    for name, relative in [("agent", "agent"), ("env", "env"), ("env.env_systems", "env/env_systems"),
                           (TRAVEL_PACKAGE, "env/env_systems/travel_planner_env"),
                           (TRAVEL_PACKAGE + ".clients", "env/env_systems/travel_planner_env/clients")]:
        if name in sys.modules:
            module_path = list(getattr(sys.modules[name], "__path__", []))
            if str(root / relative) not in module_path:
                raise RuntimeError(f"Conflicting upstream import {name}; use a fresh worker process")
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(root / relative)]
        module.__package__ = name
        sys.modules[name] = module


def import_official(root: Path, suffix: str):
    mount_upstream(root)
    return importlib.import_module(TRAVEL_PACKAGE + "." + suffix)
