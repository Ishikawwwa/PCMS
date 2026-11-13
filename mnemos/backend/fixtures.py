"""Labeled synthetic observations and a separately pinned actual tool trace."""
import json
from pathlib import Path

TASKS = [
    ("capacity", "A search agent sent twelve simultaneous requests. The service returned rate-limit errors. Retrying with two simultaneous requests completed the search."),
    ("capacity", "A document agent queued six downloads against a busy server. Limiting concurrent downloads to two avoided further overload errors."),
    ("capacity", "A deployment agent submitted four builds at once. The worker pool was exhausted. A bounded queue let the remaining builds finish."),
    ("recovery", "A file agent saved the original file and an operation receipt before editing. After a failed edit it restored the original and checked its checksum."),
    ("recovery", "A deployment agent captured a checkpoint before an upgrade. The health check failed. It restored the checkpoint and verified service health."),
    ("recovery", "A data agent recorded the input version before a migration. A failed migration was retried from the saved input after checking its integrity."),
    ("revision", "An assistant withdrew a recommendation when the cited source was corrected. It marked dependent claims for review and kept the previous citation."),
    ("revision", "A research agent found a contradiction in a supporting observation. It flagged all derived conclusions for review while preserving their source history."),
    ("revision", "A planning agent learned that a source date was disputed. It reconsidered dependent plans without deleting the original observation."),
]

LORE = [
    ("lore-observation", "In A New Hope, Luke disables his targeting computer and trusts the Force during the Death Star attack."),
    ("lore-observation", "In The Empire Strikes Back, Yoda trains Luke to notice assumptions that limit his use of the Force."),
    ("lore-observation", "In Return of the Jedi, Luke refuses to kill his father despite the Emperor's pressure."),
    ("lore-observation", "In A New Hope, Leia entrusts the Death Star plans to R2-D2 before capture."),
    ("lore-observation", "In Rogue One, the rebels transmit the Death Star plans despite the loss of their local team."),
    ("lore-observation", "In A New Hope, the rebels use the delivered plans to identify a weakness in the Death Star."),
]


def datasets():
    return [
        {"id": "agent-task-observations", "name": "Agent task observations", "description": "9 authored synthetic task observations; not actual execution traces.", "records": records(TASKS, "synthetic-task-observation")},
        {"id": "recorded-agent-tools", "name": "Recorded agent tool interactions", "description": "6 actual recorded MemoryArena planner tool events, with original source lines and SHA-256 provenance. Benchmark tool data; no task-success claim.", "records": json.loads((Path(__file__).parents[1] / "fixtures" / "recorded_agent_tools.json").read_text(encoding="utf-8"))},
        {"id": "star-wars", "name": "Star Wars source observations", "description": "6 paraphrased film observations, canon; not agent execution traces.", "records": records(LORE, "film-observation; canon")},
        {"id": "scale-500", "name": "500 node interaction fixture", "description": "Synthetic numbered observations for measuring rendering performance only.", "records": [{"content": f"Synthetic rendering sample {i}: {TASKS[i % len(TASKS)][1]}", "session": f"scale-{i // 10}", "provenance": "synthetic scale fixture; no scientific validity"} for i in range(500)]},
    ]


def records(items, provenance):
    return [{"content": content, "session": session, "provenance": provenance, "sequence": i} for i, (session, content) in enumerate(items)]
