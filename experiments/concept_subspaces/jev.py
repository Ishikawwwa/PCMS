"""TypeSafe System One relevance judgments for one concept at a time."""

import re

from constraint_memory.polza import ProviderError


JEV_MODEL = "typesafe/jev"


class JevRelevance:
    def __init__(self, client, trace):
        self.client = client
        self.trace = trace

    def __call__(self, query, concept, records):
        scores = {}
        for start in range(0, len(records), 12):
            ids = list(records)[start:start + 12]
            questions = {key: {"type": "noul", "instructions":
                         f"Does evidence record {key} directly supply a fact needed to answer the query "
                         "about this concept? Match the requested qualifiers. A mention alone is insufficient."}
                         for key in ids}
            payload = {"model": JEV_MODEL,
                       "state": {"query": query, "concept": concept,
                                 "evidence": {key: records[key] for key in ids}},
                       "questions": questions}
            self.trace.emit("retrieval", "Jev relevance request", payload=payload)
            self.trace.before("Judge concept memories with Jev", category="retrieval", concept=concept)
            try:
                result = self.client.system_one(payload)
            except ProviderError as error:
                self.trace.emit("error", "Jev relevance request failed", error=str(error))
                raise
            self.trace.emit("retrieval", "Jev relevance response", concept=concept, result=result)
            model = result["model"]
            if model != JEV_MODEL and not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
                raise ValueError(f"Polza returned an unexpected Jev model: {model}")
            if set(result["answers"]) != set(ids):
                raise ValueError("Jev returned an unexpected model or evidence IDs")
            for key, answer in result["answers"].items():
                score = answer.get("noul")
                if answer["type"] != "noul" or type(score) not in (int, float) or not 0 <= score <= 1:
                    raise ValueError("Jev returned an invalid relevance probability")
                scores[key] = score
        return scores
