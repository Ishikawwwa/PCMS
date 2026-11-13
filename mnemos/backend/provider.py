"""Backend-only credentials and the repository's audited Polza transport."""
import os
import math
from pathlib import Path

from src.constraint_memory.polza import PolzaClient
from src.constraint_memory.secrets import redact


def api_key():
    value = os.environ.get("MNEMOS_POLZA_API_KEY")
    path = Path(__file__).parents[1] / "state" / "polza.env"
    if not value and path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, _, item = line.partition("=")
            if name.strip() == "POLZA_API_KEY":
                value = item.strip()
    value = value or os.environ.get("POLZA_API_KEY")
    if not value:
        raise ValueError("Polza credential is not configured on the backend")
    return value


def client(settings):
    return PolzaClient(model=settings.llm_model, base_url=settings.llm_url,
                       api_key=api_key(), temperature=settings.temperature,
                       max_tokens=settings.max_tokens, seed=settings.seed,
                       max_total_tokens=2_000_000_000, max_cost_rub=None,
                       timeout=180, retries=0, stream=True,
                       reasoning={"enabled": False})


def status():
    try:
        api_key()
        configured = True
    except ValueError:
        configured = False
    return {"provider": "polza", "configured": configured,
            "default_model": "qwen/qwen3.6-27b",
            "generation": "remote", "embedding": "pinned local pretrained encoder",
            "revision_note": "Provider-managed alias; returned model, provider and request identity are recorded per call."}


def safe_error(error):
    return redact(str(error))


def accounted_receipts(receipts):
    """Apply append-only accounting corrections without rewriting original calls."""
    resolutions = {r["data"]["original_receipt_id"]: r["data"] for r in receipts if r["data"].get("operation") == "billing_reconciliation"}
    result = []
    for row in receipts:
        data = row["data"]
        if data.get("operation") == "billing_reconciliation":
            continue
        if row["id"] in resolutions:
            correction = resolutions[row["id"]]
            data = {**data, "cost": correction["cost"], "usage": correction["usage"], "billing_reconciliation": correction}
        result.append({**row, "data": data})
    return result


def reconcile_failed_billing(store, run_id, *, fetch=None):
    """Read exact provider history IDs; never repeat inference or accept its output."""
    if fetch is None:
        import requests
        def fetch(request_id):
            response = requests.get("https://polza.ai/api/v1/history/generations/" + request_id,
                                    headers={"Authorization": "Bearer " + api_key()}, timeout=30)
            response.raise_for_status()
            return response.json()
    for row in accounted_receipts(store.receipts(run_id)):
        data = row["data"]
        if data.get("cost_unit") != "RUB" or data.get("cost") is not None:
            continue
        accounting = data.get("provider_accounting") or []
        request_id = accounting[-1].get("request_id") if accounting else None
        if not request_id or not request_id.startswith("gen_"):
            continue
        history = fetch(request_id)
        if history.get("id") != request_id or history.get("status") not in {"completed", "failed"}:
            raise ValueError("Provider accounting is not final for the exact failed request")
        cost = float(history["clientCost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Provider accounting returned an invalid charge")
        usage = history.get("usage") or {}
        correction = {"operation": "billing_reconciliation", "original_receipt_id": row["id"],
                      "request_id": request_id, "cost": cost, "cost_unit": "RUB",
                      "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")},
                      "provider_history": {k: history.get(k) for k in ("id", "status", "clientCost", "usage", "completedAt", "isCancelled", "attemptsCount")},
                      "policy": "Exact provider history accounting only; failed output discarded; no inference retry."}
        store.receipt(run_id, row["job_id"], correction)
        with store.connection() as db:
            store.event(db, run_id, "billing.reconciled", correction)
