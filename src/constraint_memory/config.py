from __future__ import annotations
import hashlib
import json
from pathlib import Path
import yaml

VARIANTS = {"B0", "B1", "B2", "E0", "E1", "E2", "E3", "E3F", "E3T", "E3I", "E3R", "E4", "E5", "E6"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path,overlay=None):
    result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if overlay:
        additions=yaml.safe_load(Path(overlay).read_text(encoding="utf-8"))
        def merge(target,source):
            for key,value in source.items():
                if isinstance(value,dict) and isinstance(target.get(key),dict):
                    merge(target[key],value)
                else:
                    target[key]=value
        merge(result,additions)
        result["experiment"]["variant_overlay"]=str(overlay)
    validate_config(result)
    return result


def verify_tokenizer(root):
    root=Path(root)
    manifest=json.loads((root/"data/manifests/qwen_tokenizer.json").read_text(encoding="utf-8"))
    path=root/manifest["tokenizer_path"]
    if manifest.get("model")!="Qwen/Qwen3.6-27B" or manifest.get("revision")!="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9":
        raise ValueError("Unexpected model tokenizer revision")
    if sha256(path)!=manifest["sha256"] or manifest["sha256"]!="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42":
        raise ValueError("Pinned Qwen tokenizer checksum mismatch")
    return path,manifest


def validate_config(config):
    for section in ("experiment", "agent", "memory", "evaluation", "safety"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    if "SET_" in json.dumps(config):
        raise ValueError("Unresolved configuration pin")
    if config["agent"]["model"] != "qwen/qwen3.6-27b":
        raise ValueError("The matched research target requires exact qwen/qwen3.6-27b")
    if config["memory"]["backend"] not in VARIANTS:
        raise ValueError("Unknown or unreproduced memory backend (authentic MAGE unavailable)")
    inheritance = config["experiment"].get("declared_inheritance_contract", False)
    if not isinstance(inheritance, bool):
        raise ValueError("Declared inheritance must be an explicit shared protocol boolean")
    if not isinstance(config["experiment"].get("tool_city_state_annotation_normalization",False),bool):
        raise ValueError("Tool city annotation adapter must be an explicit shared protocol boolean")
    if config["experiment"].get("tool_price_unit") not in {None,"USD"}:
        raise ValueError("Unknown declared travel price unit")
    if config["memory"]["backend"] in {"E3I","E3R"} and not inheritance:
        raise ValueError(config["memory"]["backend"]+" requires the declared inheritance contract shared by all conditions")
    if config["memory"]["backend"]=="E3R" and config["experiment"].get("tool_city_state_annotation_normalization") is not True:
        raise ValueError("E3R requires the declared city tool adapter shared by all conditions")
    if config["memory"]["backend"]=="E3R" and config["experiment"].get("tool_price_unit")!="USD":
        raise ValueError("E3R requires the declared price schema shared by all conditions")
    if config["memory"]["read_schedule"] != "session_start":
        raise ValueError("Per-step memory is a separate experiment and is not implemented in this protocol")
    if config["safety"].get("judgement_mode") != "none":
        raise ValueError("Gold-derived feedback cannot enter the research runtime")
    for field in ("hide_future_questions", "hide_gold_answers", "hide_oracle_annotations", "no_cross_group_memory"):
        if config["safety"].get(field) is not True:
            raise ValueError(f"Required causal isolation disabled: {field}")
    if config["memory"].get("group_isolated") is not True:
        raise ValueError("Each group must have fresh memory")
    for field in ("dataset_revision", "upstream_repo_sha"):
        pin = config["experiment"].get(field, "")
        if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError(f"{field} must be a pinned commit SHA")
    if not set(config["evaluation"]["metrics"]).issubset({"SR", "PS", "SPS"}):
        raise ValueError("This upstream scorer exposes SPS; paper sPS is not assumed equivalent")
    for field in ("max_steps_per_traveler", "max_completion_tokens", "max_agent_visible_memory_tokens", "max_total_tokens_per_group"):
        if not isinstance(config["agent"].get(field), int) or config["agent"][field] <= 0:
            raise ValueError(f"Invalid positive limit: {field}")
    if "max_extraction_tokens" in config["agent"] and (not isinstance(config["agent"]["max_extraction_tokens"],int) or config["agent"]["max_extraction_tokens"]<=0):
        raise ValueError("Invalid positive limit: max_extraction_tokens")
    if config["experiment"].get("partition") == "holdout" and not config["experiment"].get("final_holdout_frozen"):
        raise ValueError("Freeze methodology in config before final holdout")
    if config["experiment"].get("stop_on_extraction_failure") and config["experiment"].get("partition") not in {"smoke","dev"}:
        raise ValueError("The writer readiness gate is development-only; evaluation must retain model failures")
    if config["memory"].get("parser") not in {"llm", "rules"}:
        raise ValueError("Parser must be llm or rules; rules are only a limited diagnostic")
    if any(k in json.dumps(config).lower() for k in ("pza_", '"api_key"')):
        raise ValueError("Credentials belong in .env, not run configuration")
