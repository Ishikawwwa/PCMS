"""Small local dotenv reader; credentials never enter run configurations."""
import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    source = Path(path)
    if not source.exists():
        return
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"POLZA_API_KEY", "POLZA_MODEL", "POLZA_BASE_URL"}:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def redact(value, _path=()):
    """Redact secrets recursively before writing any metadata or errors."""
    import re
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            name = str(key)
            # A source-file hash is provenance, even for secrets.py. Restrict
            # the exception to a SHA map's Python paths and valid SHA-256 data;
            # credential-like values still pass through normal redaction.
            source_hash = (_path[-1:] == ("source_sha256",)
                and name.replace("\\", "/").startswith("src/")
                and name.endswith(".py") and isinstance(item, str)
                and re.fullmatch(r"[0-9a-f]{64}", item) is not None)
            sensitive = any(part in name.lower() for part in ("api_key", "authorization", "secret"))
            output[key] = "[REDACTED]" if sensitive and not source_hash else redact(item, (*_path, name))
        return output
    if isinstance(value, list):
        return [redact(v, (*_path, "*")) for v in value]
    if isinstance(value, str):
        value = re.sub(r"pza_[A-Za-z0-9_-]+", "[REDACTED]", value)
        secret = os.environ.get("POLZA_API_KEY")
        return value.replace(secret, "[REDACTED]") if secret else value
    return value
