"""Fetch only the pinned official Qwen tokenizer, never model weights."""
import hashlib
import json
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parents[1]
revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
expected="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
target=root/"external/qwen-tokenizer/tokenizer.json"
url=f"https://huggingface.co/Qwen/Qwen3.6-27B/resolve/{revision}/tokenizer.json"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if not target.exists() or digest(target)!=expected:
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(".download")
    with urllib.request.urlopen(url,timeout=60) as source, temporary.open("wb") as output:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            output.write(chunk)
    if digest(temporary)!=expected:
        raise ValueError("Official tokenizer checksum mismatch; downloaded file kept for inspection")
    temporary.replace(target)
manifest={"model":"Qwen/Qwen3.6-27B","revision":revision,"tokenizer_path":"external/qwen-tokenizer/tokenizer.json",
          "source_url":url,"sha256":expected,"bytes":target.stat().st_size,
          "scope":"tokenizer only; does not pin provider inference revision"}
path=root/"data/manifests/qwen_tokenizer.json"
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":"ready","revision":revision,"sha256":expected}))
