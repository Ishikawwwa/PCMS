# Concept Memory Lab

From the repository root, install Python 3.12 and uv, then run the following commands in PowerShell.

```powershell
uv sync --frozen --python 3.12 --extra dev
uv pip install openai==2.41.0 pydantic
uv run --no-sync python scripts/prepare_tokenizer.py
git clone https://github.com/Komorebi660/MAGE.git external/MAGE
git -C external/MAGE checkout --detach ace693139dc43e8bdb3076da4127483b41a5060f
New-Item -ItemType Directory -Force data/hidden, tmp | Out-Null
Invoke-WebRequest https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/da1a37c8b19280e18627ca01cf368195a5e1d92e/group_travel_planner/data.jsonl -OutFile data/hidden/memoryarena_travel.jsonl
Invoke-WebRequest https://huggingface.co/datasets/osunlp/TravelPlanner/resolve/8736504/train.csv -OutFile tmp/concept-subspaces-travel-train.csv
Copy-Item .env.example .env
```

Set `POLZA_API_KEY` in `.env` before starting a live run. Live requests incur provider charges. To enable the shopping retrieval diagnostic, also download the pinned shopping input listed below.

```powershell
uv run --no-sync python -X utf8 -m experiments.concept_subspaces.lab_server --open
```

The console listens on [localhost port 8765](http://127.0.0.1:8765/). `--port` selects another local port. Keep the server terminal open while running experiments. The provider and route come from the pinned travel configuration. The existing `.env` supplies the Polza credential to the worker.

## Run and inspect

1. Choose **New run**, select the experiment and its limits, then choose **Start paused**. The paired travel option compares official MAGE and standalone PCMS. Set the development group, traveler prefix or full group, action limit per attempt, and shared actor reasoning setting.
2. In **Timeline**, open a checkpoint and follow its link to the request, validated extraction, or tool action. Filter by event kind or turn and click an event to keep its inspector fixed. **Raw JSON** and **Copy** show the complete recorded event.
3. Use **Step once** to execute a single checkpoint, **Run to** for the next matching breakpoint, or **Resume** for continuous execution. Checkpoints cover model requests, memory commits, context packing, MAGE actions, scoring, and comparison-arm startup where applicable.
4. Use **Pause** or **Stop** to wait at the next boundary. A request already in flight is allowed to finish and log its response. **Force stop** terminates the local worker immediately and can leave that request's final outcome or charge unknown. The timeline and artifact files remain available.

## What is recorded

- Exact Polza request bodies, model outputs and provider-returned reasoning when supplied, streaming progress, token usage, charges, and errors. Authorization headers stay outside the trace.
- Incoming evidence, validation rejections, pre-commit extracted facts and aliases, committed tables and concept state, retrieval scope, ranked candidates, packing decisions, and the exact packed context.
- MAGE's parsed actions, effective observations, environment snapshots and restorations, active tree pointers, summaries, arm scores, and configuration. The timeline preserves actions before tool execution and extraction candidates before they can change memory.

The **Artifacts** tab includes prior run files, worker output, and journals. Runs launched before this console can still be inspected as their files update. Control applies to workers launched by the current console session. The shopping test trace pauses a real Polza extraction before commit and then stops it. The paired test trace pauses before an author tool action, executes that action with one step, and stops before the next request.

The console opens **Results** when a run ends. Paired runs show each arm's traveler scores, full-group outcome, requests, tokens, billed cost, and wall time. Diagnostic runs show each recorded decision or retrieval outcome and shared run usage. Failed or stopped runs show the results and usage recorded before they ended.

**Travel · Full benchmark** runs all 270 travel groups, including the 180-group held-out split. Choose PCMS alone with the published MAGE reference, or run both agents for a matched comparison. The launch form defaults to 12 actions per attempt, 256 PCMS memory tokens, actor reasoning disabled, and eight groups in parallel. Paired runs alternate arm order by group. The Results tab updates as groups finish and includes known charges from failed attempts. Pause waits for every active group to reach a checkpoint. Resume continues the workers, and Stop ends the run at their next checkpoints. Keep the lab server and machine running for the duration.

The travel writer asks the model to index numbered spans of the traveler's request. Code copies the original text into the journal and records selected itinerary options directly from the tool state. For a malformed or truncated extraction, the repair requests fewer spans per call. The replacement PCMS-only run checks the seven groups that failed in the original paired run before scheduling the rest of the release.

## Input files

The shopping launch uses `tmp/concept-subspaces-shopping-release.jsonl` from the pinned [MemoryArena shopping release](https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/da1a37c8b19280e18627ca01cf368195a5e1d92e/bundled_shopping/data.jsonl). The paired launch uses the existing pinned `data/hidden/memoryarena_travel.jsonl`, the author checkout in `external/MAGE`, and `tmp/concept-subspaces-travel-train.csv` from the [TravelPlanner release](https://huggingface.co/datasets/osunlp/TravelPlanner/resolve/8736504/train.csv). The travel confirmation diagnostic uses in-repository development fixtures.
