"""Print observed readiness and research blockers without fabricating benchmark evidence."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from constraint_memory.benchmark_adapter import verify_readiness
from constraint_memory.config import sha256
from constraint_memory.orchestration.logging import write_json

root=Path(__file__).resolve().parents[1]
readiness=verify_readiness(root)
probe=root/"runs/provider_probe/result.json"
verification=root/"reports/verification.json"
live_manifests=sorted((root/"runs").glob("qwen36-*/manifest.json"))
live_runs=[]
for path in live_manifests:
    manifest=json.loads(path.read_text(encoding="utf-8"))
    completed_conditions={}
    for score_path in path.parent.glob("official_scores_*.json"):
        score=json.loads(score_path.read_text(encoding="utf-8"))
        completed_conditions[score_path.stem.removeprefix("official_scores_")]={
            "metrics":score["official_metrics"], "groups":len(score["per_group"])}
    archive_path=path.parent/"source_archive.json"
    archive=json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.exists() else {}
    live_runs.append({"path":str(path.parent.relative_to(root)),"status":manifest["status"],
        "error":manifest.get("error"),"completed_conditions":completed_conditions,
        "paired_run_complete":manifest["status"]=="complete" and len(completed_conditions)==len(manifest["variants"]),
        "original_source_pin_verification":archive.get("original_source_pin_verification","not_archived")})
attempts=[]
for path in sorted((root/"reports").glob("qwen36-*_failure.json"),key=lambda path:path.stat().st_mtime):
    failure=json.loads(path.read_text(encoding="utf-8"))
    attempts.append({"path":str(path.relative_to(root)),"status":failure.get("status"),
        "error":failure.get("error"),"usage":failure.get("usage"),
        "request_sent":[record.get("request_sent") for record in failure.get("records",[])],
        "completed_model_calls":sum(record.get("request_status")=="completed" for record in failure.get("records",[]))})
freeze_path=root/"reports/E3R_development_source_freeze.json"
freeze=json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
freeze_matches=bool(freeze.get("source_sha256")) and all((root/path).is_file() and sha256(root/path)==digest
    for path,digest in freeze.get("source_sha256",{}).items())
portable_runs=[]
for path in sorted((root/"runs").glob("portable-transfer-*/manifest.json")):
    row=json.loads(path.read_text(encoding="utf-8"))
    portable_runs.append({"path":str(path.parent.relative_to(root)),"status":row.get("status"),
                         "completed_conditions":row.get("completed_conditions"),"error":row.get("error"),
                         "complete_comparison":row.get("status")=="complete"})
freeform_runs=[]
freeform_summary_path=root/"reports/freeform_live_verification/summary.json"
freeform_summary=json.loads(freeform_summary_path.read_text(encoding="utf-8")) if freeform_summary_path.exists() else None
for path in sorted((root/"runs").glob("freeform-transfer-*/manifest.json")):
    row=json.loads(path.read_text(encoding="utf-8"))
    freeform_runs.append({"path":str(path.parent.relative_to(root)),"status":row.get("status"),
        "completed_conditions":row.get("completed_conditions"),"error":row.get("error"),"passed":row.get("passed"),
        "usage":row.get("usage"),"independently_verified":bool(freeform_summary and freeform_summary["run_id"]==row.get("run_id")
            and freeform_summary["status"]=="independently_verified_complete")})
cross_family_runs=[]
for path in sorted((root/"runs").glob("cross-family-*/manifest.json")):
    row=json.loads(path.read_text(encoding="utf-8"))
    completion_path=path.parent/"session_completions.jsonl"
    planning_sessions=sum(bool(line.strip()) for line in completion_path.read_text(encoding="utf-8").splitlines()) if completion_path.exists() else 0
    cross_family_runs.append({"path":str(path.parent.relative_to(root)),"status":row.get("status"),
        "protocol_label":row.get("protocol_label"),"completed_conditions":row.get("completed_conditions"),
        "completed_planning_sessions":planning_sessions,"parent_run_id":row.get("parent_run_id"),
        "error":row.get("error"),"live_comparison_validated":row.get("live_comparison_validated",False)})
recovery_pin_path=root/"reports/cross_family_recovery/source_freeze.json"
recovery_proof_path=root/"reports/cross_family_recovery/replay_verified.json"
cross_family_recovery={"parent_run_id":"cross-family-primary-20261015T192052114101Z",
    "runtime":"experiments/cross_family_recovery/run.py","ready":False,
    "source_freeze":"reports/cross_family_recovery/source_freeze.json",
    "offline_verification":"reports/cross_family_recovery/replay_verified.json",
    "original_complete_planning_sessions":6,"original_accepted_responses":155,
    "original_known_cost_rub_exact":"294.18187364","original_total_cost_rub":None,
    "original_failed_bill_known":False,"new_paid_process_started_by_assistant":False}
if recovery_pin_path.exists() and recovery_proof_path.exists():
    pin=json.loads(recovery_pin_path.read_text(encoding="utf-8"))
    proof=json.loads(recovery_proof_path.read_text(encoding="utf-8"))
    parent_dir=root/"runs"/cross_family_recovery["parent_run_id"]
    actual_parent={p.relative_to(parent_dir).as_posix():sha256(p) for p in sorted(parent_dir.rglob("*")) if p.is_file()}
    recovery_source={p.relative_to(root).as_posix():sha256(p) for p in sorted((root/"experiments/cross_family_recovery").rglob("*.py"))}
    cross_family_recovery["ready"]=(proof.get("status")=="exact_semantic_prefix"
        and proof.get("accepted_responses_reused")==155 and pin.get("replay_verified_sha256")==sha256(recovery_proof_path)
        and recovery_source==pin.get("recovery_source_sha256") and actual_parent==pin.get("parent_files_sha256"))
    cross_family_recovery["unit_tests_passed"]=pin.get("unit_tests",{}).get("passed")
first_claim_path=root/"reports/cross_family_recovery/live_lineage_claim.json"
first_claim=json.loads(first_claim_path.read_text(encoding="utf-8")) if first_claim_path.exists() else None
cross_family_recovery["offline_prefix_verified"]=cross_family_recovery["ready"]
cross_family_recovery["lineage_claimed"]=first_claim is not None
cross_family_recovery["ready"]=cross_family_recovery["ready"] and first_claim is None
if first_claim:
    child_manifest_path=root/"runs"/first_claim["child_run_id"]/"manifest.json"
    child_manifest=json.loads(child_manifest_path.read_text(encoding="utf-8")) if child_manifest_path.exists() else {}
    cross_family_recovery.update(child_run_id=first_claim["child_run_id"], child_status=child_manifest.get("status"),
        child_error=child_manifest.get("error"), repeat_live_command=False,
        instruction="Preserve the used claim and both runs. Use only the separately audited second continuation.")
second_reports=root/"reports/cross_family_recovery_second"
second_pin_path=second_reports/"source_freeze.json"
second_proof_path=second_reports/"replay_verified.json"
second_recovery={"runtime":"experiments/cross_family_recovery_second/run.py", "ready":False,
    "source_freeze":"reports/cross_family_recovery_second/source_freeze.json",
    "offline_verification":"reports/cross_family_recovery_second/replay_verified.json",
    "parent_run_id":"cross-family-recovery-20261016T144804072163Z",
    "original_run_id":cross_family_recovery["parent_run_id"],
    "accepted_responses_reused":155, "inherited_sent_failures":2,
    "remaining_new_olivia_actor_calls":27, "failed_bills_known":False,
    "known_accepted_cost_rub_exact":"294.18187364", "actual_total_cost_rub":None,
    "new_request_transport":"recorded_chat_sse", "automatic_paid_retries":0,
    "prior_lineage_claim_retained":first_claim is not None,
    "new_paid_process_started_by_assistant":False}
if second_pin_path.exists() and second_proof_path.exists():
    second_pin=json.loads(second_pin_path.read_text(encoding="utf-8"))
    second_proof=json.loads(second_proof_path.read_text(encoding="utf-8"))
    second_source={p.relative_to(root).as_posix():sha256(p)
        for p in sorted((root/"experiments/cross_family_recovery_second").rglob("*.py"))}
    predecessors_unchanged=True
    for run_id, pin_field in ((second_recovery["original_run_id"],"parent_files_sha256"),
                              (second_recovery["parent_run_id"],"child_files_sha256")):
        run_directory=root/"runs"/run_id
        actual={p.relative_to(run_directory).as_posix():sha256(p)
            for p in sorted(run_directory.rglob("*")) if p.is_file()}
        predecessors_unchanged &= actual==second_pin.get(pin_field)
    second_recovery["offline_prefix_verified"]=(second_proof.get("status")=="exact_semantic_prefix"
        and second_proof.get("accepted_responses_reused")==155 and second_proof.get("model_calls")==0
        and second_proof.get("inherited_sent_failures")==2
        and second_proof.get("remaining_new_olivia_actor_calls")==27
        and second_pin.get("replay_verified_sha256")==sha256(second_proof_path)
        and second_source==second_pin.get("source_sha256") and predecessors_unchanged
        and second_pin.get("unit_tests",{}).get("exit_code")==0)
    second_recovery["unit_tests_passed"]=second_pin.get("unit_tests",{}).get("passed")
    second_recovery["both_predecessors_unchanged"]=predecessors_unchanged
    second_recovery["lineage_claimed"]=(second_reports/"live_lineage_claim.json").exists()
    second_recovery["ready"]=(second_recovery["offline_prefix_verified"]
        and not second_recovery["lineage_claimed"] and second_recovery["prior_lineage_claim_retained"])
continuation={"runtime":"experiments/cross_family_continuation/run.py", "ready":False,
    "new_paid_process_started_by_assistant":False, "automatic_paid_retries":0,
    "minimum_new_request_start_interval_seconds":10, "detailed_console_progress":True,
    "console_heartbeat_seconds":10}
preparations=sorted((root/"reports/cross_family_continuation/preparations").glob("*/source_freeze.json"),
                    key=lambda path:path.stat().st_mtime)
if preparations:
    pin_path=preparations[-1]
    current_pin=json.loads(pin_path.read_text(encoding="utf-8"))
    proof_path=pin_path.parent/"replay_verified.json"
    current_proof=json.loads(proof_path.read_text(encoding="utf-8"))
    actual_source={p.relative_to(root).as_posix():sha256(p)
        for p in sorted((root/"experiments/cross_family_continuation").rglob("*.py"))}
    lineage_unchanged=True
    for run_id, expected in current_pin["lineage_files_sha256"].items():
        run_directory=root/"runs"/run_id
        actual={p.relative_to(run_directory).as_posix():sha256(p)
                for p in sorted(run_directory.rglob("*")) if p.is_file()}
        lineage_unchanged &= actual==expected
    for relative, expected in current_pin.get("claim_files_sha256", {}).items():
        lineage_unchanged &= (root/relative).is_file() and sha256(root/relative)==expected
    claimed=(root/"reports/cross_family_continuation/claims"/(current_pin["parent_run_id"]+".json")).exists()
    continuation.update(parent_run_id=current_pin["parent_run_id"], source_freeze=str(pin_path.relative_to(root)),
        unit_tests_passed=current_pin["unit_tests"]["passed"], lineage_unchanged=lineage_unchanged,
        lineage_claimed=claimed, accepted_responses_not_repurchased=current_proof["accepted_responses_not_repurchased"],
        retained_closed_session_responses=current_proof["retained_closed_session_responses"],
        native_active_session_responses_replayed=current_proof["native_active_session_responses_replayed"],
        inherited_sent_failures=current_proof["inherited_sent_failures"],
        recovery_disclosure=current_proof.get("recovery_disclosure"),
        ready=(not claimed and lineage_unchanged and actual_source==current_pin["source_sha256"]
            and current_proof["status"]=="exact_semantic_prefix" and current_proof["model_calls"]==0
            and sha256(proof_path)==current_pin["replay_verified_sha256"]
            and current_pin["unit_tests"]["exit_code"]==0))
mage_release_path=root/"reports/MAGE_release_recheck_20261016/evidence.json"
mage_release=json.loads(mage_release_path.read_text(encoding="utf-8")) if mage_release_path.exists() else None
completion_path=root/"reports/cross_family_completion/completion.json"
cross_family_completion=json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.exists() else None
if cross_family_completion:
    completed_directory=root/cross_family_completion["directory"]
    completed_manifest=json.loads((completed_directory/"manifest.json").read_text(encoding="utf-8"))
    completion_evidence_matches=all((root/path).is_file() and sha256(root/path)==digest
        for path,digest in cross_family_completion["evidence_sha256"].items())
    completion_verified=(cross_family_completion.get("status")=="independently_verified_complete"
        and completed_manifest.get("status")=="complete" and completion_evidence_matches
        and cross_family_completion.get("completed_sessions")==14
        and cross_family_completion.get("processed_travelers")==98)
    continuation.update(status="complete" if completion_verified else "completion_evidence_changed",
        ready=False, child_run_id=cross_family_completion["run_id"], child_status=completed_manifest["status"],
        completed_run_verified=completion_verified, repeat_live_command=False)
status={"implementation":"working local harness; matched MAGE evaluation pending",
        "local_verification":json.loads(verification.read_text(encoding="utf-8")) if verification.exists() else None,
        "model":"qwen/qwen3.6-27b", "MAGE_main_model_verified":True,
        "MAGE_primary_source":"https://arxiv.org/html/2606.06090v1#S4.SS1",
        "dataset_groups":readiness["provenance"]["group_count"],
        "base_template_families":readiness["provenance"]["base_family_count"],
        "official_database_complete":readiness["database"]["complete"],
        "provider_probe":json.loads(probe.read_text(encoding="utf-8")) if probe.exists() else {"status":"not_run"},
        "provider_probe_is_historical":True,
        "failed_live_attempts":attempts,
        "candidate_source_freeze_matches_current":freeze_matches,
        "live_runs":live_runs,
        "structured_transfer_runs":portable_runs,
        "freeform_transfer_runs":freeform_runs,
        "cross_family_runs":cross_family_runs,
        "cross_family_recovery":cross_family_recovery,
        "cross_family_second_recovery":second_recovery,
        "cross_family_continuation":continuation,
        "cross_family_completion":cross_family_completion,
        "recovery":"causal scoped planner repair shares the current arrival step allowance; E3I measured on one development family, E3R unmeasured",
        "authentic_MAGE_source":mage_release["repository"]["url"] if mage_release else "release not yet verified",
        "authentic_MAGE_source_commit":mage_release["repository"]["head_commit"] if mage_release else None,
        "authentic_MAGE_release_evidence":"reports/MAGE_release_recheck_20261016/evidence.json" if mage_release else None,
        "authentic_MAGE_author_runtime_ready":False,
        "superiority_claim_supported":False,
        "next_valid_experiment":"The latest continuation saved 12 new responses before an unsent, verified-zero-cost preflight timeout. Retain 231 accepted responses; four prior sent failures remain unbilled/unknown. Planning: 10/14 sessions closed, 74/98 travelers processed, 24 remaining; Chloe arrival 5 in E2:64 is next with 30 actor calls. Verbose timestamps, connection stage/attempts, stream counters, accepted-response tokens/cost, ten-second heartbeats, remaining counts and rough ETA are now observable without changing prompts/retries. Reconstruct 12 active replies and all causal logs/snapshots before new requests. Prior declared writer bound99/total3039 remains unchanged: infrastructure_recovered_diagnostic, not original fixed-dispatch comparison. Same model/route/inference/memory32768/grid; no money or aggregate-token stops, padding, partial scoring or MAGE superiority claim.",
        "development_diagnostics":"reports/development_memory_stress_audit.md",
        "proposed_fixes_applied":True,
        "current_constraint_candidate":"E3R verified request frontier; portable ConstraintMemory API is separately tested and shares the predicate kernel",
        "frontier_prototype_live_efficacy_measured":True,
        "inheritance_candidate_live_efficacy_measured":True,
        "request_candidate_live_efficacy_measured":False,
        "portable_model_transfer_measured":bool(freeform_summary),
        "portable_model_transfer_scope":"Ten project-authored development workflows, one decision per case; no broad transfer claim",
        "broad_generality_established":False,
        "freeform_live_result":freeform_summary,
        "advisory_writer_promoted":False,
        "partial_structured_model_observations_available":bool(portable_runs),
        "partial_structured_transfer_verification":"reports/portable_transfer_partial_verification/verification.md",
        "freeform_runtime":"experiments/freeform_memory/run.py",
        "freeform_runtime_is_unpromoted":True,
        "cross_family_runtime":"experiments/cross_family/run.py",
        "cross_family_protocol":"reports/cross_family_mechanism/protocol.md",
        "cross_family_source_freeze":"reports/cross_family_mechanism/source_freeze.json",
        "cross_family_runtime_is_unpromoted":True,
        "generality_release_contract":"reports/generalization_contract.md",
        "latest_verified_paired_run":"runs/qwen36-E0-E3I-vpn-off-20261014T135633482517Z",
        "latest_result_audit":"reports/third_live_pilot_result.md",
        "candidate_budget_audit":"reports/E3R_constraint_integration/verification.md"}
if cross_family_completion:
    status["latest_verified_paired_run"]=(cross_family_completion["directory"] if completion_verified else None)
    status["latest_result_audit"]="reports/cross_family_completion/analysis.md"
    status["dependency_closure_promoted"]=False
    status["current_constraint_candidate"]=(
        "E3R remains an unpromoted prototype; the factual ledger is retained as the development control. "
        "A new constraint-memory intervention has not been selected on independent transfer evidence.")
    status["cross_family_completion_evidence_matches"]=completion_evidence_matches
    status["next_valid_experiment"]=(
        "The seven-family E1/E2 development screen is complete; do not rerun a consumed parent. "
        "E1 passes 18/49 full traveler plans and E2 passes 16/49; neither passes a full group. "
        "The current dependency-closure policy is not promoted. Both arms had omitted memory context "
        "at the shared 32768-token ceiling, and eleven missing submissions consumed all 16384 response "
        "tokens as reasoning. These are development observations, not identified causal explanations. "
        "Preserve the simpler ledger control; diagnose faithful obligation retention and current-state "
        "bindings on independently authored multi-action development workflows, then freeze validation. "
        "Integrate the pinned authentic MAGE runtime with matched task visibility, tools and lifetime "
        "accounting before a superiority experiment. No next paid suite is prepared. The completed run "
        "remains infrastructure_recovered_diagnostic with one disclosed extra physical writer request "
        "and four unknown failed bills; no generality or MAGE-superiority claim.")
    if not completion_verified:
        status["next_valid_experiment"]=(
            "Completion evidence changed or no longer verifies. Preserve all run trees and claims; "
            "rerun the independent offline completion audits before interpreting results or preparing "
            "another experiment. No live continuation is ready.")
network_path=root/"reports/cross_family_mechanism/connectivity.json"
if network_path.exists():
    status["current_connectivity"]=json.loads(network_path.read_text(encoding="utf-8"))
write_json(root/"reports/implementation_status.json",status)
print(json.dumps(status,indent=2))
