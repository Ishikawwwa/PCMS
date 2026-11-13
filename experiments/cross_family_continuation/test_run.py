"""Scoring boundaries and timing provenance with invented isolated sessions."""
import copy
import json
from types import SimpleNamespace

import pytest

from . import run


def fixture_parent():
    groups = ['1', '2', '3', '4', '5', '6', '7']
    schedule = [{'group_id': group, 'variant': variant} for group in groups for variant in ('E1', 'E2')]
    completed = []
    for index, item in enumerate(schedule[:9], 1):
        row = {'group_id': item['group_id'], 'variant': item['variant'], 'base_template_id': item['group_id'],
               'wall_clock_seconds': 21, 'usage': [{'model_call': False, 'latency_seconds': 1}], 'travelers': []}
        if item == {'group_id': '4', 'variant': 'E2'}:
            row['usage'] = [{'group_id': 'E2:4', 'origin_run_id': origin, 'request_status': 'failed', 'cost_rub': None}
                            for origin in ('first', 'second')]
        completed.append({**item, 'execution_index': index, 'row': row, 'predictions': []})
    return {'run_id': 'fixture-parent', 'protocol': {'schedule': schedule, 'group_ids': groups,
            'bounds': {'max_actor_requests': 2940, 'max_writer_requests': 98, 'max_total_model_requests': 3038},
            'config': {'agent': {}, 'experiment': {'upstream_root': 'invented', 'upstream_repo_sha': 'fixture'}}},
            'failed_intent': {'group_id': 'E2:5'},
            'all_failures': [{'group_id': 'E2:5', 'origin_run_id': 'third', 'request_status': 'failed', 'cost_rub': None}],
            'completed': completed,
            'timing_sources': {f"{item['variant']}:{item['group_id']}":
                {'run_id': 'first-completing-run', 'row': copy.deepcopy(item['row']),
                 'mixed': item['variant'] == 'E2' and item['group_id'] == '4'} for item in completed}}


def retained_import(parent, destination, client):
    for item in parent['completed']:
        run.base.frozen.durable(destination / 'session_completions.jsonl', item, append=True)
    return copy.deepcopy(parent['completed']), {'closed_sessions': 9}


def test_offline_boundary_cannot_score_or_construct_closed_sessions(tmp_path, monkeypatch):
    parent = fixture_parent()
    monkeypatch.setattr(run.core, 'import_completed', retained_import)
    monkeypatch.setattr(run.base, 'install_historical_seed_order', lambda session, *_: session)
    opened = []
    class ActiveSession:
        def __init__(self, root, destination, variant, group, *args):
            opened.append((variant, group))
        def run(self):
            raise run.base.ReplayBoundaryReached('invented exact prefix boundary')
    def forbidden_score(*args, **kwargs):
        pytest.fail('Incomplete planning must never invoke the hidden scorer')
    with pytest.raises(run.base.ReplayBoundaryReached):
        run.execute_grid(tmp_path, parent, None, None, SimpleNamespace(live=False),
                         session_factory=ActiveSession, scorer=forbidden_score)
    assert opened == [('E2', '5')]
    assert not (tmp_path / 'planning_complete.json').exists()
    assert not (tmp_path / 'results.jsonl').exists()
    assert len(run.base.rows(tmp_path / 'session_completions.jsonl')) == 9


def test_all_fourteen_finish_before_scoring_and_keep_mixed_times_unknown(tmp_path, monkeypatch):
    parent = fixture_parent()
    parent['timing_sources']['E1:4']['row']['usage'][0]['latency_seconds'] = 88
    monkeypatch.setattr(run.core, 'import_completed', retained_import)
    monkeypatch.setattr(run.base, 'install_historical_seed_order', lambda session, *_: session)
    opened, scoring_calls = [], []
    class ActiveSession:
        def __init__(self, root, destination, variant, group, *args):
            self.variant, self.group = variant, group
            opened.append((variant, group))
        def run(self):
            return [], {'variant': self.variant, 'group_id': self.group, 'base_template_id': self.group,
                'wall_clock_seconds': 4, 'travelers': [], 'usage': [{'model_call': False, 'latency_seconds': 1}]}
    def scorer(*args, **kwargs):
        assert len(run.base.rows(tmp_path / 'session_completions.jsonl')) == 14
        assert (tmp_path / 'planning_complete.json').is_file()
        scoring_calls.append(True)
        return {'per_group': [{'group_id': group, 'base_template_id': group, 'official_metrics': {}}
                              for group in parent['protocol']['group_ids']]}
    rows = run.execute_grid(tmp_path, parent, None, {'provenance': {'dataset_sha256': 'fixture'}},
        SimpleNamespace(live=True), session_factory=ActiveSession, scorer=scorer)
    assert len(rows) == 14 and len(opened) == 5 and len(scoring_calls) == 2
    by_scope = {f"{row['variant']}:{row['group_id']}": row for row in rows}
    assert by_scope['E2:4']['wall_clock_seconds'] is None
    assert by_scope['E2:5']['wall_clock_seconds'] is None
    assert by_scope['E1:4']['wall_clock_seconds'] == 21
    assert by_scope['E1:4']['usage'][0]['latency_seconds'] == 88
    assert by_scope['E1:4']['usage'][0]['new_billable'] is False
    assert by_scope['E2:7']['wall_clock_seconds'] == 4
    assert by_scope['E2:5']['usage'][0]['latency_seconds'] is None
    failures = [usage for row in rows for usage in row['usage'] if usage.get('request_status') == 'failed']
    assert {item['origin_run_id'] for item in failures} == {'first', 'second', 'third'}
    assert all(item['cost_rub'] is None for item in failures)


def test_scoring_function_requires_durable_planning_marker(tmp_path):
    with pytest.raises(run.base.ReplayMismatch, match='complete fixed planning grid'):
        run.finish_scores(tmp_path, fixture_parent(), {}, {}, {}, {})


def test_parent_id_cannot_escape_preparation_directory():
    for parent_id in ('..', '.', '../other', 'nested/parent', 'C:/outside'):
        with pytest.raises(ValueError):
            run.preparation(parent_id)


def test_extra_sent_writer_is_disclosed_without_altering_actor_or_original_bounds():
    parent = fixture_parent()
    parent['all_failures'].extend([
        {'group_id': 'E2:6', 'operation': 'extraction', 'request_sent': True, 'cost_rub': None},
        {'group_id': 'E2:6', 'operation': 'extraction', 'request_sent': False, 'cost_rub': 0},
        {'group_id': 'E2:5', 'operation': 'agent', 'request_sent': True, 'cost_rub': None}])
    before = copy.deepcopy(parent)
    disclosure = run.recovery_disclosure(parent)
    assert disclosure['classification'] == 'infrastructure_recovered_diagnostic'
    assert disclosure['original_dispatch_bound_satisfied'] is False
    assert disclosure['effective_physical_dispatch_bounds'] == {
        'max_actor_requests': 2940, 'max_writer_requests': 99, 'max_total_model_requests': 3039}
    assert disclosure['accepted_writers_per_arrival'] == 1
    assert disclosure['additional_sent_writer_allowance'] == 1
    assert parent == before
