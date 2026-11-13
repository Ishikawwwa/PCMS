"""Verify source hashes survive logging while credentials remain redacted."""

import pytest

from constraint_memory import secrets as module


@pytest.mark.parametrize('name',['src/constraint_memory/secrets.py',r'src\constraint_memory\secrets.py',
    'src/api_key.py','src/authorization.py'])
def test_only_legitimate_source_digest_keys_are_preserved(name):
    original={'source_sha256':{name:'a'*64}}
    assert module.redact(original)==original


@pytest.mark.parametrize('data',[
    {'secret':'a'*64},{'api_key':'a'*64},{'authorization':'a'*64},
    {'src/secrets.py':'a'*64},{'source_sha256':{'outside/secrets.py':'a'*64}},
    {'source_sha256':{'src/secrets.env':'a'*64}},
    {'source_sha256':{'src/secrets.py':'pza_FAKE_TEST_ONLY'}},
    {'source_sha256':{'src/secrets.py':'not-a-digest'}},
])
def test_sensitive_values_remain_redacted(data):
    assert '[REDACTED]' in str(module.redact(data))


def test_key_value_redaction_nested_and_environment_secret(monkeypatch):
    monkeypatch.setenv('POLZA_API_KEY','fake-environment-value')
    result=module.redact({'logs':[{'message':'prefix fake-environment-value pza_FAKE_TEST_ONLY',
                                 'authorization':'Bearer fake-environment-value'}]})
    assert result['logs'][0]['message']=='prefix [REDACTED] [REDACTED]'
    assert result['logs'][0]['authorization']=='[REDACTED]'


def test_original_metadata_is_not_mutated():
    original={'source_sha256':{'src/secrets.py':'a'*64},'secret':'private'}
    module.redact(original)
    assert original['secret']=='private'
