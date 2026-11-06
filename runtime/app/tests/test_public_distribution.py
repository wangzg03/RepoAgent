"""Distribution controls: optional data and relocatable preparation, no inference."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def installer():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('prepare_official_fixture', root/'scripts/prepare_humaneval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_provisioning_checks_hash_and_preserves_conflicting_files(tmp_path):
    helper = installer()
    raw = b'public synthetic installer control, not benchmark data'
    manifest = {'files':{helper.DATA:hashlib.sha256(raw).hexdigest()}}
    (tmp_path/'UPSTREAM.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='hash_mismatch'):
        helper.install(b'wrong', tmp_path)
    assert not (tmp_path/helper.DATA).exists()
    target = helper.install(raw, tmp_path)
    assert target.read_bytes() == raw
    assert helper.install(raw, tmp_path) == target
    target.write_bytes(b'preserve existing local file')
    with pytest.raises(ValueError, match='existing_humaneval_data_differs'):
        helper.install(raw, tmp_path)
    assert target.read_bytes() == b'preserve existing local file'


def test_missing_official_data_explains_explicit_setup_without_network(tmp_path, monkeypatch):
    import urllib.request

    from repoagent import benchmarks
    monkeypatch.setattr(benchmarks, 'OFFICIAL', tmp_path)
    (tmp_path/'UPSTREAM.json').write_text(json.dumps({'files':{'data/HumanEval.jsonl.gz':'0'*64}}))
    def forbidden(*args, **kwargs):
        raise AssertionError('optional data cannot trigger implicit networking')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    with pytest.raises(FileNotFoundError, match='prepare_humaneval.py'):
        benchmarks.verify_upstream()


def test_local_example_profile_prepares_without_credentials_or_docker(tmp_path, monkeypatch):
    from repoagent import legacy_launcher
    from repoagent.cli import ROOT, prepare
    def forbidden(*args, **kwargs):
        raise AssertionError('prepare cannot start a process or access credentials')
    monkeypatch.setattr(legacy_launcher, 'credential_environment', forbidden)
    monkeypatch.setattr(legacy_launcher.subprocess, 'Popen', forbidden)
    config = prepare('Create a tested calculator', tmp_path/'delivery', tmp_path/'packet',
                     profile=ROOT/'config/provider.local.example.json')
    assert config['image'] == 'repoagent-sandbox:local'
    assert config['provider'] == 'ollama'
    assert not (tmp_path/'packet/run').exists()
