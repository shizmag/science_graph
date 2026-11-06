import importlib.util,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'integrity.py';s=importlib.util.spec_from_file_location('integrity',p);i=importlib.util.module_from_spec(s);s.loader.exec_module(i)

def test_prepare_standalone_has_no_private_or_code(tmp_path,monkeypatch):
    monkeypatch.setattr(i,'WORK',tmp_path/'judge');monkeypatch.setattr(i,'REVIEW_DIR',tmp_path/'reviews')
    x=i.prepare();assert x['tasks']==200
    assert not (i.WORK/'private').exists()
    names={p.name for p in i.WORK.iterdir()};assert 'mapping.json' not in names and 'bridge.py' not in names
    assert len(list((i.WORK/'tasks').glob('*.json')))==200

def test_inventory_empty_is_safe(tmp_path,monkeypatch):
    monkeypatch.setattr(i,'WORK',tmp_path/'judge');monkeypatch.setattr(i,'REVIEW_DIR',tmp_path/'reviews');i.prepare()
    report=i.inventory()[0];assert report['core_structurally_valid']==0 and report['core_pending']==200
