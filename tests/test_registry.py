import json
import pytest
from pathlib import Path
from onda.registry import load_registry, get_model, is_downloaded, model_dir, ModelEntry


SAMPLE_MODELS = [
    {
        "name": "en_US-amy-low",
        "backend": "piper",
        "onnx_url": "https://example.com/amy.onnx",
        "config_url": "https://example.com/amy.onnx.json",
        "size_mb": 5,
        "description": "Fast model.",
    }
]


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps(SAMPLE_MODELS))
    monkeypatch.setattr("onda.registry.MODELS_PATH", models_file)
    return models_file


def test_load_registry(fake_registry):
    models = load_registry()
    assert len(models) == 1
    assert isinstance(models[0], ModelEntry)
    assert models[0].name == "en_US-amy-low"
    assert models[0].backend == "piper"
    assert models[0].size_mb == 5


def test_get_model_found(fake_registry):
    m = get_model("en_US-amy-low")
    assert m is not None
    assert m.name == "en_US-amy-low"


def test_get_model_not_found(fake_registry):
    assert get_model("nonexistent") is None


def test_model_dir_returns_correct_path():
    d = model_dir("en_US-amy-low")
    assert d == Path.home() / ".onda" / "models" / "en_US-amy-low"


def test_is_downloaded_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("onda.registry.ONDA_DIR", tmp_path)
    assert is_downloaded("en_US-amy-low") is False


def test_is_downloaded_true_when_files_exist(tmp_path, monkeypatch):
    monkeypatch.setattr("onda.registry.ONDA_DIR", tmp_path)
    d = tmp_path / "models" / "en_US-amy-low"
    d.mkdir(parents=True)
    (d / "en_US-amy-low.onnx").touch()
    (d / "en_US-amy-low.onnx.json").touch()
    assert is_downloaded("en_US-amy-low") is True


def test_is_downloaded_false_when_only_onnx(tmp_path, monkeypatch):
    monkeypatch.setattr("onda.registry.ONDA_DIR", tmp_path)
    d = tmp_path / "models" / "en_US-amy-low"
    d.mkdir(parents=True)
    (d / "en_US-amy-low.onnx").touch()
    assert is_downloaded("en_US-amy-low") is False
