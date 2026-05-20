import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from onda.backends.piper import PiperBackend, load


@pytest.fixture
def fake_model_dir(tmp_path):
    name = "en_US-amy-low"
    (tmp_path / f"{name}.onnx").touch()
    (tmp_path / f"{name}.onnx.json").touch()
    return tmp_path


def test_load_returns_piper_backend(fake_model_dir):
    backend = load(fake_model_dir)
    assert isinstance(backend, PiperBackend)
    assert backend.model_dir == fake_model_dir


def test_load_raises_if_onnx_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def _make_mock_tmpdir(mocker, tmp_path):
    """Return a context manager mock that yields str(tmp_path)."""
    m = mocker.MagicMock()
    m.__enter__ = mocker.MagicMock(return_value=str(tmp_path))
    m.__exit__ = mocker.MagicMock(return_value=False)
    return m


def test_speak_invokes_piper_binary(fake_model_dir, tmp_path, mocker):
    mock_run = mocker.patch("onda.backends.piper.subprocess.run")
    mock_play = mocker.patch("onda.backends.piper.play_wav")
    mocker.patch("onda.backends.piper.get_piper_binary", return_value=Path("/fake/piper"))
    mocker.patch(
        "onda.backends.piper.tempfile.TemporaryDirectory",
        return_value=_make_mock_tmpdir(mocker, tmp_path),
    )

    backend = load(fake_model_dir)
    backend.speak("Hello world")

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "/fake/piper" in cmd
    assert "--model" in cmd
    mock_play.assert_called_once()


def test_speak_saves_to_output_path(fake_model_dir, tmp_path, mocker):
    mocker.patch("onda.backends.piper.subprocess.run")
    mocker.patch("onda.backends.piper.get_piper_binary", return_value=Path("/fake/piper"))
    mock_copy = mocker.patch("onda.backends.piper.shutil.copy2")
    mock_play = mocker.patch("onda.backends.piper.play_wav")
    mocker.patch(
        "onda.backends.piper.tempfile.TemporaryDirectory",
        return_value=_make_mock_tmpdir(mocker, tmp_path),
    )

    backend = load(fake_model_dir)
    out = tmp_path / "final.wav"
    backend.speak("Save me", output_path=out)

    mock_copy.assert_called_once()
    mock_play.assert_not_called()  # play_wav must NOT be called when output_path is set
