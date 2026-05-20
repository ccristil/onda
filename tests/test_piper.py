import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock
from onda.backends.piper import PiperBackend, load


@pytest.fixture
def fake_model_dir(tmp_path):
    name = "en_US-amy-low"
    (tmp_path / f"{name}.onnx").touch()
    (tmp_path / f"{name}.onnx.json").touch()
    return tmp_path


def _make_fake_chunk(samples: int = 100, sample_rate: int = 22050):
    """Create a mock AudioChunk with float32 audio data."""
    chunk = MagicMock()
    chunk.audio_float_array = np.zeros(samples, dtype="float32")
    chunk.sample_rate = sample_rate
    return chunk


def test_load_returns_piper_backend(fake_model_dir):
    backend = load(fake_model_dir)
    assert isinstance(backend, PiperBackend)
    assert backend.model_dir == fake_model_dir


def test_load_raises_if_onnx_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_speak_calls_synthesize_and_plays(fake_model_dir, mocker):
    """speak() with no output_path should synthesize and play the audio."""
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [_make_fake_chunk()]
    # PiperVoice is imported lazily inside _get_voice — mock _get_voice directly
    mocker.patch.object(PiperBackend, "_get_voice", return_value=mock_voice)
    mock_play = mocker.patch("onda.backends.piper.play_wav")

    backend = load(fake_model_dir)
    backend.speak("Hello world")

    mock_voice.synthesize.assert_called_once_with("Hello world")
    mock_play.assert_called_once()


def test_speak_saves_to_output_path(fake_model_dir, tmp_path, mocker):
    """speak() with output_path should write WAV and NOT call play_wav."""
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [_make_fake_chunk(100, 22050)]
    mocker.patch.object(PiperBackend, "_get_voice", return_value=mock_voice)
    mock_play = mocker.patch("onda.backends.piper.play_wav")
    mock_write = mocker.patch("onda.backends.piper.sf.write")

    backend = load(fake_model_dir)
    out = tmp_path / "final.wav"
    backend.speak("Save me", output_path=out)

    mock_write.assert_called_once()
    args = mock_write.call_args[0]
    assert str(out) in args[0]
    mock_play.assert_not_called()
