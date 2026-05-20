import pytest
from pathlib import Path
from onda.utils import chunk_text, read_text_file, play_wav, fetch_url_text


def test_chunk_empty_string():
    assert chunk_text("") == []


def test_chunk_short_text_is_single_chunk():
    result = chunk_text("Hello world.")
    assert result == ["Hello world."]


def test_chunk_splits_at_sentence_boundary():
    text = "First sentence. Second sentence. Third sentence."
    result = chunk_text(text, max_chars=30)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 30 or " " not in chunk  # hard-split exception


def test_chunk_accumulates_short_sentences():
    # Both sentences fit in 100 chars — should be one chunk
    text = "Hello. World."
    result = chunk_text(text, max_chars=100)
    assert len(result) == 1
    assert "Hello" in result[0]
    assert "World" in result[0]


def test_chunk_long_sentence_is_hard_split():
    # Single sentence longer than max_chars must be split
    long = "A" * 600
    result = chunk_text(long + ".", max_chars=500)
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk) <= 500


def test_chunk_handles_exclamation_and_question():
    text = "Hello! How are you? I am fine."
    result = chunk_text(text, max_chars=20)
    assert len(result) >= 2


def test_chunk_no_empty_strings_in_result():
    text = "One. Two. Three."
    result = chunk_text(text, max_chars=5)
    assert all(c.strip() for c in result)


def test_read_text_file(tmp_path):
    f = tmp_path / "input.txt"
    f.write_text("Hello from file.")
    assert read_text_file(f) == "Hello from file."


def test_read_text_file_unsupported_extension(tmp_path):
    f = tmp_path / "input.pdf"
    f.write_bytes(b"%PDF-1.4")
    with pytest.raises(SystemExit):
        read_text_file(f)


def test_play_wav_calls_sounddevice(tmp_path, mocker):
    wav = tmp_path / "test.wav"
    # Write a minimal WAV via soundfile
    import numpy as np
    import soundfile as sf
    sf.write(str(wav), np.zeros(1000, dtype="float32"), 22050)

    mock_play = mocker.patch("onda.utils.sd.play")
    mock_wait = mocker.patch("onda.utils.sd.wait")

    play_wav(wav)

    mock_play.assert_called_once()
    mock_wait.assert_called_once()


def _mock_trafilatura(mocker, fetch_return, extract_return=None):
    """Inject a fake trafilatura into sys.modules (it may not be installed yet)."""
    mock = mocker.MagicMock()
    mock.fetch_url.return_value = fetch_return
    mock.extract.return_value = extract_return
    mocker.patch.dict("sys.modules", {"trafilatura": mock})
    return mock


def test_fetch_url_text_returns_extracted_content(mocker):
    _mock_trafilatura(mocker, fetch_return="<html>article</html>", extract_return="  Article body text.  ")
    result = fetch_url_text("https://example.com/article")
    assert result == "Article body text."


def test_fetch_url_text_raises_on_failed_fetch(mocker):
    _mock_trafilatura(mocker, fetch_return=None)
    with pytest.raises(RuntimeError, match="Could not fetch URL"):
        fetch_url_text("https://unreachable.invalid")


def test_fetch_url_text_raises_on_no_extractable_content(mocker):
    _mock_trafilatura(mocker, fetch_return="<html></html>", extract_return=None)
    with pytest.raises(RuntimeError, match="No readable content found"):
        fetch_url_text("https://example.com/login-wall")
