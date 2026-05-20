from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from onda.utils import play_wav

_CUSTOM_TOKEN_PREFIX = "<custom_token_"


def _token_text_to_code(token_text: str, index: int) -> int | None:
    """Convert a <custom_token_N> text fragment to a SNAC code ID.

    The model emits tokens whose text looks like "<custom_token_N>".  The code
    value is derived from N with an index-dependent offset so that each of the
    7 positions in a SNAC frame maps to a distinct vocabulary range:

        code = N - 10 - (index % 7) * 4096

    Valid codes are in (0, 4096].  Returns None for non-audio tokens.
    """
    s = token_text.strip()
    pos = s.rfind(_CUSTOM_TOKEN_PREFIX)
    if pos == -1:
        return None
    last = s[pos:]
    if last.startswith(_CUSTOM_TOKEN_PREFIX) and last.endswith(">"):
        try:
            n = int(last[14:-1])
            return n - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


@dataclass
class OrpheusBackend:
    model_dir: Path
    voice: str = "tara"
    _llm: object = field(default=None, init=False, repr=False)
    _snac: object = field(default=None, init=False, repr=False)

    def _get_llm(self):
        if self._llm is None:
            import os
            from llama_cpp import Llama

            gguf_path = next(self.model_dir.glob("*.gguf"))
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
            try:
                self._llm = Llama(
                    model_path=str(gguf_path),
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    verbose=False,
                    batch_size=1,
                )
            finally:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
        return self._llm

    def _get_snac(self):
        if self._snac is None:
            from snac import SNAC

            snac_dir = self.model_dir / "snac_24khz"
            source = str(snac_dir) if snac_dir.is_dir() else "hubertsiuzdak/snac_24khz"
            self._snac = SNAC.from_pretrained(source).eval()
        return self._snac

    def _stream_tokens(self, text: str):
        """Yield raw token text strings from the llama.cpp streaming completion."""
        llm = self._get_llm()
        prompt = f"<|audio|>{self.voice}: {text}<|eot_id|><custom_token_4>"
        for chunk in llm(
            prompt,
            max_tokens=2048,
            stream=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            min_p=0.05,
        ):
            yield chunk["choices"][0]["text"]

    def _collect_codes(self, token_gen) -> list[int]:
        """Parse token text stream into a flat list of valid SNAC codes."""
        codes: list[int] = []
        index = 0
        for token_text in token_gen:
            code = _token_text_to_code(token_text, index)
            if code is not None and code > 0:
                codes.append(code)
                index += 1
        return codes

    def _decode_codes(self, codes: list[int]) -> np.ndarray:
        """Decode a flat SNAC code list into a float32 audio waveform."""
        import torch

        n_frames = len(codes) // 7
        if n_frames == 0:
            return np.zeros(0, dtype=np.float32)

        frame = codes[: n_frames * 7]
        c0: list[int] = []
        c1: list[int] = []
        c2: list[int] = []

        for j in range(n_frames):
            i = 7 * j
            # Frame layout (per canopylabs/orpheus-tts reference):
            #   [i+0] coarse   [i+1] mid-a  [i+2] fine-a  [i+3] fine-b
            #   [i+4] mid-b    [i+5] fine-c  [i+6] fine-d
            c0.append(frame[i])
            c1.extend([frame[i + 1], frame[i + 4]])
            c2.extend([frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

        t0 = torch.tensor(c0, dtype=torch.int32).unsqueeze(0)
        t1 = torch.tensor(c1, dtype=torch.int32).unsqueeze(0)
        t2 = torch.tensor(c2, dtype=torch.int32).unsqueeze(0)

        for t in (t0, t1, t2):
            if torch.any(t < 0) or torch.any(t > 4096):
                return np.zeros(0, dtype=np.float32)

        snac = self._get_snac()
        with torch.inference_mode():
            audio_hat = snac.decode([t0, t1, t2])

        return audio_hat.squeeze().numpy().astype(np.float32)

    def synthesize_audio(self, text: str) -> tuple[np.ndarray, int]:
        codes = self._collect_codes(self._stream_tokens(text))
        audio = self._decode_codes(codes)
        return audio, 24000

    def speak(self, text: str, output_path: Path | None = None) -> None:
        import soundfile as sf

        audio, sr = self.synthesize_audio(text)
        if output_path:
            sf.write(str(output_path), audio, sr)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sr)
                play_wav(Path(f.name))


def load(model_dir: Path) -> OrpheusBackend:
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(
            f"No GGUF file found in {model_dir}. Run: onda pull orpheus/3b"
        )
    return OrpheusBackend(model_dir=model_dir)
