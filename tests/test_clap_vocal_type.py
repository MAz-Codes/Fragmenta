"""The Music template only says "VocalType: Instrumental" when CLAP is confident.

Uses a stub CLAP model, so no checkpoint or GPU is needed. Pure stdlib + the
project venv; no pytest required. Run:
    venv/bin/python tests/test_clap_vocal_type.py
"""
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.backend.data import auto_annotator as AA  # noqa: E402
from app.backend.data.projects import PROMPT_TEMPLATE_PRESETS, apply_template  # noqa: E402

VOCAL, INSTRUMENTAL = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])


class StubClap:
    """Each audio window embeds to its own first sample, a 'vocalness' angle."""

    def __init__(self):
        self.batches = []

    def get_text_embedding(self, prompts, use_tensor=True):
        side = VOCAL if prompts is AA.CLAP_VOCAL_PROMPTS["vocal"] else INSTRUMENTAL
        return side.repeat(len(prompts), 1)

    def get_audio_embedding_from_data(self, x, use_tensor=True):
        self.batches.append(x.shape[0])
        angle = x[:, 0]
        return torch.stack([angle, 1 - angle], dim=1)


def tagger():
    t = AA._ClapTagger(Path("unused"))
    t._model = StubClap()
    return t


def clip(*angles, tail=0):
    """One CLAP window per angle (+ optional short tail)."""
    y = np.concatenate([np.full(AA.CLAP_WIN, a, dtype=np.float32) for a in angles] + [np.zeros(tail, np.float32)])
    return y, AA.CLAP_SR


def tag(*angles, tail=0):
    t = tagger()
    with mock.patch.object(AA, "load_mono", return_value=clip(*angles, tail=tail)):
        return t.tag(Path("x.wav"), {}), t


def test_confident_instrumental_is_tagged():
    out, _ = tag(*[0.0] * 10)
    assert out.get("vocal_type") == "Instrumental", out


def test_vocals_leave_the_tag_out():
    out, _ = tag(*[1.0] * 10)
    assert "vocal_type" not in out, out


def test_a_short_vocal_passage_is_enough_to_leave_it_out():
    # Two sung windows in twenty sit above the 90th percentile.
    out, _ = tag(*[0.0] * 17, 1.0, 1.0, 1.0)
    assert "vocal_type" not in out, out


def test_genre_embedding_still_only_hears_the_first_windows():
    t = tagger()
    y, sr = clip(*[1.0] * 30)
    with mock.patch.object(AA, "load_mono", return_value=(y, sr)) as load:
        embeds = t._embed_windows(Path("x.wav"))
    assert load.call_args.kwargs["duration"] == AA.CLAP_VOCAL_MAX_WINDOWS * 10
    assert embeds.shape[0] == 30
    # The same first batch as before the vocal gate, so genre/mood don't move.
    assert t._model.batches[0] == AA.CLAP_MAX_WINDOWS


def test_template_keeps_or_drops_the_segment():
    template = PROMPT_TEMPLATE_PRESETS["music"]["template"]
    assert apply_template(template, {"vocal_type": "Instrumental", "bpm": 120}) == (
        "TrackType: Music, VocalType: Instrumental, BPM: 120")
    assert apply_template(template, {"bpm": 120}) == "TrackType: Music, BPM: 120"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
