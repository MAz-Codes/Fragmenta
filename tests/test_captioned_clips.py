"""Only clips with prompts reach pre-encode and training (1.2 follow-up 2).

SA3 swaps a clip without a .txt for a random other clip (duplicate latents)
and trains a clip with an empty .txt on an empty prompt; Fragmenta passes a
filelist of the captioned clips instead.

Run:  PYTHONPATH=vendor/stable-audio-3 python tests/test_captioned_clips.py
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.core.training.sa3_lora_runner import (  # noqa: E402
    audio_files, build_train_command, captioned_clips, write_filelist)

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def clip(path, prompt=None, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * 44100)) / 44100
    sf.write(str(path), np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32) * 0.3, 44100)
    if prompt is not None:
        path.with_suffix(".txt").write_text(prompt, encoding="utf-8")


proj = Path(tempfile.mkdtemp(prefix="fragmenta-captioned-"))
clip(proj / "a.wav", "warm pad")
clip(proj / "b.wav", "   \n")             # committed with no prompt
clip(proj / "c.flac")                      # added after the last commit
clip(proj / "sub" / "d.wav", "kick drum")
clip(proj / "ünï cödé.wav", "tape loop")   # non-ASCII name
clip(proj / ".hidden.wav", "hidden")
clip(proj / ".latents" / "x.wav", "cache")
(proj / "notes.txt").write_text("not a caption")

print("=== which clips are used ===")
names = lambda paths: sorted(p.relative_to(proj).as_posix() for p in paths)
check("audio files: accepted formats, subfolders, no hidden", names(audio_files(proj)) ==
      ["a.wav", "b.wav", "c.flac", "sub/d.wav", "ünï cödé.wav"], str(names(audio_files(proj))))
clips = captioned_clips(proj)
check("captioned: only clips with a non-empty prompt", names(clips) == ["a.wav", "sub/d.wav", "ünï cödé.wav"], str(names(clips)))

print("=== filelist ===")
fl = write_filelist(proj, clips, proj.parent / f"{proj.name}-clips.txt")
lines = fl.read_text(encoding="utf-8").splitlines()
check("one relative POSIX path per line, UTF-8", sorted(lines) == ["a.wav", "sub/d.wav", "ünï cödé.wav"], str(lines))

print("=== the vendored loader honours it ===")
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("pre_encode_dataset", REPO / "vendor/stable-audio-3/scripts/pre_encode_dataset.py")
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)
from stable_audio_3.data.dataset import LocalDatasetConfig, SampleDataset  # noqa: E402


def dataset(filelist):
    return SampleDataset([LocalDatasetConfig(id="t", path=str(proj), custom_metadata_fn=pre.caption_metadata_fn,
                                             filelist_path=filelist)], sample_size=44100, sample_rate=44100)


full = dataset(None)
listed = dataset(str(fl))
# Upstream also scans hidden folders, so it finds .latents/x.wav too.
check("without a filelist SA3 sees every clip", len(full) == 6, str(len(full)))
check("with it, only the captioned ones", len(listed) == 3, str(len(listed)))
items = [listed[i][1] for i in range(len(listed))]
paths = [Path(md["path"]).name for md in items]
check("every item is its own clip (no swaps)", sorted(paths) == ["a.wav", "d.wav", "ünï cödé.wav"], str(paths))
check("every item has its own prompt", {Path(md["path"]).name: md["prompt"] for md in items} ==
      {"a.wav": "warm pad", "d.wav": "kick drum", "ünï cödé.wav": "tape loop"})

print("=== command ===")
kw = dict(venv_python="python", sa3_vendor_dir=REPO / "vendor/stable-audio-3", sa3_model_name="small-music-base",
          data_dir=proj, save_dir=proj / "out")
cmd = build_train_command(filelist=fl, **kw)
check("live encoding passes --filelist", cmd[cmd.index("--filelist") + 1] == str(fl) if "--filelist" in cmd else False)
cmd = build_train_command(filelist=fl, encoded_dir=proj / ".latents", **kw)
check("pre-encoded latents don't need it", "--filelist" not in cmd and "--encoded_dir" in cmd)

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("All captioned-clip checks passed.")
