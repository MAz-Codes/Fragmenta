"""Automatic audio annotation for bulk dataset creation.

Two tiers:
  - basic: librosa-only DSP (tempo, key). No downloads. CPU. ~instant per file.
  - rich:  basic + LAION-CLAP zero-shot tagging (genre, mood, instrument).
           Lazy-loaded; downloads ~2.35 GB checkpoint on first use.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")

CLAP_CKPT_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_REPO = "lukewys/laion_clap"

KEY_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Schmuckler key profiles.
KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _iter_audio_files(folder: Path) -> List[Path]:
    results: List[Path] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.startswith("."):
                continue
            if name.lower().endswith(AUDIO_EXTENSIONS):
                results.append(Path(root) / name)
    results.sort()
    return results


def _estimate_tempo(y, sr) -> Optional[int]:
    import librosa
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo if hasattr(tempo, "__float__") else tempo[0])
        if bpm <= 0:
            return None
        return int(round(bpm))
    except Exception as exc:
        logger.debug("tempo estimation failed: %s", exc)
        return None


def _estimate_brightness(y, sr) -> Optional[str]:
    import librosa
    try:
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    except Exception as exc:
        logger.debug("centroid estimation failed: %s", exc)
        return None
    if centroid <= 0:
        return None
    if centroid < 1500:
        return "dark"
    if centroid > 3500:
        return "bright"
    return None


def _estimate_character(y, sr) -> Optional[str]:
    import librosa
    import numpy as np
    try:
        harm, perc = librosa.effects.hpss(y)
        eh = float(np.mean(harm ** 2))
        ep = float(np.mean(perc ** 2))
    except Exception as exc:
        logger.debug("HPSS failed: %s", exc)
        return None
    total = eh + ep
    if total <= 0:
        return None
    perc_ratio = ep / total
    if perc_ratio > 0.65:
        return "percussion-driven"
    if perc_ratio < 0.20:
        return "melodic"
    return None


def _estimate_key(y, sr) -> Optional[str]:
    import librosa
    import numpy as np
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        if chroma_mean.sum() <= 0:
            return None
        chroma_mean = chroma_mean / chroma_mean.sum()

        major = np.asarray(KRUMHANSL_MAJOR)
        minor = np.asarray(KRUMHANSL_MINOR)

        best_score = -1.0
        best_key = None
        for i in range(12):
            maj_score = float(np.corrcoef(chroma_mean, np.roll(major, i))[0, 1])
            min_score = float(np.corrcoef(chroma_mean, np.roll(minor, i))[0, 1])
            if maj_score > best_score:
                best_score = maj_score
                best_key = f"{KEY_NAMES_SHARP[i]} major"
            if min_score > best_score:
                best_score = min_score
                best_key = f"{KEY_NAMES_SHARP[i]} minor"
        return best_key
    except Exception as exc:
        logger.debug("key estimation failed: %s", exc)
        return None


def _compose_prompt(parts: Dict[str, Any]) -> str:
    genre = parts.get("genre")
    mood = parts.get("mood")
    instruments = parts.get("instruments") or []
    bpm = parts.get("bpm")
    key = parts.get("key")
    brightness = parts.get("brightness")
    character = parts.get("character")

    head_bits: List[str] = []
    if mood:
        head_bits.append(str(mood))
    if genre:
        head_bits.append(f"{genre} track")
    elif head_bits:
        head_bits[-1] = f"{head_bits[-1]} track"
    opening = " ".join(head_bits)

    descriptors = [d for d in (brightness, character) if d]

    fragments: List[str] = []
    if opening:
        fragments.append(opening)
    if descriptors:
        fragments.append(", ".join(descriptors))
    if bpm:
        fragments.append(f"{bpm} BPM")
    if key:
        fragments.append(f"in {key}")
    if instruments:
        fragments.append("with " + ", ".join(instruments))

    out = ", ".join(fragments)
    return out[:1].upper() + out[1:] if out else ""


class _ClapTagger:
    """Lazy holder for a LAION-CLAP model used for zero-shot tagging."""

    def __init__(self, ckpt_path: Path):
        self.ckpt_path = ckpt_path
        self._model = None
        self._lock = threading.Lock()
        self._label_embeds: Dict[str, Any] = {}

    def ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.ckpt_path.exists():
                raise FileNotFoundError(
                    f"CLAP checkpoint not found at {self.ckpt_path}. "
                    "Download it first via /api/bulk-annotate/download-clap."
                )
            import laion_clap
            import torch
            logging.getLogger("transformers").setLevel(logging.ERROR)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)

            # torch >= 2.6 flipped torch.load(weights_only=True) and newer
            # transformers dropped the roberta position_ids buffer, so
            # laion_clap's own load_ckpt errors twice: unpickling, then strict
            # state_dict mismatch. Replicate its logic safely here.
            from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict

            orig_load = torch.load
            def _trusted_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return orig_load(*args, **kwargs)
            torch.load = _trusted_load
            try:
                state = clap_load_state_dict(str(self.ckpt_path), skip_params=True)
            finally:
                torch.load = orig_load
            missing, unexpected = model.model.load_state_dict(state, strict=False)
            if unexpected:
                logger.debug("CLAP unexpected keys ignored: %s", unexpected[:5])
            if missing:
                logger.debug("CLAP missing keys: %s", missing[:5])
            self._model = model
            self._device = device
            logger.info("CLAP loaded on %s from %s", device, self.ckpt_path)

    def _embed_labels(self, group: str, prompts: List[str]):
        import torch
        key = f"{group}:{'|'.join(prompts)}"
        if key in self._label_embeds:
            return self._label_embeds[key]
        with torch.no_grad():
            embed = self._model.get_text_embedding(prompts, use_tensor=True)
        embed = embed / embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self._label_embeds[key] = embed
        return embed

    def tag(self, audio_path: Path, label_sets: Dict[str, List[str]], top_k_instruments: int = 2) -> Dict[str, Any]:
        self.ensure_loaded()
        import torch

        with torch.no_grad():
            audio_embed = self._model.get_audio_embedding_from_filelist(
                x=[str(audio_path)], use_tensor=True
            )
        audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        out: Dict[str, Any] = {}
        for group in ("genre", "mood"):
            labels = label_sets.get(group) or []
            if not labels:
                continue
            prompts = [f"a {lab} music track" if group == "genre" else f"a {lab} sounding music track" for lab in labels]
            text_embed = self._embed_labels(group, prompts)
            sims = (audio_embed @ text_embed.T).squeeze(0)
            top = int(sims.argmax().item())
            out[group] = labels[top]

        instruments = label_sets.get("instruments") or []
        if instruments:
            prompts = [f"music featuring {lab}" for lab in instruments]
            text_embed = self._embed_labels("instruments", prompts)
            sims = (audio_embed @ text_embed.T).squeeze(0)
            k = min(top_k_instruments, len(instruments))
            top_idx = torch.topk(sims, k=k).indices.tolist()
            out["instruments"] = [instruments[i] for i in top_idx]
        return out


_clap_tagger_singleton: Optional[_ClapTagger] = None
_clap_tagger_lock = threading.Lock()


def get_clap_tagger(clap_ckpt_path: Path) -> _ClapTagger:
    global _clap_tagger_singleton
    with _clap_tagger_lock:
        if _clap_tagger_singleton is None or _clap_tagger_singleton.ckpt_path != clap_ckpt_path:
            _clap_tagger_singleton = _ClapTagger(clap_ckpt_path)
    return _clap_tagger_singleton


def clap_checkpoint_path(models_pretrained_dir: Path) -> Path:
    return models_pretrained_dir / "clap" / CLAP_CKPT_FILENAME


def clap_checkpoint_available(models_pretrained_dir: Path) -> bool:
    return clap_checkpoint_path(models_pretrained_dir).exists()


def download_clap_checkpoint(
    models_pretrained_dir: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    target = clap_checkpoint_path(models_pretrained_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    from huggingface_hub import hf_hub_download
    import os

    if progress_cb:
        progress_cb("Downloading CLAP checkpoint (~630 MB)…")

    # Use custom CLAP from fragmenta-models on HF Spaces
    use_custom_repo = os.getenv('FRAGMENTA_USE_CUSTOM_MODELS', '').lower() == 'true'
    if use_custom_repo:
        repo_id = "MazCodes/fragmenta-models"
    else:
        repo_id = CLAP_REPO

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=CLAP_CKPT_FILENAME,
        local_dir=str(target.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != target:
        try:
            downloaded_path.replace(target)
        except OSError:
            import shutil
            shutil.copy2(downloaded_path, target)
    return target


def load_label_sets(label_sets_path: Optional[Path]) -> Dict[str, List[str]]:
    if not label_sets_path or not label_sets_path.exists():
        return {"genre": [], "mood": [], "instruments": []}
    with open(label_sets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "genre": list(data.get("genre") or []),
        "mood": list(data.get("mood") or []),
        "instruments": list(data.get("instruments") or []),
    }


def annotate_file(
    audio_path: Path,
    tier: str,
    clap_tagger: Optional[_ClapTagger],
    label_sets: Dict[str, List[str]],
    sr: int = 22050,
    max_seconds: float = 60.0,
) -> Dict[str, Any]:
    import librosa

    parts: Dict[str, Any] = {}
    try:
        y, loaded_sr = librosa.load(str(audio_path), sr=sr, mono=True, duration=max_seconds)
    except Exception as exc:
        logger.warning("librosa failed to load %s: %s", audio_path.name, exc)
        return {
            "file_name": audio_path.name,
            "prompt": "",
            "path": str(audio_path),
            "error": f"load failed: {exc}",
        }

    parts["bpm"] = _estimate_tempo(y, loaded_sr)
    parts["key"] = _estimate_key(y, loaded_sr)
    parts["brightness"] = _estimate_brightness(y, loaded_sr)
    parts["character"] = _estimate_character(y, loaded_sr)

    if tier == "rich" and clap_tagger is not None:
        try:
            tags = clap_tagger.tag(audio_path, label_sets)
            parts.update(tags)
        except Exception as exc:
            logger.warning("CLAP tagging failed for %s: %s", audio_path.name, exc)

    prompt = _compose_prompt(parts)
    return {
        "file_name": audio_path.name,
        "prompt": prompt,
        "path": str(audio_path),
        "attributes": parts,
    }


def annotate_folder(
    folder: Path,
    tier: str,
    label_sets: Dict[str, List[str]],
    clap_ckpt_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    files = _iter_audio_files(folder)
    if not files:
        raise ValueError(f"No audio files found in {folder}")

    clap_tagger: Optional[_ClapTagger] = None
    if tier == "rich":
        if not clap_ckpt_path or not Path(clap_ckpt_path).exists():
            raise FileNotFoundError(
                "Rich tier requires the CLAP checkpoint; download it first."
            )
        clap_tagger = get_clap_tagger(Path(clap_ckpt_path))
        clap_tagger.ensure_loaded()

    results: List[Dict[str, Any]] = []
    total = len(files)
    for i, audio_path in enumerate(files, start=1):
        if progress_cb:
            progress_cb(i, total, audio_path.name)
        entry = annotate_file(audio_path, tier, clap_tagger, label_sets)
        results.append(entry)
    return results


def unload_clap():
    """Free CLAP weights from VRAM. Call before training starts."""
    global _clap_tagger_singleton
    with _clap_tagger_lock:
        if _clap_tagger_singleton is not None:
            _clap_tagger_singleton._model = None
            _clap_tagger_singleton._label_embeds = {}
            _clap_tagger_singleton = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
