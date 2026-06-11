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

# Text-side dependencies laion_clap pulls from HF on construction.
# We stage these into models/pretrained/clap/hub/ so the rich tier is
# fully offline after a single download and nothing leaks to ~/.cache.
CLAP_TEXT_DEPS = ("roberta-base", "bert-base-uncased", "facebook/bart-base")

# Whole-clip CLAP embedding windowing. laion_clap (enable_fusion=False) only
# looks at max_len=480000 samples (10 s @ 48 kHz) and picks a RANDOM offset
# for longer clips — non-deterministic tags and a single arbitrary window.
# We instead embed non-overlapping 10 s windows and mean-pool, which is
# deterministic (rand_trunc is a no-op on exact-length windows) and hears the
# whole clip. CLAP_MAX_WINDOWS caps compute (12 ≈ 120 s/clip); a trailing
# window shorter than CLAP_MIN_TAIL is dropped so near-silent tails don't
# skew the pooled embedding.
CLAP_SR = 48000
CLAP_WIN = 480000
CLAP_MAX_WINDOWS = 12
CLAP_MIN_TAIL = CLAP_WIN // 2

# Confidence gates: a tag is only emitted when the top-1 cosine similarity
# clears the per-group floor, and (genre/mood) beats the runner-up by
# CLAP_MARGIN — near-ties mean "model can't tell", not a coin-flip. Missing
# keys are intentional: apply_template() drops the segment and
# _compose_prompt() guards each field, same as a missing BPM/key.
# Starting points — calibrate against a hand-labeled set if tags get sparse.
# Measured on the bundled demo clips: real-music top-1 sims run ~0.13 (mood)
# to ~0.22 (genre) with top1-top2 margins of 0.012-0.034, while an SFX-style
# impulse clip sits at 0.04-0.09 across all groups. A margin of 0.02 would
# have dropped genre/mood from real music, hence 0.01. Caveat: stationary
# noise legitimately scores ~0.18 as "experimental", too close to real music
# to gate on tau alone.
CLAP_TAU = {"genre": 0.10, "mood": 0.10, "instruments": 0.12}
CLAP_MARGIN = 0.01

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
    import numpy as np
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa 0.10+ returns tempo as np.ndarray (shape (1,) typically).
        # numpy 2.x removed implicit float() conversion of N-d arrays —
        # `float(np.array([120.]))` now raises TypeError instead of returning
        # 120.0 like numpy 1.x did. Normalize via .flat[0] which handles
        # scalar, 0-d, 1-d, and N-d uniformly.
        arr = np.atleast_1d(np.asarray(tempo))
        if arr.size == 0:
            return None
        bpm = float(arr.flat[0])
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


# Serializes checkpoint loads that need relaxed unpickling. The CLAP loader
# below holds this while it reads the (trusted) .pt; any other model-load site
# that has to allow weights_only=False should acquire the same lock so the
# windows never overlap. It is module-level (process-wide) on purpose.
CHECKPOINT_LOAD_LOCK = threading.Lock()


def _clap_load_state_dict_trusted(checkpoint_path: str, map_location: str = "cpu",
                                  skip_params: bool = True):
    """Vendored copy of laion_clap.clap_module.factory.load_state_dict.

    We replicate it here so weights_only=False is passed to torch.load for THIS
    call only. The previous approach swapped torch.load process-wide while CLAP
    loaded, which silently leaked weights_only=False into any concurrent
    torch.load on another thread (e.g. a generation/LoRA model load) — an unsafe
    unpickling foot-gun. Loading the state dict directly keeps the relaxation
    local to this one trusted checkpoint and touches no global state.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params and state_dict:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        # Newer transformers dropped the roberta position_ids buffer; remove it
        # so the strict-ish load doesn't choke. pop() is a no-op if absent.
        try:
            from packaging import version
            import transformers
            if version.parse(transformers.__version__) >= version.parse("4.31.0"):
                state_dict.pop("text_branch.embeddings.position_ids", None)
        except Exception:
            state_dict.pop("text_branch.embeddings.position_ids", None)
    return state_dict


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
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # Point HF resolution at our project-local cache and disable the
            # HEAD-revalidation traffic. After download_clap_checkpoint() has
            # staged the text deps under <pretrained>/clap/hub/, CLAP_Module
            # loads them offline with zero HF hub requests.
            #
            # Two reasons env vars alone aren't enough:
            # 1. huggingface_hub.constants.HF_HUB_OFFLINE is captured at
            #    module-import time (constants.py:185). model_manager.py
            #    imports huggingface_hub at app startup, so the constant is
            #    already False by the time we set the env var here.
            #    transformers.utils.hub.is_offline_mode reads that same
            #    constant — patching the attribute makes both libraries see
            #    offline mode.
            # 2. laion_clap/training/data.py:44-46 runs three from_pretrained
            #    calls at MODULE LEVEL — those fire the first time we do
            #    `import laion_clap` and predate any patch we do after the
            #    import. So we patch BEFORE the import, not after.
            hub_dir = self.ckpt_path.parent / "hub"
            env_keys = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            prev_env = {k: os.environ.get(k) for k in env_keys}
            os.environ["HF_HUB_CACHE"] = str(hub_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            import huggingface_hub.constants as _hhc
            prev_offline_attr = _hhc.HF_HUB_OFFLINE
            prev_cache_attr = _hhc.HF_HUB_CACHE
            _hhc.HF_HUB_OFFLINE = True
            # HF_HUB_CACHE is captured at import time too (same reason as
            # HF_HUB_OFFLINE), so the env vars above don't actually redirect it.
            # Patch the constant directly — otherwise transformers falls back to
            # the default ~/.cache/huggingface, which may hold only the tokenizer
            # (not config.json / weights) and then fails in offline mode.
            _hhc.HF_HUB_CACHE = str(hub_dir)
            try:
                import laion_clap  # noqa: E402 — must follow the offline patch
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
            finally:
                _hhc.HF_HUB_OFFLINE = prev_offline_attr
                _hhc.HF_HUB_CACHE = prev_cache_attr
                for k, v in prev_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            # torch >= 2.6 flipped torch.load(weights_only=True) and newer
            # transformers dropped the roberta position_ids buffer, so
            # laion_clap's own load_ckpt errors twice: unpickling, then strict
            # state_dict mismatch. We load the (trusted) checkpoint via a
            # vendored helper that passes weights_only=False for this call only,
            # under a shared lock — no process-wide torch.load patch, so a
            # concurrent generation load can never inherit unsafe unpickling.
            with CHECKPOINT_LOAD_LOCK:
                state = _clap_load_state_dict_trusted(str(self.ckpt_path), skip_params=True)
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

    def _embed_audio(self, audio_path: Path):
        """Deterministic whole-clip embedding: mean-pool over 10 s windows.

        Each window is exactly CLAP_WIN samples, so laion_clap's rand_trunc
        path never fires (overflow=0) and repeated runs produce identical
        embeddings. If laion_clap is ever run with enable_fusion=True this
        windowing must be revisited.
        """
        import librosa
        import numpy as np
        import torch

        # Decode only what the windowing below can use (CLAP_MAX_WINDOWS x
        # 10 s). Without the cap a 2-hour file decodes fully into RAM
        # (~1.3 GB of float32 at 48 kHz) just to embed its first 2 minutes.
        y, _ = librosa.load(str(audio_path), sr=CLAP_SR, mono=True,
                            duration=CLAP_MAX_WINDOWS * 10)
        if y.size == 0:
            raise ValueError("empty audio")

        # Non-overlapping 10 s windows. Pad a short *first/only* window;
        # drop a short trailing window so near-silent tails don't skew the mean.
        windows = []
        for start in range(0, y.size, CLAP_WIN):
            chunk = y[start:start + CLAP_WIN]
            if chunk.size < CLAP_WIN:
                if windows and chunk.size < CLAP_MIN_TAIL:
                    break  # negligible tail — ignore it
                chunk = np.pad(chunk, (0, CLAP_WIN - chunk.size))
            windows.append(chunk)
            if len(windows) >= CLAP_MAX_WINDOWS:
                break

        batch = torch.from_numpy(np.stack(windows)).float()
        with torch.no_grad():
            embs = self._model.get_audio_embedding_from_data(x=batch, use_tensor=True)
        # Mean-pool across windows, then L2-normalize once.
        emb = embs.mean(dim=0, keepdim=True)
        return emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def tag(self, audio_path: Path, label_sets: Dict[str, List[str]], top_k_instruments: int = 2) -> Dict[str, Any]:
        self.ensure_loaded()
        import torch

        audio_embed = self._embed_audio(audio_path)

        out: Dict[str, Any] = {}
        for group in ("genre", "mood"):
            labels = label_sets.get(group) or []
            if not labels:
                continue
            prompts = [f"a {lab} music track" if group == "genre" else f"a {lab} sounding music track" for lab in labels]
            text_embed = self._embed_labels(group, prompts)
            sims = (audio_embed @ text_embed.T).squeeze(0)

            if sims.numel() >= 2:
                top2 = torch.topk(sims, k=2)
                top_val, top_idx = float(top2.values[0]), int(top2.indices[0])
                margin = top_val - float(top2.values[1])
            else:
                top_val, top_idx, margin = float(sims[0]), 0, float("inf")

            logger.debug(
                "CLAP %s for %s: top=%r sim=%.4f margin=%.4f",
                group, audio_path.name, labels[top_idx], top_val, margin,
            )
            if top_val >= CLAP_TAU[group] and margin >= CLAP_MARGIN:
                out[group] = labels[top_idx]
            # else: omit — the prompt template drops the segment

        instruments = label_sets.get("instruments") or []
        if instruments:
            prompts = [f"music featuring {lab}" for lab in instruments]
            text_embed = self._embed_labels("instruments", prompts)
            sims = (audio_embed @ text_embed.T).squeeze(0)
            k = min(top_k_instruments, len(instruments))
            top = torch.topk(sims, k=k)
            logger.debug(
                "CLAP instruments for %s: %s",
                audio_path.name,
                [(instruments[i], round(v, 4)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
            )
            kept = [instruments[i] for v, i in zip(top.values.tolist(), top.indices.tolist())
                    if v >= CLAP_TAU["instruments"]]
            if kept:  # only set the key if at least one passes
                out["instruments"] = kept
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


def clap_hub_dir(models_pretrained_dir: Path) -> Path:
    """HF cache for laion_clap's text-side deps. Sibling of the .pt."""
    return models_pretrained_dir / "clap" / "hub"


def clap_checkpoint_available(models_pretrained_dir: Path) -> bool:
    return clap_checkpoint_path(models_pretrained_dir).exists()


def _text_dep_snapshot_present(hub_dir: Path, repo_id: str) -> bool:
    safe = "models--" + repo_id.replace("/", "--")
    snap_root = hub_dir / safe / "snapshots"
    if not snap_root.exists():
        return False
    return any(snap_root.iterdir())


def download_clap_checkpoint(
    models_pretrained_dir: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
    phase_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """Download the CLAP audio .pt plus laion_clap's text-side HF snapshots.

    Four sequential phases — emit a phase update (current, total, label) at the
    start of each so a multi-phase progress UI can show real context. Skips
    phases whose artifacts are already on disk.

    `progress_cb` (str-only) is kept for the bulk-annotate API.
    `phase_cb` (current, total, label) is the structured channel.
    """
    target = clap_checkpoint_path(models_pretrained_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    hub_dir = clap_hub_dir(models_pretrained_dir)
    hub_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download, snapshot_download
    import os

    total_phases = 1 + len(CLAP_TEXT_DEPS)

    def _emit(phase_index: int, label: str) -> None:
        if phase_cb:
            phase_cb(phase_index, total_phases, label)
        if progress_cb:
            progress_cb(f"[{phase_index}/{total_phases}] {label}")

    if not target.exists():
        _emit(1, "CLAP audio model (~2.35 GB)")

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

    # laion_clap's CLAP_Module(...) constructor instantiates a Roberta text
    # branch plus bert/bart tokenizers at import time. Pre-stage them into
    # our own cache so the rich tier is fully offline after this step.
    # safetensors only — pytorch_model.bin is a redundant copy.
    for i, repo_id in enumerate(CLAP_TEXT_DEPS, start=2):
        if _text_dep_snapshot_present(hub_dir, repo_id):
            continue
        _emit(i, f"Text encoder: {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hub_dir),
            allow_patterns=[
                "config.json",
                "tokenizer*",
                "vocab*",
                "merges.txt",
                "special_tokens_map.json",
                "model.safetensors",
            ],
        )

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
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    import librosa
    import warnings

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

    # Silent / harmonically flat clips trip librosa's "Trying to estimate
    # tuning from empty frequency set" warning during chroma extraction.
    # The warning is benign — the analysis returns sensible defaults — but
    # it spams stderr on every silent file, so we mute it here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Trying to estimate tuning from empty frequency set",
        )
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

    # Template-driven prompt assembly. Falls back to the legacy descriptive
    # prose if no template is supplied (call sites that haven't been
    # threaded with project metadata yet).
    if prompt_template is not None and prompt_template.strip():
        from app.backend.data.projects import apply_template
        prompt = apply_template(prompt_template, parts)
    else:
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
