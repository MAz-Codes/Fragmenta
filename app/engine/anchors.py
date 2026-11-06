"""Anchors: cached text-conditioner outputs the engine blends between.

The template is fixed in order and wording (perpetual/decisions/03): the
`TrackType: SFX` prefix, then `Type`, `Place`, `Perspective`, `Density`,
`Time`, `Setting`, `Sources` (multi-value, canonical order), `Mood`. A
segment whose value is missing is dropped, like the Workbench's templates.

An anchor is a centroid: its base fields plus variants are rendered to
prompts, encoded, and averaged. The cache key hashes the encoder's identity
(model + LoRA stack), and the anchor definition, so a changed checkpoint
re-encodes by itself. Blending is a weighted mean of the [T, D] matrices
with the union of masks and optional norm rescaling (decisions/02).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

PREFIX = "TrackType: SFX"
# (field, identifier) in the fixed order. `sources` is multi-value and last of
# the sound fields so it never shifts the others' token positions.
FIELDS: Tuple[Tuple[str, str], ...] = (
    ("type", "Type"), ("place", "Place"), ("perspective", "Perspective"),
    ("density", "Density"), ("time", "Time"), ("setting", "Setting"),
    ("sources", "Sources"), ("mood", "Mood"),
)
VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "type": ("ambience", "event"),
    "place": ("bazaar", "street", "intersection", "shop", "cafe", "park", "metro"),
    "perspective": ("close", "mid", "far"),
    "density": ("sparse", "moderate", "dense"),
    "time": ("morning", "midday", "evening", "night"),
    "setting": ("interior", "exterior"),
    "sources": ("voices", "footsteps", "scooters", "traffic", "vendors", "music",
                "machinery", "birds", "water", "doors", "bells", "horns", "wind", "rain"),
}
_ASCII = re.compile(r"[^a-z0-9 ]")


def normalise_value(v: Any) -> str:
    """Lowercase ASCII, single spaces: one spelling in training and at playback."""
    s = str(v).strip().lower()
    s = (s.replace("é", "e").replace("è", "e").replace("ê", "e").replace("á", "a")
          .replace("à", "a").replace("ö", "o").replace("ü", "u").replace("ç", "c"))
    s = _ASCII.sub("", s)
    return " ".join(s.split())


def canonical_sources(values: Iterable[str]) -> List[str]:
    return sorted({normalise_value(v) for v in values if normalise_value(v)})


def render_prompt(fields: Dict[str, Any]) -> str:
    """The fixed template; empty fields drop their segment."""
    parts = [PREFIX]
    for key, ident in FIELDS:
        v = fields.get(key)
        if v is None or v == "" or v == []:
            continue
        if key == "sources":
            vals = canonical_sources(v if isinstance(v, (list, tuple)) else str(v).split(","))
            if not vals:
                continue
            parts.append(f"{ident}: " + ", ".join(vals))
        else:
            vals = v if isinstance(v, (list, tuple)) else [v]
            vals = [normalise_value(x) for x in vals if normalise_value(x)]
            if not vals:
                continue
            parts.append(f"{ident}: " + ", ".join(vals))
    return ", ".join(parts)


@dataclass
class AnchorDef:
    id: str
    label: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)
    variants: List[Dict[str, Any]] = field(default_factory=list)
    # The blend pad places anchors on a plane; None = auto (regular polygon).
    position: Optional[Tuple[float, float]] = None
    color: str = ""

    def prompts(self) -> List[str]:
        out = [render_prompt(self.fields)]
        for v in self.variants:
            out.append(render_prompt({**self.fields, **v}))
        # de-duplicate, keep order
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def digest(self) -> str:
        return hashlib.sha256(json.dumps({"p": self.prompts()}, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label or self.id, "fields": self.fields,
                "variants": self.variants, "prompt": render_prompt(self.fields),
                "n_prompts": len(self.prompts()), "position": self.position, "color": self.color}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnchorDef":
        pos = d.get("position")
        return cls(id=str(d["id"]), label=d.get("label", ""), fields=dict(d.get("fields", {})),
                   variants=[dict(v) for v in d.get("variants", [])],
                   position=tuple(pos) if pos else None, color=d.get("color", ""))


class Encoder(Protocol):
    key: str

    def encode(self, prompts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """→ (embeddings [N, T, D] float32, mask [N, T] bool)."""


class FakeEncoder:
    """Deterministic stand-in: a prompt's words map to a fixed random matrix."""

    def __init__(self, tokens: int = 32, dim: int = 16, key: str = "fake") -> None:
        self.tokens, self.dim, self.key = tokens, dim, key
        self.calls = 0

    def encode(self, prompts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        embs = np.zeros((len(prompts), self.tokens, self.dim), dtype=np.float32)
        masks = np.zeros((len(prompts), self.tokens), dtype=bool)
        for i, p in enumerate(prompts):
            words = p.replace(",", " ").split()
            n = min(self.tokens, len(words))
            for t, w in enumerate(words[:n]):
                rng = np.random.default_rng(int(hashlib.sha256(w.encode()).hexdigest()[:8], 16))
                embs[i, t] = rng.normal(0, 1, self.dim)
            masks[i, :n] = True
        return embs, masks


@dataclass
class Anchor:
    definition: AnchorDef
    embedding: Optional[np.ndarray] = None    # [T, D]
    mask: Optional[np.ndarray] = None         # [T]
    norm: float = 0.0

    @property
    def id(self) -> str:
        return self.definition.id

    @property
    def encoded(self) -> bool:
        return self.embedding is not None


class AnchorBank:
    def __init__(self, definitions: Iterable[AnchorDef], cache_dir: Optional[Path] = None) -> None:
        self._anchors: Dict[str, Anchor] = {d.id: Anchor(d) for d in definitions}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_key: str = ""
        self._lock = threading.RLock()

    # --- definitions ---------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path, cache_dir: Optional[Path] = None) -> "AnchorBank":
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        defs = [AnchorDef.from_dict(d) for d in data.get("anchors", [])]
        return cls(defs, cache_dir)

    def ids(self) -> List[str]:
        return list(self._anchors)

    def get(self, aid: str) -> Anchor:
        return self._anchors[aid]

    def __contains__(self, aid: str) -> bool:
        return aid in self._anchors

    def __len__(self) -> int:
        return len(self._anchors)

    def add(self, d: AnchorDef) -> Anchor:
        with self._lock:
            a = Anchor(d)
            self._anchors[d.id] = a
            return a

    def remove(self, aid: str) -> bool:
        with self._lock:
            return self._anchors.pop(aid, None) is not None

    def describe(self) -> List[Dict[str, Any]]:
        n = len(self._anchors)
        out = []
        for i, a in enumerate(self._anchors.values()):
            d = a.definition.to_dict()
            if d["position"] is None:
                ang = 2 * np.pi * i / max(1, n) - np.pi / 2
                d["position"] = [round(0.5 + 0.42 * float(np.cos(ang)), 4), round(0.5 + 0.42 * float(np.sin(ang)), 4)]
            d["encoded"] = a.encoded
            d["norm"] = round(a.norm, 3)
            out.append(d)
        return out

    # --- encoding ------------------------------------------------------------
    def _cache_path(self, a: Anchor, encoder_key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(f"{encoder_key}|{a.definition.digest()}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{a.id}_{h}.npz"

    def encode_all(self, encoder: Encoder, force: bool = False) -> Dict[str, bool]:
        """Encode every anchor (cache hit → loaded, else encoded and saved).

        Returns {anchor id: True if it had to be encoded}.
        """
        result: Dict[str, bool] = {}
        self.encoder_key = encoder.key
        for a in list(self._anchors.values()):
            cp = self._cache_path(a, encoder.key)
            if not force and cp and cp.exists():
                try:
                    z = np.load(cp)
                    self._set(a, z["embedding"], z["mask"])
                    result[a.id] = False
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            embs, masks = encoder.encode(a.definition.prompts())
            # centroid: mean over variants; mask = any variant used the token
            self._set(a, embs.mean(axis=0), masks.any(axis=0))
            if cp:
                try:
                    np.savez(cp, embedding=a.embedding, mask=a.mask)
                except OSError:
                    pass
            result[a.id] = True
        return result

    def _set(self, a: Anchor, emb: np.ndarray, mask: np.ndarray) -> None:
        with self._lock:
            a.embedding = np.asarray(emb, dtype=np.float32)
            a.mask = np.asarray(mask, dtype=bool)
            a.norm = float(np.linalg.norm(a.embedding))

    # --- blending ------------------------------------------------------------
    @staticmethod
    def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
        w = {k: max(0.0, float(v)) for k, v in (weights or {}).items()}
        s = sum(w.values())
        if s <= 0:
            return {}
        return {k: v / s for k, v in w.items() if v > 0}

    def blend(self, weights: Dict[str, float], rescale: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Weighted mean of anchors → (embedding [T, D], mask [T], weights used)."""
        w = self.normalise_weights({k: v for k, v in weights.items() if k in self._anchors and self._anchors[k].encoded})
        if not w:
            raise ValueError("no encoded anchors in the blend")
        with self._lock:
            emb = None
            mask = None
            target_norm = 0.0
            for aid, wt in w.items():
                a = self._anchors[aid]
                emb = a.embedding * wt if emb is None else emb + a.embedding * wt
                mask = a.mask.copy() if mask is None else (mask | a.mask)
                target_norm += a.norm * wt
            if rescale and len(w) > 1:
                n = float(np.linalg.norm(emb))
                if n > 1e-8:
                    emb = emb * (target_norm / n)
        return emb.astype(np.float32), mask, w

    def prompt_for(self, weights: Dict[str, float]) -> str:
        """A text stand-in for a blend: the dominant anchor's prompt."""
        w = self.normalise_weights(weights)
        if not w:
            return PREFIX
        top = max(w, key=w.get)
        return render_prompt(self._anchors[top].definition.fields)
