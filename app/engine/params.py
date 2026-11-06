"""The parameter model: everything is a parameter, every parameter can be modulated.

A parameter has a base value (what HTTP, OSC or MIDI set), a range, a response
curve for normalised control input, a smoothing time, and an optional
modulator. `tick(dt)` advances smoothing and modulation at control rate and
produces the *effective* value the conductor and the audio engine read.

Paths are hierarchical strings like `/events/density`. Non-float kinds (int,
bool, str, list) have neither smoothing nor modulation. Thread-safe: one lock
around the tree; listeners are called outside it.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

MODULATOR_TYPES = ("fixed", "walk", "lfo", "distribution", "sequence")
CURVES = ("lin", "exp", "log", "exp3")


def apply_curve(curve: str, x: float) -> float:
    """Normalised control position (0–1) → normalised value (0–1)."""
    x = min(1.0, max(0.0, float(x)))
    if curve == "exp":
        return x * x
    if curve == "exp3":
        return x * x * x
    if curve == "log":
        return math.sqrt(x)
    return x


def invert_curve(curve: str, y: float) -> float:
    y = min(1.0, max(0.0, float(y)))
    if curve == "exp":
        return math.sqrt(y)
    if curve == "exp3":
        return y ** (1.0 / 3.0)
    if curve == "log":
        return y * y
    return y


@dataclass
class Modulator:
    """A source of slow change, in normalised units (fractions of the range).

    type: fixed (no change) · walk (bounded random walk) · lfo (sine) ·
          distribution (resample every 1/rate s) · sequence (step through values)
    rate: per-second speed (walk), Hz (lfo, distribution, sequence)
    depth: maximum excursion as a fraction of the range (walk, lfo, distribution)
    seed: makes a walk or a distribution reproducible
    values: the sequence's values, in real units
    dist: "uniform" or "normal" for distribution
    """
    type: str = "fixed"
    rate: float = 0.05
    depth: float = 0.2
    seed: int = 0
    values: List[Any] = field(default_factory=list)
    dist: str = "uniform"
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.type not in MODULATOR_TYPES:
            raise ValueError(f"unknown modulator type {self.type!r}")
        self._rng = np.random.default_rng(int(self.seed))
        self._t = 0.0
        self._x = 0.0          # walk position / last distribution sample
        self._next = 0.0       # next resample time (distribution, sequence)
        self._i = 0            # sequence index

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": self.type, "rate": self.rate, "depth": self.depth, "seed": self.seed}
        if self.type == "sequence":
            d["values"] = list(self.values)
        if self.type == "distribution":
            d["dist"] = self.dist
        if self.type == "lfo":
            d["phase"] = self.phase
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Modulator":
        if not d:
            return cls()
        allowed = {k: d[k] for k in ("type", "rate", "depth", "seed", "values", "dist", "phase") if k in d}
        return cls(**allowed)

    def step(self, dt: float) -> Tuple[Optional[float], Optional[Any]]:
        """Advance by dt seconds. Returns (normalised offset, replacement value).

        A walk / lfo / distribution returns an offset in [-depth, depth]; a
        sequence returns the value that replaces the base; fixed returns (0, None).
        """
        self._t += dt
        if self.type == "fixed":
            return 0.0, None
        if self.type == "lfo":
            return self.depth * math.sin(2 * math.pi * self.rate * self._t + self.phase), None
        if self.type == "walk":
            # rate: how fast, depth: how far. Reflect at the bounds so the walk
            # stays inside ±depth instead of pinning to an edge.
            self._x += float(self._rng.normal(0.0, 1.0)) * self.depth * self.rate * dt * 4.0
            if self._x > self.depth:
                self._x = 2 * self.depth - self._x
            elif self._x < -self.depth:
                self._x = -2 * self.depth - self._x
            self._x = min(self.depth, max(-self.depth, self._x))
            return self._x, None
        if self.type == "distribution":
            if self._t >= self._next:
                self._next = self._t + (1.0 / self.rate if self.rate > 0 else math.inf)
                if self.dist == "normal":
                    self._x = float(np.clip(self._rng.normal(0.0, self.depth / 2.0), -self.depth, self.depth))
                else:
                    self._x = float(self._rng.uniform(-self.depth, self.depth))
            return self._x, None
        if self.type == "sequence":
            if not self.values:
                return 0.0, None
            if self._next == 0.0:
                # first step: start on the first value, schedule the next
                self._next = self._t + (1.0 / self.rate if self.rate > 0 else math.inf)
            elif self._t >= self._next:
                self._i = (self._i + 1) % len(self.values)
                self._next = self._t + (1.0 / self.rate if self.rate > 0 else math.inf)
            return None, self.values[self._i % len(self.values)]
        return 0.0, None


@dataclass
class Parameter:
    id: str
    value: Any
    range: Optional[Tuple[float, float]] = None
    curve: str = "lin"
    smoothing_s: float = 0.0
    modulator: Modulator = field(default_factory=Modulator)
    kind: str = "float"                     # float | int | bool | str | list
    label: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.curve not in CURVES:
            raise ValueError(f"unknown curve {self.curve!r}")
        if self.kind == "float" and self.range is None:
            raise ValueError(f"{self.id}: a float parameter needs a range")
        self.value = self.coerce(self.value)
        self._smoothed = float(self.value) if self.kind == "float" else None
        self.current: Any = self.value

    # --- values --------------------------------------------------------------
    def coerce(self, v: Any) -> Any:
        if self.kind == "float":
            lo, hi = self.range
            return float(min(hi, max(lo, float(v))))
        if self.kind == "int":
            v = int(round(float(v)))
            if self.range is not None:
                lo, hi = self.range
                v = int(min(hi, max(lo, v)))
            return v
        if self.kind == "bool":
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "on", "yes")
            return bool(v)
        if self.kind == "list":
            return list(v) if isinstance(v, (list, tuple)) else [v]
        return str(v)

    def set(self, v: Any) -> None:
        self.value = self.coerce(v)
        if self.kind != "float":
            self.current = self.value

    def set_normalized(self, x: float) -> None:
        """Set from a 0–1 control (MIDI, a knob) through the response curve."""
        if self.kind not in ("float", "int") or self.range is None:
            raise ValueError(f"{self.id}: not a ranged parameter")
        lo, hi = self.range
        self.set(lo + (hi - lo) * apply_curve(self.curve, x))

    def normalized(self) -> Optional[float]:
        if self.kind not in ("float", "int") or self.range is None:
            return None
        lo, hi = self.range
        if hi == lo:
            return 0.0
        return invert_curve(self.curve, (float(self.value) - lo) / (hi - lo))

    def tick(self, dt: float) -> Any:
        if self.kind != "float":
            off, repl = self.modulator.step(dt)
            self.current = self.coerce(repl) if repl is not None else self.value
            return self.current
        lo, hi = self.range
        target = float(self.value)
        if self.smoothing_s > 0 and dt > 0:
            a = 1.0 - math.exp(-dt / self.smoothing_s)
            self._smoothed += (target - self._smoothed) * a
            if abs(target - self._smoothed) < 1e-6 * max(1.0, abs(hi - lo)):
                self._smoothed = target
        else:
            self._smoothed = target
        off, repl = self.modulator.step(dt)
        if repl is not None:
            self.current = self.coerce(repl)
        else:
            self.current = float(min(hi, max(lo, self._smoothed + (off or 0.0) * (hi - lo))))
        return self.current

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "value": self.value, "current": self.current,
            "range": list(self.range) if self.range is not None else None,
            "curve": self.curve, "smoothing_s": self.smoothing_s,
            "modulator": self.modulator.to_dict(), "kind": self.kind,
            "label": self.label, "unit": self.unit,
        }


Listener = Callable[[str, Any], None]


class ParameterStore:
    """The tree shared by HTTP, OSC and MIDI. One lock; listeners outside it."""

    def __init__(self) -> None:
        self._params: Dict[str, Parameter] = {}
        self._lock = threading.RLock()
        self._listeners: List[Listener] = []

    # --- definition ----------------------------------------------------------
    @staticmethod
    def norm_path(path: str) -> str:
        p = "/" + "/".join(s for s in str(path).strip().split("/") if s)
        return p

    def define(self, path: str, value: Any, range: Optional[Tuple[float, float]] = None,
               curve: str = "lin", smoothing_s: float = 0.0, modulator: Optional[Dict[str, Any]] = None,
               kind: Optional[str] = None, label: str = "", unit: str = "") -> Parameter:
        path = self.norm_path(path)
        if kind is None:
            if isinstance(value, bool):
                kind = "bool"
            elif isinstance(value, int):
                kind = "int"
            elif isinstance(value, float):
                kind = "float"
            elif isinstance(value, (list, tuple)):
                kind = "list"
            else:
                kind = "str"
        p = Parameter(id=path, value=value, range=tuple(range) if range is not None else None,
                      curve=curve, smoothing_s=smoothing_s, modulator=Modulator.from_dict(modulator),
                      kind=kind, label=label, unit=unit)
        with self._lock:
            self._params[path] = p
        return p

    def undefine(self, path: str) -> None:
        with self._lock:
            self._params.pop(self.norm_path(path), None)

    def exists(self, path: str) -> bool:
        return self.norm_path(path) in self._params

    def paths(self, prefix: str = "/") -> List[str]:
        prefix = self.norm_path(prefix)
        with self._lock:
            if prefix == "/":
                return sorted(self._params)
            return sorted(p for p in self._params if p == prefix or p.startswith(prefix + "/"))

    # --- access --------------------------------------------------------------
    def param(self, path: str) -> Parameter:
        path = self.norm_path(path)
        with self._lock:
            if path not in self._params:
                raise KeyError(path)
            return self._params[path]

    def get(self, path: str) -> Any:
        """The effective (smoothed, modulated) value."""
        return self.param(path).current

    def base(self, path: str) -> Any:
        return self.param(path).value

    def set(self, path: str, value: Any, notify: bool = True) -> Any:
        p = self.param(path)
        with self._lock:
            p.set(value)
            v = p.value
        if notify:
            self._emit(p.id, v)
        return v

    def set_normalized(self, path: str, x: float) -> Any:
        p = self.param(path)
        with self._lock:
            p.set_normalized(x)
            v = p.value
        self._emit(p.id, v)
        return v

    def set_modulator(self, path: str, mod: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = self.param(path)
        with self._lock:
            p.modulator = Modulator.from_dict(mod)
            d = p.modulator.to_dict()
        self._emit(p.id, p.value)
        return d

    def set_many(self, values: Dict[str, Any]) -> None:
        for k, v in values.items():
            if self.exists(k):
                self.set(k, v)

    # --- control rate --------------------------------------------------------
    def tick(self, dt: float) -> None:
        with self._lock:
            for p in self._params.values():
                p.tick(dt)

    # --- listeners -----------------------------------------------------------
    def subscribe(self, cb: Listener) -> Callable[[], None]:
        self._listeners.append(cb)
        return lambda: self._listeners.remove(cb)

    def _emit(self, path: str, value: Any) -> None:
        for cb in list(self._listeners):
            try:
                cb(path, value)
            except Exception:  # a listener must never break a set
                pass

    # --- views ---------------------------------------------------------------
    def describe(self, path: str) -> Dict[str, Any]:
        return self.param(path).to_dict()

    def flat(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: p.to_dict() for k, p in sorted(self._params.items())}

    def tree(self) -> Dict[str, Any]:
        """Nested dict of the namespace; leaves are parameter descriptors."""
        out: Dict[str, Any] = {}
        for path, d in self.flat().items():
            node = out
            parts = path.strip("/").split("/")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    break
            node[parts[-1]] = d
        return out

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: {"value": p.value, "modulator": p.modulator.to_dict()} for k, p in self._params.items()}

    def restore(self, snap: Dict[str, Dict[str, Any]]) -> int:
        n = 0
        for k, d in (snap or {}).items():
            if not self.exists(k):
                continue
            p = self.param(k)
            with self._lock:
                try:
                    p.set(d.get("value", p.value))
                    if "modulator" in d:
                        p.modulator = Modulator.from_dict(d["modulator"])
                    p._smoothed = float(p.value) if p.kind == "float" else None
                    p.current = p.value
                    n += 1
                except (TypeError, ValueError):
                    continue
        return n
