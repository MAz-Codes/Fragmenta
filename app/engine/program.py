"""A program is the piece: an ordered list of steps, each a scene played for a
length of time with its own transition, looped or not. Programs are data in
<data>/engine/programs/<id>.yaml so a piece is edited, not coded
(perpetual/decisions/11). The conductor plays one program at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BED_SOURCES = ("generated", "pool", "both")


@dataclass
class ProgramStep:
    scene: str
    minutes: float = 10.0
    transition_s: Optional[float] = None       # None → the engine's /scene/transition_s
    from_azimuth: Optional[float] = None       # None → /scene/from_azimuth
    beds: str = "generated"                    # generated | pool | both (pool playback arrives with E4.6)
    params: Dict[str, Any] = field(default_factory=dict)   # parameter overrides for this step
    label: str = ""

    @property
    def seconds(self) -> float:
        return max(1.0, float(self.minutes) * 60.0)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"scene": self.scene, "minutes": self.minutes, "beds": self.beds}
        if self.transition_s is not None:
            d["transition_s"] = self.transition_s
        if self.from_azimuth is not None:
            d["from_azimuth"] = self.from_azimuth
        if self.params:
            d["params"] = dict(self.params)
        if self.label:
            d["label"] = self.label
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProgramStep":
        beds = str(d.get("beds", "generated")).lower()
        if beds not in BED_SOURCES:
            raise ValueError(f"beds must be one of {BED_SOURCES}, not {beds!r}")
        t = d.get("transition_s")
        az = d.get("from_azimuth")
        return cls(scene=str(d["scene"]), minutes=float(d.get("minutes", 10.0)),
                   transition_s=None if t is None else float(t), from_azimuth=None if az is None else float(az),
                   beds=beds, params=dict(d.get("params") or {}), label=str(d.get("label", "")))


@dataclass
class Program:
    id: str
    label: str = ""
    scenes: str = ""                 # scene file the steps refer to ('' = whatever is loaded)
    loop: bool = True
    steps: List[ProgramStep] = field(default_factory=list)
    notes: str = ""

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label or self.id, "scenes": self.scenes, "loop": self.loop,
                "steps": [s.to_dict() for s in self.steps], "notes": self.notes, "total_s": self.total_seconds}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Program":
        if not d.get("id"):
            raise ValueError("a program needs an id")
        steps = [ProgramStep.from_dict(s) for s in (d.get("steps") or [])]
        return cls(id=safe_id(str(d["id"])), label=str(d.get("label", "")), scenes=str(d.get("scenes", "") or ""),
                   loop=bool(d.get("loop", True)), steps=steps, notes=str(d.get("notes", "")))

    def validate(self, scene_ids: List[str]) -> List[str]:
        """Problems a user should see before the program plays."""
        out = []
        if not self.steps:
            out.append("no steps")
        for i, s in enumerate(self.steps):
            if s.scene not in scene_ids:
                out.append(f"step {i + 1}: scene '{s.scene}' is not in the loaded scene file")
            if s.minutes <= 0:
                out.append(f"step {i + 1}: minutes must be positive")
        return out


def safe_id(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "_-" else "_" for c in s.strip().lower())
    return out.strip("_") or "program"


def load_program(path: Path) -> Program:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.setdefault("id", Path(path).stem)
    return Program.from_dict(data)


def save_program(program: Program, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{program.id}.yaml"
    d = program.to_dict()
    d.pop("total_s", None)
    out.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def list_programs(*folders: Path) -> List[Dict[str, Any]]:
    """Every program file across the folders; the first folder wins on a name clash (user over shipped)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for folder in folders:
        if not folder or not Path(folder).is_dir():
            continue
        for f in sorted(Path(folder).glob("*.yaml")):
            if f.stem in seen:
                continue
            try:
                p = load_program(f)
                seen[f.stem] = {"id": p.id, "label": p.label or p.id, "steps": len(p.steps), "total_s": p.total_seconds,
                                "loop": p.loop, "scenes": p.scenes, "file": str(f)}
            except (OSError, ValueError, yaml.YAMLError, KeyError) as exc:
                seen[f.stem] = {"id": f.stem, "label": f.stem, "error": str(exc), "file": str(f)}
    return list(seen.values())
