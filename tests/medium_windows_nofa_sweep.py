"""Path-B validation: run sa3-medium WITHOUT Flash Attention 2 and sweep
duration to find the correctness/VRAM ceiling of the PyTorch-native attention
fallback (flex_attention -> chunked-halo SDPA -> masked SDPA).

This is a MANUAL harness — run it on a Windows + NVIDIA (Ampere+) box where
flash_attn is NOT installed. It sets FRAGMENTA_MEDIUM_NO_FLASH=1 so the medium
gate in audio_generator falls through to the native backends.

What it measures, per duration:
  * success / failure (OOM is reported, not fatal — the sweep continues)
  * wall-clock seconds
  * peak CUDA memory (allocated + reserved)
  * output WAV path

Usage (from the repo root, inside the venv):
    python tests/medium_windows_nofa_sweep.py
    python tests/medium_windows_nofa_sweep.py --durations 30,60,120,180,240,300,380
    python tests/medium_windows_nofa_sweep.py --model-id sa3-medium --steps 8 --seed 42

The summary table at the end is what we want for the medium-on-Windows
decision. If it OOMs past some duration N, we ship Windows-medium capped at the
largest passing duration; if it passes to 380s, the fallback alone is enough and
no flash-attn wheel is needed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force the Path-B fallback BEFORE importing the generator / torch-heavy code.
os.environ.setdefault("FRAGMENTA_MEDIUM_NO_FLASH", "1")

# Repo root on sys.path so `app.*` / `utils.*` import. audio_generator adds the
# vendored SA3 path itself on import, so we don't touch vendor/ here.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from app.core.config import get_config  # noqa: E402
from app.core.generation.audio_generator import AudioGenerator  # noqa: E402


def _gib(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def _print_env() -> None:
    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  python              : {sys.version.split()[0]}")
    print(f"  platform            : {sys.platform}")
    print(f"  torch               : {torch.__version__}")
    print(f"  cuda available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        print(f"  cuda device         : {torch.cuda.get_device_name(idx)}")
        print(f"  compute capability  : {cap[0]}.{cap[1]}  "
              f"({'Ampere+' if cap[0] >= 8 else 'PRE-Ampere — flash-attn 2 unsupported'})")
        print(f"  total VRAM          : {_gib(torch.cuda.get_device_properties(idx).total_memory):.2f} GiB")
        print(f"  torch cuda build    : {torch.version.cuda}")
    try:
        import flash_attn  # noqa: F401
        print(f"  flash_attn          : PRESENT ({flash_attn.__version__}) "
              "— NOTE: this run is meant to test WITHOUT it; uninstall to "
              "exercise the fallback")
    except ImportError:
        print("  flash_attn          : absent (good — testing native fallback)")
    print(f"  FRAGMENTA_MEDIUM_NO_FLASH = {os.environ.get('FRAGMENTA_MEDIUM_NO_FLASH')}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", default="30,60,120,180,240,300,380",
                        help="comma-separated seconds to sweep")
    parser.add_argument("--model-id", default="sa3-medium",
                        choices=["sa3-medium", "sa3-medium-base"])
    parser.add_argument("--prompt", default="a warm analog synthwave groove, "
                        "punchy drums, deep bassline, 110 bpm")
    parser.add_argument("--steps", type=int, default=None,
                        help="override sampler steps (default: model default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=str(_REPO_ROOT / "output" / "nofa_sweep"))
    args = parser.parse_args()

    durations = [float(d) for d in args.durations.split(",") if d.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _print_env()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Medium needs a CUDA GPU. Aborting.")
        return 2

    config = get_config()
    generator = AudioGenerator(config)

    print("=" * 72)
    print(f"SWEEP: {args.model_id}  steps={args.steps or 'default'}  "
          f"seed={args.seed}")
    print(f"prompt: {args.prompt!r}")
    print("=" * 72)

    results = []
    for dur in durations:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {"duration_s": dur, "status": None, "wall_s": None,
               "peak_alloc_gib": None, "peak_reserved_gib": None,
               "wav": None, "error": None}
        print(f"\n--- duration = {dur:>6.1f}s ---")
        t0 = time.time()
        try:
            wav = generator.generate_audio(
                args.prompt,
                model_id=args.model_id,
                duration=dur,
                steps=args.steps,
                seed=args.seed,
                batch_size=1,
            )
            row["status"] = "ok"
            row["wall_s"] = round(time.time() - t0, 1)
            row["wav"] = str(wav)
            # Move the artifact into the sweep dir with a descriptive name.
            dest = outdir / f"medium_nofa_{int(dur)}s.wav"
            try:
                Path(wav).replace(dest)
                row["wav"] = str(dest)
            except Exception:
                pass
            print(f"    OK   {row['wall_s']}s -> {row['wav']}")
        except torch.cuda.OutOfMemoryError as exc:
            row["status"] = "oom"
            row["wall_s"] = round(time.time() - t0, 1)
            row["error"] = str(exc).splitlines()[0] if str(exc) else "OOM"
            print(f"    OOM after {row['wall_s']}s")
        except Exception as exc:  # noqa: BLE001 — we want every failure recorded
            row["status"] = "error"
            row["wall_s"] = round(time.time() - t0, 1)
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR {type(exc).__name__}: {exc}")
        finally:
            if torch.cuda.is_available():
                row["peak_alloc_gib"] = round(_gib(torch.cuda.max_memory_allocated()), 2)
                row["peak_reserved_gib"] = round(_gib(torch.cuda.max_memory_reserved()), 2)
                print(f"    peak VRAM: alloc={row['peak_alloc_gib']} GiB  "
                      f"reserved={row['peak_reserved_gib']} GiB")
        results.append(row)

    # --- summary ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'dur(s)':>8} | {'status':>6} | {'wall(s)':>8} | "
          f"{'alloc GiB':>9} | {'reserved GiB':>12} | note")
    print("-" * 72)
    for r in results:
        note = "" if r["status"] == "ok" else (r["error"] or "")
        print(f"{r['duration_s']:>8.1f} | {r['status']:>6} | "
              f"{str(r['wall_s']):>8} | {str(r['peak_alloc_gib']):>9} | "
              f"{str(r['peak_reserved_gib']):>12} | {note[:24]}")

    ok = [r for r in results if r["status"] == "ok"]
    max_ok = max((r["duration_s"] for r in ok), default=None)
    print("-" * 72)
    if max_ok is not None:
        print(f"Largest passing duration: {max_ok:.0f}s "
              f"({'full 380s OK — fallback alone suffices' if max_ok >= 380 else 'cap Windows-medium here, or build a flash-attn wheel for full 380s'})")
    else:
        print("No duration passed — the native fallback is not viable; "
              "Path A (flash-attn wheel) is required.")

    report = outdir / "sweep_report.json"
    report.write_text(json.dumps({
        "model_id": args.model_id,
        "steps": args.steps,
        "seed": args.seed,
        "prompt": args.prompt,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": results,
        "max_ok_duration_s": max_ok,
    }, indent=2))
    print(f"\nReport written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
