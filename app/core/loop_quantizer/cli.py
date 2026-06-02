"""Thin CLI wrapper for manual testing.

Usage::

    python -m app.core.loop_quantizer \\
        --bpm 120 --bars 4 --grid 16 input.wav output.wav

Reads a WAV via ``soundfile``, runs ``quantize_to_loop`` with the given
parameters, writes the result at the same sample rate. Intended for
ad-hoc experiments during module development — production paths call the
library functions directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .quantizer import quantize_to_loop


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="loop_quantizer", description=__doc__)
    p.add_argument("input_path", type=Path, help="source WAV")
    p.add_argument("output_path", type=Path, help="destination WAV")
    p.add_argument("--bpm", required=True, type=float)
    p.add_argument("--bars", required=True, type=int)
    p.add_argument("--grid", default=16, type=int, choices=(8, 16))
    p.add_argument(
        "--time-sig",
        default="4/4",
        type=str,
        help="time signature like 4/4 or 3/4 (default: 4/4)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    num, denom = (int(x) for x in args.time_sig.split("/", 1))

    audio, sr = sf.read(str(args.input_path), always_2d=True)
    audio = audio.astype(np.float32, copy=False)

    out = quantize_to_loop(
        audio,
        bpm=args.bpm,
        bars=args.bars,
        grid=args.grid,
        time_sig=(num, denom),
        sample_rate=sr,
    )
    sf.write(str(args.output_path), out, sr, subtype="PCM_16")
    print(
        f"quantized {args.input_path} -> {args.output_path} "
        f"({out.shape[0]} samples @ {sr} Hz, bpm={args.bpm}, bars={args.bars}, grid={args.grid})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
