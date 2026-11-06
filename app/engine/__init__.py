"""Fragmenta Perpetual Engine: a headless, perpetual, parametric generator.

Render ahead on the GPU, play in real time, steer over HTTP, OSC and MIDI.
The Fragmenta UI's Engine tab is one client of the API in `api.py`; nothing
in this package imports the Flask backend (`app/backend`).

Run headless:  python -m app.engine --scene places --layout stereo
Plan and decisions: redesign/PERPETUAL_ENGINE_PLAN.md, perpetual/decisions/.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
