"""Generation feature gates.

Lives in its own module so the backend can read a flag without importing
the module that implements the feature: /api/environment is hit on every
app load, and importing `audio_post_process` for one boolean dragged
librosa (and numba) into the process.
"""
import os


def beatsync_v2_enabled() -> bool:
    """Feature gate for the hardened beat-sync pipeline (sample-exact loop
    length, anchor-to-tracked-beat alignment — Stage A in
    audio_post_process.py — plus the frontend's 44.1 kHz AudioContext pin).

    Off by default: generation delivers raw SA3 output and the frontend
    uses the device's native sample rate (the pin collapses multi-channel
    output to stereo on Chromium — see performanceAudio.js).
    Enable with ``FRAGMENTA_BEATSYNC_V2=1``.
    """
    return os.environ.get("FRAGMENTA_BEATSYNC_V2", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
