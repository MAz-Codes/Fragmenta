#!/usr/bin/env python3
"""Check the signed AI-output marking (C2PA Content Credentials) in a file.

Needs only the c2pa library, not the rest of Fragmenta:
    pip install c2pa-python==0.38.0
    python verify_credentials.py song.wav [more.wav ...]

Exit code: 0 all intact, 1 any file changed since signing, 2 any file unsigned
or unreadable.
"""
import json
import mimetypes
import sys

import c2pa

INTACT, CHANGED, UNSIGNED = 0, 1, 2


def _codes(results: dict, kind: str) -> set:
    return {entry.get("code") for entry in results.get(kind) or []}


def check(path: str) -> tuple:
    """(status, lines to print) for one file."""
    fmt = mimetypes.guess_type(path)[0] or "audio/wav"
    try:
        with open(path, "rb") as f, c2pa.Reader(fmt, f) as reader:
            store = json.loads(reader.json())
    except OSError as exc:
        return UNSIGNED, [f"cannot read: {exc}"]
    except c2pa.C2paError as exc:
        return UNSIGNED, [f"no Content Credentials ({exc})"]

    manifest = store["manifests"][store["active_manifest"]]
    results = store.get("validation_results", {}).get("activeManifest", {})
    failures = _codes(results, "failure")
    # An unknown signer is expected: each install signs with its own key.
    failures.discard("signingCredential.untrusted")

    lines = []
    for assertion in manifest.get("assertions", []):
        if assertion.get("label", "").startswith("c2pa.actions"):
            for action in assertion["data"].get("actions", []):
                if action.get("digitalSourceType"):
                    lines.append(f"source type: {action['digitalSourceType'].rsplit('/', 1)[-1]}")
                if action.get("parameters"):
                    lines.append(f"details: {json.dumps(action['parameters'])}")
    signature = manifest.get("signature_info", {})
    lines.append(f"signer: {signature.get('issuer', 'unknown')}")
    if signature.get("time"):
        lines.append(f"time-stamped: {signature['time']}")

    if failures or store.get("validation_state") == "Invalid":
        return CHANGED, ["CHANGED since it was signed: " + ", ".join(sorted(failures))] + lines
    return INTACT, ["intact: unchanged since it was signed"] + lines


def main(paths: list) -> int:
    if not paths:
        print(__doc__)
        return UNSIGNED
    worst = INTACT
    for path in paths:
        status, lines = check(path)
        print(path)
        for line in lines:
            print(f"  {line}")
        # A changed file outranks an unsigned one in the exit code.
        worst = CHANGED if CHANGED in (worst, status) else max(worst, status)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
