#!/usr/bin/env python3
"""Build the latest.json release manifest the in-app update notifier reads.

Stdlib only. Reads VERSION and the matching CHANGELOG.md section (first five
bullets become the highlights), hashes any artifact files, and writes
latest.json. Attach it to the GitHub release while it's still a draft, so
releases/latest/download/latest.json exists the moment it's published.

    # 1.2.x (source-only release, Python 3.11)
    python packaging/make_manifest.py --requires-python 3.11

    # 2.0.0 (installers; source installs must reinstall)
    python packaging/make_manifest.py --requires-python 3.14 --reinstall-required \\
        --upgrade-guide-url https://github.com/MAz-Codes/Fragmenta/blob/main/docs/UPGRADING.md \\
        --artifact macos-arm64=dist/Fragmenta-2.0.0-macOS-arm64.dmg ...

Schema: redesign/UPDATE_NOTIFIER.md §2.1.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/MAz-Codes/Fragmenta"
ARTIFACT_KEYS = ("macos-arm64", "windows-x64", "linux-x86_64-appimage", "linux-x86_64-tar")


def changelog_highlights(version: str) -> list:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no '## [{version}]' section")
    bullets = []
    for line in m.group(1).splitlines():
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line.startswith("  ") and line.strip():
            bullets[-1] += " " + line.strip()
    return bullets[:5]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--requires-python", required=True, help="major.minor the release runs on, e.g. 3.14")
    p.add_argument("--reinstall-required", action="store_true", help="source installs can't update in place")
    p.add_argument("--upgrade-guide-url", help="page explaining a major upgrade (must be on the GitHub repo)")
    p.add_argument("--critical", action="store_true", help="security or data-loss fix")
    p.add_argument("--min-supported", help="older versions are told they're unsupported")
    p.add_argument("--artifact", action="append", default=[], metavar="KEY=PATH",
                   help=f"release asset; KEY is one of {', '.join(ARTIFACT_KEYS)}")
    p.add_argument("-o", "--output", default="latest.json")
    args = p.parse_args()

    if not re.fullmatch(r"\d+\.\d+", args.requires_python):
        raise SystemExit("--requires-python must look like 3.14")
    if args.upgrade_guide_url and not args.upgrade_guide_url.startswith(REPO + "/"):
        raise SystemExit("--upgrade-guide-url must be on the Fragmenta GitHub repo")

    version = (ROOT / "VERSION").read_text().strip()
    tag = f"v{version}"
    manifest = {
        "schema": 1,
        "version": version,
        "released_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes_url": f"{REPO}/releases/tag/{tag}",
        "highlights": changelog_highlights(version),
        "critical": args.critical,
        "requires": {"python": args.requires_python},
        "reinstall_required": args.reinstall_required,
        "artifacts": {},
    }
    if args.min_supported:
        manifest["min_supported"] = args.min_supported
    if args.upgrade_guide_url:
        manifest["upgrade_guide_url"] = args.upgrade_guide_url
    for spec in args.artifact:
        key, _, path = spec.partition("=")
        if key not in ARTIFACT_KEYS or not path:
            raise SystemExit(f"bad --artifact {spec!r}")
        f = Path(path)
        manifest["artifacts"][key] = {"url": f"{REPO}/releases/download/{tag}/{f.name}",
                                      "sha256": sha256(f), "size": f.stat().st_size}

    Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {version}, {len(manifest['highlights'])} highlights, "
          f"{len(manifest['artifacts'])} artifacts, requires Python {args.requires_python}")


if __name__ == "__main__":
    main()
