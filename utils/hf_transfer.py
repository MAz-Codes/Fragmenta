"""Control over which transfer huggingface_hub uses for a download.

hf-hub 1.x routes downloads through the Xet backend (the `hf_xet` Rust
extension) whenever the repo is Xet-backed and the package is installed. It is
faster, but it is not a drop-in substitute for the classic HTTP transfer:

* It resolves its own auth against `/api/models/<repo>/xet-read-token/<sha>`,
  which — unlike the regular API — does NOT follow the 307 redirect from a
  legacy alias repo id to its canonical org-prefixed name. `roberta-base`
  404s there; `FacebookAI/roberta-base` does not.
* It reconstructs a blob from chunks rather than writing it front-to-back,
  which some filesystems refuse.

Callers that hit either case can force the classic path for a block.
"""
import contextlib
import os
from pathlib import Path


def xet_enabled() -> bool:
    """Whether hf-hub would route a download through the Xet transfer."""
    from huggingface_hub.utils import _runtime
    return _runtime.is_xet_available()


@contextlib.contextmanager
def xet_disabled():
    """Force hf-hub onto the plain HTTP transfer for the duration of the block.

    `huggingface_hub.constants` snapshots HF_HUB_DISABLE_XET at import time, so
    setting the env var alone is too late once we're running — the constant has
    to be patched too. The env var is still worth setting: any subprocess
    spawned mid-download inherits it.

    Not thread-safe, by nature: it toggles process-wide state. Downloads are
    single-flight (see `ModelManager.start_download`), which is what makes it
    safe to use there.
    """
    from huggingface_hub import constants as _hf_constants
    prev_const = _hf_constants.HF_HUB_DISABLE_XET
    prev_env = os.environ.get("HF_HUB_DISABLE_XET")
    _hf_constants.HF_HUB_DISABLE_XET = True
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        yield
    finally:
        _hf_constants.HF_HUB_DISABLE_XET = prev_const
        if prev_env is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = prev_env


def purge_incomplete(root: Path) -> None:
    """Delete half-written blobs so a retry can't resume onto them.

    hf-hub resumes a download by appending to `<blob>.incomplete`. That is only
    sound when the partial file was itself written front-to-back, which a Xet
    transfer does not guarantee.
    """
    if not root.is_dir():
        return
    for stale in root.rglob("*.incomplete"):
        try:
            stale.unlink()
        except OSError:
            pass
