"""Pre-Studio: fetch the three frozen V-JEPA 2 encoder weight sets into the HF cache.

Runs the real downloads via huggingface_hub.snapshot_download so the Studio never
waits on a download. The encoder is frozen, so a weight pulled now is the exact
substrate the Studio will use. No model is instantiated here; only files are fetched.
No model is instantiated here. Local 64-frame CPU forwards for ViT-L, ViT-H, and ViT-g are
verified separately by the encoder-scale probe receipts; this helper remains download-only.
"""

import sys
import time

from huggingface_hub import snapshot_download

REPOS = [
    "facebook/vjepa2-vitl-fpc64-256",
    "facebook/vjepa2-vith-fpc64-256",
    "facebook/vjepa2-vitg-fpc64-384",
]


def main() -> int:
    rc = 0
    for repo in REPOS:
        start = time.time()
        print(f"[download] start {repo}", flush=True)
        try:
            path = snapshot_download(repo)
            dt = time.time() - start
            print(f"[download] done  {repo} -> {path} ({dt:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            rc = 1
            print(f"[download] FAIL  {repo}: {exc!r}", flush=True)
    print(f"[download] all-finished rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
