from __future__ import annotations

import json
import sys

from ..config import compose
from .runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = compose(argv)
    metrics = run_experiment(cfg)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
