#!/usr/bin/env python3
"""Policy-compatible entry point for Generation-1 evidence synthesis.

The active detached program seals a finite process-marker allowlist.  Keeping
the approved basename lets an injected post-run capsule use that unchanged
resource policy while the authority path and imported producer remain exact.
"""

from mop.studies.generation1_evidence_synthesis import main

if __name__ == "__main__":
    raise SystemExit(main())
