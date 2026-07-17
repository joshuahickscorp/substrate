"""Parent marker for net-new Stage 3 experimental beds.

A bed is a self-contained, deterministic task environment plus its sealed referee, controls, matched
budget accounting, statistics, and an independently authored verifier. Beds live here so they never
touch the sealed general-run or chain modules. Importing this package establishes no scientific claim.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

BEDS_NAMESPACE = "mop.beds"
