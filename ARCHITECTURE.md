# Architecture

MOP keeps a small set of maintained authorities:

1. `mop.science`: experiment records, budgets, statistics, receipts, and sealing.
2. `mop.substrate`: custom-substrate execution, data contracts, caches, and canonical evidence primitives.
3. `mop.experiments`: compact CM7/CM8 declarations over the shared engine.
4. `mop.beds.starss23`: the retained real-data scientific bed and independent referee path.
5. Runtime registry/configuration trees: frozen identities and capability state.
6. Host utilities: profiles, intake, download, encoding, diagnostics, and the narrow operations CLI.

Unique scientific mathematics stays local; repeated orchestration belongs to shared authorities.
Independent verifiers do not import graded producer mathematics. Historical code is recovered by tag
through `collapse/MOP_HISTORICAL_CODE_INDEX.json`, not maintained beside current implementations.
