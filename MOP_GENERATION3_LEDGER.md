# MOP Generation 3 Discovery Ledger

Program `mop-generation3-discovery-v1` on `agent/mop-generation3-discovery` off `89eeca5`.

Two concurrent streams: C1 (P1R sampling-priority, measured headroom +0.169) and C3 (model-error-aware simulation, precompute-gated).

SESOI 0.05. A tie is a null. No positive without independent adversarial verification. Activation false.

## C1 stages: A calibration -> B canary -> C confirmation -> D replication (staged, sealed continuation).

## C3: 8 precompute gates -> bounded canary only if all pass.

Resume: read MOP_GENERATION3_STATE.json.
