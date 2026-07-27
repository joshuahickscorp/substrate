# Substrate ledger

Generated from the tree at commit `4a6c95086ccf4844ea7e8211a1578f10fc5fc658`. Status is derived, never asserted: an item is
implemented because its files exist, tested because a recorded test ledger says so, measured
because its evidence is sealed, and terminal because a scientific classification exists for it.

Items: 35. Levels: measured 5, not_started 21, terminal 7, tested 2.

| id | section | title | level | dependencies | next action |
|---|---|---|---|---|---|
| A1 | 2 | Naming and historical continuity authority | terminal | none | none, an authority is terminal once sealed and tested |
| A2 | 21 | Master deliverable set exists and binds to real things | terminal | A1 | none, an authority is terminal once sealed and tested |
| A3 | 18 | Experimental requirements bind every new Substrate experiment | terminal | A2 | none, an authority is terminal once sealed and tested |
| A4 | 19 | Developmental safety envelope | terminal | A2 | none, an authority is terminal once sealed and tested |
| A5 | 16 | Sentience research boundary | terminal | A2 | none, an authority is terminal once sealed and tested |
| A6 | 17 | Continuous six batch research program | terminal | A2 | none, an authority is terminal once sealed and tested |
| A7 | 20 | Scorecard separates implementation from evidence | terminal | A2 | none, an authority is terminal once sealed and tested |
| C1 | 6.1 | Temporal core identified and selected | tested | none | seal temporal:MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json |
| C2 | 6.2 | Typed cognitive workspace | measured | A2 | classify the result through mop.method.gate.classify_result and record it |
| C3 | 6.3 | Mixture of Perspectives, perspectives as processes | measured | C2 | classify the result through mop.method.gate.classify_result and record it |
| C4 | 6.4 | Perspective selection ladder | measured | C3 | classify the result through mop.method.gate.classify_result and record it |
| C5 | 6.5 | Perspective arbitration | measured | C3 | classify the result through mop.method.gate.classify_result and record it |
| M1 | 7.1 | Working memory | not_started | C2 | implement src/mop/cognition/memory.py |
| M2 | 7.2 | Episodic memory | not_started | C2 | implement src/mop/cognition/memory.py |
| M3 | 7.3 | Semantic memory | not_started | M2 | implement src/mop/cognition/memory.py |
| M4 | 7.4 | Procedural memory | not_started | M2 | implement src/mop/cognition/memory.py |
| M5 | 7.5 | Consolidation | not_started | M2, M3, M4 | implement src/mop/cognition/memory.py |
| M6 | 7.6 | Forgetting and hygiene | not_started | M3 | implement src/mop/cognition/memory.py |
| W1 | 8 | World model | not_started | C2, M2 | implement src/mop/cognition/world.py |
| S1 | 9 | Self model | not_started | C2, M2 | implement src/mop/cognition/selfmodel.py |
| K1 | 10 | Metacognition | not_started | C5, S1 | implement src/mop/cognition/metacog.py |
| P1 | 11.1 | Plasticity hierarchy | not_started | A4, C2 | implement src/mop/cognition/plasticity.py |
| P2 | 11.2 | Fast adaptation | not_started | P1, M1 | implement src/mop/cognition/plasticity.py |
| P3 | 11.3 | Slow adaptation | not_started | P1, M5 | implement src/mop/cognition/plasticity.py |
| P4 | 11.4 | Plasticity policy | not_started | P2, P3 | implement src/mop/cognition/plasticity.py |
| P5 | 11.5 | Learning to learn | not_started | P4 | implement src/mop/cognition/plasticity.py |
| R1 | 12 | Bounded functional reorganization | not_started | P1, C4, A4 | implement src/mop/cognition/plasticity.py |
| B1 | 13 | Model body interface | not_started | C2 | implement src/mop/cognition/body.py |
| T1 | 14 | Operationalized thinking | not_started | C5, K1 | implement src/mop/cognition/batteries.py |
| E1 | 15.1 | Continuity | not_started | M2, S1 | implement src/mop/cognition/batteries.py |
| E2 | 15.2 | Unity | not_started | C2, C5 | implement src/mop/cognition/batteries.py |
| E3 | 15.3 | Reflective access | not_started | S1, M3 | implement src/mop/cognition/batteries.py |
| E4 | 15.4 | Endogenous attention | not_started | K1 | implement src/mop/cognition/metacog.py |
| E5 | 15.5 | Autonomous goal maintenance | measured | A4, E1 | classify the result through mop.method.gate.classify_result and record it |
| E6 | 15.6 | Cognitive integrity | tested | A4, M2 | seal SUBSTRATE_INDEPENDENT_VERIFICATION.json |

## Selected next batch

Primary: C1 Temporal core identified and selected. seal temporal:MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json
Secondary: C2 Typed cognitive workspace. classify the result through mop.method.gate.classify_result and record it

Activation remains false.
