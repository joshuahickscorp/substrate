# Substrate ledger

Generated from the tree at commit `f4bf65cba05be8acb6996a9d62bf1afc9a13de58`. Status is derived, never asserted: an item is
implemented because its files exist, tested because a recorded test ledger says so, measured
because its evidence is sealed, and terminal because a scientific classification exists for it.

Items: 35. Levels: measured 20, not_started 6, terminal 7, tested 2.

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
| M1 | 7.1 | Working memory | measured | C2 | classify the result through mop.method.gate.classify_result and record it |
| M2 | 7.2 | Episodic memory | measured | C2 | classify the result through mop.method.gate.classify_result and record it |
| M3 | 7.3 | Semantic memory | measured | M2 | classify the result through mop.method.gate.classify_result and record it |
| M4 | 7.4 | Procedural memory | measured | M2 | classify the result through mop.method.gate.classify_result and record it |
| M5 | 7.5 | Consolidation | measured | M2, M3, M4 | classify the result through mop.method.gate.classify_result and record it |
| M6 | 7.6 | Forgetting and hygiene | measured | M3 | classify the result through mop.method.gate.classify_result and record it |
| W1 | 8 | World model | measured | C2, M2 | classify the result through mop.method.gate.classify_result and record it |
| S1 | 9 | Self model | measured | C2, M2 | classify the result through mop.method.gate.classify_result and record it |
| K1 | 10 | Metacognition | not_started | C5, S1 | implement src/mop/cognition/metacog.py |
| P1 | 11.1 | Plasticity hierarchy | measured | A4, C2 | classify the result through mop.method.gate.classify_result and record it |
| P2 | 11.2 | Fast adaptation | measured | P1, M1 | classify the result through mop.method.gate.classify_result and record it |
| P3 | 11.3 | Slow adaptation | measured | P1, M5 | classify the result through mop.method.gate.classify_result and record it |
| P4 | 11.4 | Plasticity policy | measured | P2, P3 | classify the result through mop.method.gate.classify_result and record it |
| P5 | 11.5 | Learning to learn | measured | P4 | classify the result through mop.method.gate.classify_result and record it |
| R1 | 12 | Bounded functional reorganization | measured | P1, C4, A4 | classify the result through mop.method.gate.classify_result and record it |
| B1 | 13 | Model body interface | measured | C2 | classify the result through mop.method.gate.classify_result and record it |
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
