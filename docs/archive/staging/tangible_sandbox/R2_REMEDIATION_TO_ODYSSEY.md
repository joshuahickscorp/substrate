# R2 remediation → Odyssey bridge

The 24-hour R2 trace `r2-32aff05267f84812b59a8ce35b030db0` is preserved as
an **invalid diagnostic**.  It cannot become early Odyssey elapsed time,
candidate state, result evidence, or an R2 pass by being appended to a later
run.

The direct audit found that all three required interventions were actually
performed and checkpointed: two human corrections, a sensor interruption with
a telemetry fallback, and an ffmpeg video-decoder tool/body change. The
terminal validator read the compact trace receipt rather than the checkpoint.
That receipt omitted the correction and tool objects and overwrote the sensor
object with a boolean. It therefore failed honestly rather than silently
accepting incomplete trace evidence.

The code repair preserves those concrete objects in every compact work receipt.
The regression test creates correction, sensor, and tool events and asserts
their terminal-validator evidence remains present.

## Chain rule

```text
preserved invalid R2 diagnostic
  → source-level serialization repair + regression test
  → fresh, full 24-hour R2 remediation run + independent verification
  → Odyssey G01 may be considered
  → sealed seven-day Odyssey
```

The bridge carries only read-only operational lessons: failure classification,
checkpoint digests, and the regression test. It makes the three stressors
mandatory Odyssey material on Days 3–5 and requires a Day-7 trace/checkpoint
reconciliation before evaluator reveal. It does not pool scores or elapsed
time across the invalid and fresh histories.
