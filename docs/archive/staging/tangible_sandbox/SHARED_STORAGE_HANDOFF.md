# Shared storage handoff

Measured 2026-08-02 after the R2 replacement lane completed:

| Quantity | Bytes | Meaning |
| --- | ---: | --- |
| Free now | 363,011,293,184 | 363.01 GB decimal / 338.08 GiB on the volume |
| Retained substrate data | 175,857,762,304 | Existing corpus/cache/source material; already counted by `df` |
| R2 private receipts | 38,532 | What the R2 run itself left in its receipt trees |
| R2 measured own-growth allowance | 1,048,576 | Dynamic storage measurement, not a 100+ GB generation result |
| R2 continuity floor | 200,008,355,840 | 186.27 GiB of free space to preserve while a continuity lane is live |
| Residual above that floor | 163,002,937,344 | 163.00 GB decimal / 151.81 GiB for other work under the same guard |
| Optional 100 GiB external reserve | 107,374,182,400 | A user/project policy choice, not a substrate requirement |
| Residual after that optional reserve | 55,628,754,944 | 55.63 GB decimal / 51.81 GiB |

The 286.27 GiB figure is therefore a combined policy envelope:

```text
186.27 GiB R2 continuity floor + 100 GiB optional external reserve
```

It is not substrate’s generated-data footprint. The R2 lane completed 24 hours
with nine checkpoints. Its host free-space delta was about 59.4 GB, but that is
only a host-level observation; there is no file-level receipt attributing that
delta to R2 generation. It must not be reported as generated output.

Odyssey private generation has not started. Its current design permits up to
100 GiB of initial task material and a 120 GiB private-write cap, but those are
future bounded-write allowances subject to preflight and rehearsal. They are
not currently occupied storage.

Rules for other projects:

1. Count the retained substrate corpus once through the filesystem measurement.
2. Preserve the 200.01 GB decimal R2 continuity floor only when a new continuity
   lane is live or being admitted.
3. Treat a 100 GiB user reserve as optional and explicit; do not add it to
   substrate’s requirement by default.
4. Do not reserve both the model-ladder envelope and the same user reserve for
   the same bytes.
5. This handoff does not authorize Odyssey; its transition gate still requires
   independent evidence reconciliation and a fresh preflight.
