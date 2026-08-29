# R2-to-Odyssey storage handoff

R2 does not have a second physical data store. Its retained raw archives,
public benchmark data, and source checkouts sit on the same APFS volume as the
Odyssey cache. The R2 cache contains hard links to archive objects, so it uses
no extra physical archive space.

At the handoff measurement, the physical figures were:

| Category | Physical size |
| --- | ---: |
| R2 retained archives (including their hard-linked cache once) | 86.04 GB |
| Existing public benchmark material | 1.20 GB |
| Pinned source checkouts | 4.08 GB |
| R2 run traces, artifacts, and evidence | 32.7 MB |
| Broken R2 attempts | 16.6 KB |
| Public Odyssey prefetch target | 83.14 GB |

Thus the earlier R2 problem was not caused by large R2-generated outputs: the
program generated only tens of megabytes. It was an admission/free-space
guard problem while large shared resources were present on the same volume.

While R2 is live, its sealed minimum-free-byte requirement remains untouched.
After R2 independently verifies, it retires and is replaced by one Odyssey
reservation: dynamic protected floor plus measured private Odyssey growth,
transient writes, terminal allowance, and the user’s then-current workspace
reserve. The shared read-only lake is counted once by `df` free space rather
than reserved again as a separate dataset cost.
