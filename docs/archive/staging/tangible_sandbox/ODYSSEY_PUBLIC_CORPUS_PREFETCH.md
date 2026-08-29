# Odyssey public-corpus prefetch

This cache queue is deliberately separate from R2 and from the Odyssey
scientific adoption register. It acquires only three pinned, public sources:
the MIT-licensed MATH repository, the CC-BY-SA-4.0 AI2 ARC dataset, public
SWE-bench multimodal metadata, MOVi-A 128px physical video/state data, and
the CC-BY-4.0 CLEVR visual-spatial archive. Small sources receive tree hashes;
the 64GB MOVi tier receives a pinned Google Cloud object/MD5 manifest, which
avoids an unnecessary post-download read of every byte during R2.

It has a hard stop floor of 307,382,538,240 bytes: the currently sealed R2
floor (200,008,355,840 bytes) plus a 100 GiB reservation for unrelated user
work. The queue uses at most two downloader workers per Hugging Face source,
checks capacity before and after each item (and reserves each object before
writing it), and records an interruption
instead of retrying around that floor.

This does not accept terms for gated data or adopt any cached source into an
Odyssey arm. COCO/Flickr images, Common Voice, hosted simulator assets,
OSWorld, WorkArena, MLE-bench, and any credentials- or terms-bound resource
remain explicitly outside the queue.
