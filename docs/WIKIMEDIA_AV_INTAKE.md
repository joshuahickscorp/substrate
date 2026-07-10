# Synchronized audio-video intake: local mechanics cohort

## Decision

Use a frozen 12-object Wikimedia Commons cohort for AL3/DR15 **mechanics only**. Every selected
object is an original muxed WebM with both audio and video streams reported by Wikimedia's live
`videoinfo` API. Every object is creator-owned work marked CC0, and the manifest pins its page id,
title, original object URL, byte size, Wikimedia SHA-1, upload timestamp, license metadata, duration,
and audio/video stream descriptors.

This does not make the cohort a natural benchmark and does not make AL3 or DR15 runnable science.
It removes a narrower uncertainty: we now have a small, stable, rights-documented route for testing
same-container timing, temporal offset nulls, object derangements, exact referents, and split hygiene.
It does not yet certify semantic synchrony: after download, ffprobe must verify the two stream clocks
and a human must confirm that the visible event and recorded sound genuinely correspond.

The frozen source manifest is
`configs/audiovisual/wikimedia_commons_cc0_v1.json`. The metadata-only receipt is
`proof/WIKIMEDIA_AV_INTAKE_DRY_RUN.json`.

## Why this source won

Wikimedia Commons' official scope says hosted works must permit reuse for any purpose, and its
licensing policy requires derivative and commercial reuse. The selected files additionally report
CC0 individually. CC0 permits copying, modification, distribution, and performance without asking
permission. These are copyright permissions, not a privacy warranty: Commons itself warns that
personality, privacy, trademark, and other non-copyright restrictions can remain.

Primary sources:

- Wikimedia Commons project scope: <https://commons.wikimedia.org/wiki/Commons:Project_scope/Summary>
- Wikimedia Commons licensing policy: <https://commons.wikimedia.org/wiki/Commons:Licensing>
- Wikimedia non-copyright restrictions: <https://commons.wikimedia.org/wiki/Commons:Non-copyright_restrictions>
- CC0 1.0: <https://creativecommons.org/publicdomain/zero/1.0/>
- Wikimedia `videoinfo` API help: <https://commons.wikimedia.org/w/api.php?action=help&modules=query%2Bvideoinfo>

The selection is deliberately not a scrape or a YouTube link list. Acquisition uses the original
`upload.wikimedia.org` object URLs returned by the official API, and downloaded bytes must reproduce
the API's SHA-1 and size before atomic rename.

## Candidate comparison

| Candidate | Rights and authority | Small selective unit | Split/privacy | Decision |
|---|---|---:|---|---|
| Frozen Commons CC0 originals | Per-object CC0, page id, exact object URL, SHA-1, size, upload timestamp | Yes, individual muxed files | Curator-frozen 8/2/2; manual privacy review still required | Selected for mechanics |
| [RAVDESS](https://zenodo.org/records/1188976) | Zenodo DOI/checksums, CC BY-NC-SA 4.0 | Actor archives are about 500 MB each | A minimum speaker-disjoint three-way split is about 1.5 GB and retains identifiable actors | Reserve speech benchmark |
| [EPIC-SOUNDS](https://epic-kitchens.github.io/epic-sounds/site) + EPIC-KITCHENS-100 | Official Oxford/Bristol source, CC BY-NC for EPIC-SOUNDS | Annotations are small; source RGB recordings are the relevant AV unit | Excellent later natural benchmark, with its own privacy/source obligations | Defer, not the smallest intake |
| [TAU Urban AV Scenes 2021](https://zenodo.org/records/4477542) | Zenodo DOI and archive MD5s | No; development release is 107.7 GB of ZIP parts | Official protocol, public-scene review required | Reject for current disk envelope |
| AudioSet, VGGSound, AVE and similar link lists | Annotation rights do not grant rights to volatile third-party YouTube media | Links are selective, bytes are not publisher-authoritative | Heterogeneous privacy and link rot | Reject |
| Blender/open-film media | Strong open licensing and publisher hosting | Possible | Authored soundtrack and no official split | Inferior to smaller per-object CC0 originals for this mechanics lane |

## Frozen cohort

The cohort covers impulsive, periodic, pass-by, vocal, water, weather, phase-change, and machine
events. All creators and capture families are unique.

| Split | Objects | Source bytes | Discipline |
|---|---:|---:|---|
| train | 8 | 50,629,186 | method fitting and null construction |
| validation | 2 | 10,085,255 | thresholds and diagnostics |
| test | 2 | 35,076,985 | media locked until a protocol binds the exact manifest hash |
| total | 12 | 95,791,426 (91.35 MiB) | frozen before media download or decode |

The split is **not official**. It is defensible because it was frozen before media access, every unit
is an independently versioned source object, creators and capture families do not cross splits, and
the manifest identity is hashed into every plan. Calling it official would be false.

## Temporal-control contract

The positive pair is audio and video decoded from one original muxed container at common time zero.
No caption, separately sourced sound, or proxy encoder output may substitute for source audio.

The intake freezes two null families without choosing a winning statistic:

1. Circularly offset the audio within a clip by 0.25, 0.50, and 0.75 of that clip's duration.
2. Derange audio across objects within the same split, deterministically from the manifest hash and
   experiment seed.

Thresholds, boundary tolerance, aggregation, encoder choice, and stopping rules must be
preregistered after train/validation inspection and before any test media access. Test controls may
not be tuned.

## Safety and post-CM7 execution

This run performs metadata/API requests only. It does not open an object URL, download a media byte,
run ffprobe, or access test media.

After CM7 is complete, acquire **train and validation only** with:

```bash
PYTHONPATH=src .venv/bin/python scripts/studio/wikimedia_av_intake.py \
  --execute-train-validation --confirm-cm7-complete \
  --manifest configs/audiovisual/wikimedia_commons_cc0_v1.json \
  --destination data/raw/wikimedia_commons_cc0_av_mechanics_v1 \
  --proof proof/WIKIMEDIA_AV_TRAIN_VALIDATION_INTAKE.json
```

Expected source bytes are exactly **60,714,441** (57.90 MiB). The guard reserves twice that amount,
**121,428,882 bytes**, above the 40 GB free-disk floor for final plus atomic partial files. The
command rechecks live authority, refuses to run while a CM7 process exists, downloads sequentially,
reproduces SHA-1 and size, and runs local ffprobe on each landed train/validation container, including
an audio-versus-video stream-start tolerance of 100 ms.

The CLI has no test-download mode. The locked test payload is **35,076,985 bytes**. A later test
intake must require a frozen experiment-protocol receipt that binds this exact manifest identity.

## Privacy and claim boundary

Every selected description and category was screened before download and every file is marked as
creator-owned CC0 work. That is not enough to establish the absence of incidental people, voices,
license-incompatible background media, location sensitivity, or personality rights. After source
bytes land, a human must review audio and sampled frames and write the frozen privacy receipt named
in the manifest. Until it passes, consumer use remains blocked.

Even after privacy review, the cohort licenses only:

- same-container stream and clock verification;
- temporal-offset and cross-object-shuffle mechanics;
- exact referent/split plumbing;
- tiny local encoder-cache rehearsal.

It does not establish natural audio-video boundary universality, substrate universality, cognition,
sentience, or modality-general reasoning. AL3 still needs citable audio and video encoders over these
exact referents, a preregistered statistic, seed/uncertainty controls, and locked-test evaluation.
DR15 still needs compatible relational/language task families in addition to audio/video data.
