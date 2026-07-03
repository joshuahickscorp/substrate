# SEMANTIC_POSITIONS

The semantic layer of the Mixture of Perspectives (MoP) audit: a catalogue of the positions on THOUGHT the program commits to or contests, each carrying an operationalization, a preregistered null, and a non-vacuous control.

## 1. Preface: why this layer exists

MoP runs heterogeneous reasoning MODES over a FROZEN perceptual substrate (V-JEPA 2, a video world-model whose native perspective is visual-temporal and intuitive-physical), routed by a tiny trainable shell and licensed by error-decorrelation. The substrate thinks in video. But THOUGHT lives in many perspectives: language, code, mathematics, formal physics, each a different formal or symbolic way of carving the world. The hard questions the program raises are therefore not architectural, they are SEMANTIC:

- What is COMMUNICATION with a latent-native model whose medium is perceptual latents, not tokens? Is understanding a request a decode-criterion or a use-criterion?
- Can such a model be made PLASTIC (moldable, like a child's brain) through language, code, math, and physics, or does a frozen substrate cap plasticity at the shell?
- Is there one thought expressible across perspectives, or are the perspectives incommensurable and the mixture a mere ensemble?

This document is the SEMANTIC audit layer. The other layers (thinking modes, reasoning program, plasticity program, currencies atlas, workspace) test mechanisms. This layer tests MEANING: it names every position on thought/meaning/communication the program leans on, gives each a concrete test on a frozen-substrate + tiny-shell setup, and marks honestly which ones a frozen substrate structurally cannot answer.

### How to read the catalogue

Every position is one row with these fields:

- **id**: SEM-COMM / LANG / CODE / MATH / PHYS / PLAS / META, numbered.
- **position**: the claim about thought MoP commits to or contests.
- **question**: the single empirical question that would settle it.
- **lineage**: the real philosophical / cognitive-science source, author-title-year VERIFIED by web search where possible, UNVERIFIED where not.
- **operationalization**: how MoP tests it on frozen substrate + tiny shell.
- **null**: the preregistered outcome under which the position fails.
- **control**: the NON-VACUOUS baseline (random-init encoder at matched resolution, matched compute, tuned baseline, noisy-TV guard). NEVER a square full-rank latent projection, which is invertible and therefore vacuous.
- **tier**: `cpu-now` (runnable today on cached features), `studio` (needs GPU / more compute but no new modality), `needs-new-cache` (needs a stimulus set or a second encoder we have not built), `open-philosophical` (not cleanly operationalizable yet, marked honestly).
- **coverage**: what the corpus tests today, with the experiment id.
- **gap**: what is missing.
- **priority**: `flagship` / `high` / `medium` / `exploratory`.

### Doctrine that separates this from armchair philosophy

Every testable position carries an operationalization with a preregistered null AND a non-vacuous control. Philosophy and cognitive science are SOURCES OF HYPOTHESES here, never evidence. Two doctrinal spines recur:

1. **The vacuous-control trap.** A `frozen_random_projection` of the latent is full-rank and invertible, so a linear probe over it decodes everything the real latent does and any real-minus-control delta is forced toward zero. The only legitimate substrate control is a random-init same-architecture ViT-L at MATCHED resolution (the `substrate_vs_random_init_vit` spine, p=0.029), or a random-map-of-equal-rank scored by kNN-topology overlap (the AT1-relrep spine), never a square projection.

2. **Decodability is not meaning.** A linear probe can read a variable the shell never consults. The load-bearing semantic move in this layer is to score meaning by USE (intervention, behavioral delta, negotiation success), not by readout, and to guard that the use-metric is not itself invariant to an invertible linear map (or it reinherits the vacuity trap in semantic clothing).

### A note on scope corrections in this edition

A verification pass over the repository (cache inventory + tool inventory) forced several retiers that the raw domain briefs had wrong. The corpus caches available today are: the 5x5 shape-color bound-nuisance clipset (`bound_nuisance_v1`, 200 clips), its random-init ViT-L twin (`randominit_vitl_nuisance`), handcrafted descriptors, the 25-d programmatic reference (one-hot shape + one-hot color + 15 continuous nuisance columns; NO count, relation, or exists factor), a real-encoder clipset, single-frame V-JEPA, and text/audio sonified caches (Qwen-textified, wav2vec2). There is NO DSL, NO executor, NO physics clipset, NO numerosity/geometry stimulus set, and NO event-boundary annotation anywhere in the tree. Positions that assumed those artifacts have been retiered to `needs-new-cache` with the missing artifact named, because the live-state encoder ban means new caches cannot be produced now. These corrections are called out inline and summarized in Section 6.

---

## 2. COMMUNICATION with a latent-native model

The founding question of the program: what is it to address, prompt, or communicate with a system whose medium is perceptual latents, not tokens. The canonical result of this section is that the two central flagships are a use-vs-decode intervention and a two-agent referential game. The verification pass found these two designs RESTATED across the Communication, Meta-Semantics, and Code briefs (three copies of the use criterion, four copies of the signalling game); they are collapsed here to two canonical flagships (SEM-COMM-1, SEM-COMM-2), and the per-domain variants are cross-referenced as EXTENSIONS, not independent flagships.

### SEM-COMM-1 (flagship) — Communication is a use-difference, not a readout

| field | value |
|---|---|
| **position** | To COMMUNICATE with a latent-native model is not to decode a variable FROM its states but to change what its shell DOES. A message succeeds only if it causes a use-difference the model would not otherwise make. A readable state is not an addressed state. |
| **question** | Does a shell that has genuinely received a message behave counterfactually differently in a way a linear readout of the same latent cannot reproduce, so that communication is a use-criterion, not a decode-criterion? |
| **lineage** | Wittgenstein, *Philosophical Investigations* (Blackwell, 1953), meaning-as-use and the private-language argument from PI 243 (VERIFIED). Brandom, *Making It Explicit* (Harvard UP, 1994), meaning as inferential/use role (VERIFIED). Dennett, *The Intentional Stance* (MIT Press, 1987), aboutness as a predictive/use stance (VERIFIED). |
| **operationalization** | Frozen V-JEPA latents + two tiny NONLINEAR shells. A message is a fixed low-dim code injected into shell B alongside the latent. Communication score = the counterfactual behavior shift caused by the message (accuracy delta AND policy divergence, KL over shell outputs, between message-present and message-ablated), NOT probe decodability of the message. Preregister: communication is present only if the use-delta exceeds what a best linear probe of the same latent reads out. The decode-vs-use gap is the quantity of interest. |
| **null** | The message produces no downstream use-delta beyond what a linear probe already reads from the raw latent; every behavioral change is explained by already-decodable information, so communication collapses to a readout head. Use-delta within seed spread of the decode-only baseline. |
| **control** | Frozen random-init shell B at matched capacity and matched channel width (cannot have learned to USE the message) PLUS a decode-only floor (best linear probe of the message off the raw latent). Square full-rank projection banned. **3d-vacuity guard (from verification):** the consuming shell must be genuinely NONLINEAR with a verified nonlinear dependence on the message; "unconsulted" must be established by ablation on the TRAINED shell's OUTPUT, not by matched linear decodability; confirm the behavioral-delta metric is NOT invariant to an invertible linear map of the latent before trusting any nonzero delta, else the p10/census vacuity recurs. |
| **tier** | cpu-now |
| **coverage** | Gap. Every semantic probe (p1, p2, s1, s10) asks "is X decodable". DR4 tests counterfactual-match on the substrate, not message-driven use-change for a shell. Blind-spots 1 and 10 posed as a testable use-criterion. |
| **gap** | The corpus has no preregistered communication-success metric distinct from "a probe decodes" or "an alignment map fits". This position supplies the missing use-based definition and the exact non-vacuous control set. |
| **priority** | flagship |

### SEM-COMM-2 (flagship) — Communication is a two-agent negotiated act

| field | value |
|---|---|
| **position** | Communication is intrinsically a TWO-AGENT act: meaning is fixed by successful sender-receiver coordination through a channel, not by any property of one shell's internal code. A protocol exists only if a sender and receiver co-adapt to succeed above a no-shared-history floor. |
| **question** | Can two tiny shells over the frozen substrate NEGOTIATE a signalling convention through a bandwidth-limited channel and reach referential-game success above a chance floor, where a single shell's code stability (already refuted by p5/s5/y3) says nothing? |
| **lineage** | Lewis, *Convention* (Harvard UP, 1969), sender-receiver signalling games (VERIFIED). Skyrms, *Signals* (Oxford UP, 2010), reinforcement-learned signalling (VERIFIED). Lazaridou, Peysakhovich, Baroni, "Multi-Agent Cooperation and the Emergence of (Natural) Language", ICLR 2017, arXiv 1612.07182 (VERIFIED). |
| **operationalization** | Referential game on frozen V-JEPA: sender shell sees a target clip latent among distractors, emits a discrete symbol through a capacity-limited channel; receiver shell picks the target from latents. Success = receiver accuracy above 1/k across held-out clips. Both shells tiny and trainable, substrate frozen. Test whether success requires co-adaptation (joint training) vs fails with a frozen-random receiver. |
| **null** | Receiver accuracy ties the frozen-random-receiver floor OR the matched-channel-capacity information ceiling: coordination is explained by raw bandwidth, not a negotiated protocol. |
| **control** | (a) frozen random-init receiver at matched channel width; (b) matched-channel-capacity baseline where the symbol is a fixed random hash of the target index (bandwidth-only upper bound); (c) mismatched-history control (sender/receiver trained on disjoint seeds, paired cold). Convention is real only if joint-trained pairs beat all three. |
| **tier** | cpu-now |
| **coverage** | Gap. p5/s5/y3 tested one shell's code stability across seeds (refuted below frozen-random). No experiment tests two shells negotiating a code. Blind-spot 2, the most load-bearing under-operationalized piece of the thesis. |
| **gap** | The program ASSERTS communication-with-a-latent-model but never ran a sender-receiver success test. |
| **priority** | flagship |

### SEM-COMM-3 (high) — Aboutness is consumer-side use, not decodability

| field | value |
|---|---|
| **position** | A latent state REFERS to a world feature only if the shell that CONSUMES it treats it as being about that feature. Decodability is not aboutness; a probe can read a variable the shell never consults. (An EXTENSION of SEM-COMM-1 to the Millikan consumer-semantics frame.) |
| **question** | Does ablating/perturbing a candidate referring direction change the shell's behavior in the feature-appropriate way aboutness predicts (errors migrate toward shape-confusable classes), rather than a generic accuracy drop? |
| **lineage** | Millikan, *Language, Thought, and Other Biological Categories* (MIT Press, 1984), content fixed by the CONSUMER's proper function (VERIFIED). Dennett, *The Intentional Stance* (1987) (VERIFIED). Brandom (1994) (VERIFIED). |
| **operationalization** | Identify a candidate referring direction by a probe, ablate/rotate ONLY that direction in the latent fed to a nonlinear shell trained on a shape-dependent task. Aboutness score = the SPECIFIC predicted behavioral change (feature-confusable error migration), not a generic drop. Contrast against ablating a decodable-but-task-irrelevant direction. |
| **null** | Ablating the decodable direction produces only generic accuracy loss, indistinguishable from a matched-magnitude random direction. Behavioral-signature delta within the random-direction floor. |
| **control** | (a) matched-magnitude random-direction ablation; (b) a decodable-but-task-irrelevant direction (consumption-specificity test). Same 3d-vacuity guard as SEM-COMM-1: nonlinear shell, output-ablation for "unconsulted". |
| **tier** | cpu-now |
| **coverage** | Gap (blind-spot 1). DR4 tests counterfactual-match on the substrate, not aboutness-for-the-consuming-shell. |
| **gap** | Millikan's consumer-semantics move has no operationalization anywhere. |
| **priority** | high |

### SEM-COMM-4 (high) — Context-sensitive meaning (pragmatics)

| field | value |
|---|---|
| **position** | Meaning is CONTEXT-SENSITIVE: the same latent means different things under different task/routing contexts. A model whose latents have fixed context-free meanings cannot be prompted, only queried. |
| **question** | Does the shell's read of the SAME frozen latent shift with routing/task context (one latent, two task-relative meanings) above a context-blind classifier? |
| **lineage** | Sperber and Wilson, *Relevance* (Blackwell, 1986), ostensive-inferential communication (VERIFIED). Grice, "Meaning", *Philosophical Review* 66 (1957) 377-388 (VERIFIED). Brandom (1994) (VERIFIED). |
| **operationalization** | Same frozen latent, two tasks selected by a routing token (task A = decode shape, task B = decode motion direction). Measure whether a shell conditioned on the routing token extracts DIFFERENT content from the identical latent (both above chance) more than a single context-blind head, i.e. context re-purposes the same state rather than the router gating two independent readouts. |
| **null** | The context token buys nothing beyond running two independent probes: a matched-capacity multi-head baseline matches the routed shell. Routed-minus-blind delta within seed spread. |
| **control** | Matched-capacity multi-head baseline (no routing) AND a scrambled-routing-token control (token uncorrelated with task). |
| **tier** | cpu-now |
| **coverage** | Gap (blind-spot 3). Every meaning test is context-free. |
| **gap** | No pragmatic meaning test exists; this is the piece that shows a latent can be ADDRESSED into a sense. |
| **priority** | high |

### SEM-COMM-5 (high) — Symbol-to-percept grounding (drive, not readout)

| field | value |
|---|---|
| **position** | Grounding runs the WRONG way in the corpus: it is tested as "a code tracks a world variable" (readout), never as "a symbol DRIVES a perceptual expectation" (top-down). Genuine grounding is bidirectional; the untested direction shows the substrate is addressable, not merely readable. |
| **question** | Can a discrete symbol injected into the shell steer the frozen substrate's FORWARD prediction toward the percept the symbol denotes, above a random-symbol floor? |
| **lineage** | Harnad, "The Symbol Grounding Problem", *Physica D* 42 (1990) 335-346, DOI 10.1016/0167-2789(90)90087-6 (VERIFIED). Clark, *Surfing Uncertainty* (Oxford UP, 2016) (VERIFIED). Barsalou, "Perceptual Symbol Systems", *BBS* 22 (1999) (VERIFIED). |
| **operationalization** | Inject a symbol as a conditioning code into the shell driving V-JEPA's own predictor; measure whether the predicted future latent moves toward the region occupied by clips the symbol denotes (nearest-cluster hit-rate on held-out clips) vs a shuffled-symbol condition. |
| **null** | Conditioning on the correct symbol moves the predicted latent no closer to the denoted cluster than a shuffled symbol. Hit-rate delta within seed spread of the shuffled floor. |
| **control** | Shuffled-symbol (label permuted) AND random-init-predictor at matched architecture/resolution. Square projection banned. |
| **tier** | studio (needs the forward predictor run; retiered from cpu-now, the predictor pass is compute-heavy and touches the encoder lane) |
| **coverage** | Partial/gap (blind-spot 9). s1/ex16/s10 test grounding as READOUT (refuted). |
| **gap** | No test of the generative/top-down direction of grounding. |
| **priority** | high |

### SEM-COMM-6 (medium) — Indexicality / perspective-binding

| field | value |
|---|---|
| **position** | MoP is literally "perspectives", so the deepest content for this substrate is INDEXICAL: a latent may encode a point of view (egocentric vs allocentric). To address such a model is partly to specify a frame. Perspective-binding is distinct from viewpoint-decodability. |
| **question** | Does a latent encode viewpoint-RELATIVE vs world-RELATIVE position as separable, bound content, above a random-init-ViT floor and off-ceiling, rather than merely allowing viewpoint to be decoded? |
| **lineage** | Perry, "The Problem of the Essential Indexical", *Noûs* 13(1) (1979) 3-21 (VERIFIED: indexicals essential, cannot be eliminated for descriptions; the supermarket example). Millikan (1984) on indexical/local representations (VERIFIED). |
| **operationalization** | On clips with known camera and object poses, probe egocentric vs allocentric object position from the frozen latent, off-ceiling. Perspective-binding = both frames separably present AND a shell can be routed to USE one frame vs the other, contrast against random-init-ViT at matched resolution. |
| **null** | The two frames are not separably encoded (one is a trivial transform of the other) OR real V-JEPA ties random-init-ViT on the harder frame. Frame-separability delta within the random-init floor. |
| **control** | Random-init-ViT at matched architecture and 256px, plus an off-ceiling difficulty gate. Square projection banned. |
| **tier** | needs-new-cache (needs multi-viewpoint pose-labeled clips run through the encoder; retiered, no such cache exists) |
| **coverage** | Gap (blind-spot 6). a2 viewpoint-transfer tests decodability, not binding. |
| **gap** | Nothing tests whether a latent binds a point-of-view despite MoP being named for perspectives. |
| **priority** | medium |

### SEM-COMM-7 (medium) — The workspace as communication bus

| field | value |
|---|---|
| **position** | The workspace IS the communication bus: cross-mode I/O happens iff a narrow shared broadcast slot lets heterogeneous modes exchange correctness-relevant information no single mode holds. Real only where a NARROW bottleneck beats an unstructured wide fusion at matched capacity. |
| **question** | Does a narrow broadcast workspace over two frozen substrates carry cross-mode agreement information that predicts correctness better than single-substrate confidence, AND beat an equal-capacity unstructured MLP? |
| **lineage** | Baars/Dehaene Global Workspace Theory, broadcast bottleneck, borrowed as STRUCTURE not consciousness (VERIFIED per corpus section 07). Goyal et al. 2022 shared-workspace coordination (VERIFIED per corpus). Brandom (1994) (VERIFIED). |
| **operationalization** | WS1 as written: cross-substrate agreement (V-JEPA + a second frozen encoder, e.g. DINOv2) as a central gate; workspace-width swept at FIXED total params. Bus score = held-out correctness prediction from agreement beating single-substrate confidence AND the narrow workspace beating the equal-capacity concat-MLP. |
| **null** | The structured narrow workspace ties the unstructured equal-capacity MLP OR cross-substrate agreement ties single-substrate confidence. No narrowness benefit at fixed capacity. |
| **control** | Equal-trainable-param, equal-FLOP unstructured MLP over concatenated inputs; single-best-substrate baseline; matched-compute unrolled-depth floor for any iterative broadcast. Workspace must earn a genuine SECOND substrate, not a remap of one. |
| **tier** | cpu-now (V-JEPA + DINOv2 caches both exist) |
| **coverage** | Reframes WS1/A4. Design and controls exist; the communication-bus framing is the extension. |
| **gap** | The corpus builds the workspace as arbitration, not as the cross-mode bus. |
| **priority** | medium |

### SEM-COMM-8 (exploratory, OPEN) — Normativity: a concept can be misapplied

| field | value |
|---|---|
| **position** | A concept can be WRONG (misapplied), not merely low-accuracy: communication presupposes a correctness condition distinct from classifier error. Without a separable norm, a latent-native model can be inaccurate but never mistaken, and cannot be genuinely corrected. |
| **question** | Can a correction signal in a two-shell exchange distinguish a MISAPPLICATION (right rule, wrong instance) from ordinary low accuracy? |
| **lineage** | Millikan (1984), misrepresentation from failed proper function (VERIFIED). Brandom (1994), correctness as the giving-and-asking-for-reasons norm (VERIFIED). Wittgenstein PI (1953), rule-following (VERIFIED). |
| **operationalization** | Honestly OPEN. Candidate: in the SEM-COMM-2 game, inject systematic sender misapplications (correct convention, wrong referent by rule) vs random sender errors; test whether a receiver-side correction separates the two above a shuffled-correction floor. If no separable normative signal survives, mark OPEN, do not fake a norm. |
| **null** | The correction responds identically to systematic misapplication and random error (tracks error rate, not norm-violation); shuffled-correction ties the trained one (the ex18/DR9 pattern in the normative frame). |
| **control** | Shuffled/untrained correction signal AND a random-error condition matched in accuracy-cost. If the correction cannot separate misapplication from matched-rate random error, report OPEN. |
| **tier** | open-philosophical |
| **coverage** | Gap (blind-spot 5). Nearest machinery ex18/DR9 (refuted) tests correction-signal existence, not norm-violation specificity. |
| **gap** | Correctness-of-a-concept is entirely absent. |
| **priority** | exploratory |

---

## 3. LANGUAGE as a perspective

Is thought language-dependent or language-independent? Language as compression vs scaffold vs thought-itself, linguistic relativity on a text-free substrate, discrete-symbol vs continuous-latent reasoning at matched compute, and whether a frozen LLM's hidden states are a genuinely different perspective from V-JEPA on the same content. Note: the flagship relativity and grounding-asymmetry positions here need a language-grounded encoder cache; the Qwen-textified and wav2vec2 caches exist but a paired vision+text encoder on identical content does not, so those are `needs-new-cache`.

### SEM-LANG-1 (high) — Strong language-of-thought (systematicity ceiling)

| field | value |
|---|---|
| **position** | The substrate cannot represent structured systematic content without an inner symbolic medium, so a text-free video encoder should show a systematicity ceiling only a symbolic shell can lift. |
| **question** | Does frozen V-JEPA exhibit compositional systematicity at a level a non-symbolic continuous shell can reach, or does closing the gap require a discrete-symbolic shell interface? |
| **lineage** | Fodor, *The Language of Thought* (Crowell/Harvard, 1975); Fodor and Pylyshyn, "Connectionism and Cognitive Architecture", *Cognition* (1988) (VERIFIED via SEP LoTH entry). |
| **operationalization** | Continuous-latent shell vs discrete-VQ (symbol-like) shell at matched capacity/compute on held-out-combination systematicity. Strong-LoT predicts the discrete shell strictly exceeds the continuous shell off-ceiling. Requires a D3-certified separable, non-ceilinged regime (the standing blocker). |
| **null** | Continuous and discrete shells tie off-ceiling (symbol structure adds nothing the continuous latent lacks). |
| **control** | Matched-capacity continuous shell AND a random-codebook discrete shell at matched k (any discrete win must be structure, not discretization). Square projection banned. |
| **tier** | needs-new-cache (needs an off-ceiling systematicity regime; every current compositional bed ceilings) |
| **coverage** | Partial: c1/s6/c9 (REFUTED on synthetic), p6/s4/s7 (UNDERPOWERED, ceilinged at 1.0). Strong-LoT framing not isolated. |
| **gap** | No off-ceiling regime where a symbolic shell could beat a continuous shell; the prediction has never been given a chance to fail. |
| **priority** | high |

### SEM-LANG-2 (high) — Language-independent thought (concepts pre-symbolic)

| field | value |
|---|---|
| **position** | Conceptual structure is fully present in the perceptual substrate before language, so a text-free encoder already carries the concepts language merely names; a symbolic layer relabels rather than creates content. |
| **question** | Are concepts a symbolic shell can decode already linearly present in the frozen latent at the same fidelity, such that the symbolic layer is a relabeling? |
| **lineage** | Core-cognition tradition: Spelke and Kinzler, "Core knowledge", *Developmental Science* 10 (2007) 89-96 (VERIFIED). Carey, *The Origin of Concepts* (Oxford UP, 2009) (VERIFIED). Framed against Fodor's innate-mentalese view. |
| **operationalization** | Compare linear-probe decodability of a concept from raw frozen latents vs the best a trained symbolic shell achieves. Language-independence predicts near-parity. Non-vacuous control = frozen-random-init ViT at matched resolution; concept must beat that floor. |
| **null** | The symbolic shell decodes concepts no better than a linear probe on raw latents (delta < +0.05) AND raw latents beat frozen-random. |
| **control** | Frozen-random-init ViT at matched resolution; a shuffled-label shell floor. |
| **tier** | cpu-now |
| **coverage** | Partial: p1_concept_no_labels (REFUTED, zero purity gain over frozen-random) speaks to concept formation but not the language-adds-nothing-over-raw-latent comparison. |
| **gap** | No head-to-head isolating symbolic-shell vs raw-latent linear probe with the frozen-random floor. |
| **priority** | high |

### SEM-LANG-3 (flagship) — Linguistic relativity (weak Whorf, category-boundary warp)

| field | value |
|---|---|
| **position** | A language substrate carves the latent along its own category boundaries, warping which distinctions are cheap to decode, so a language-derived encoder and V-JEPA should disagree about where category borders fall on identical content. |
| **question** | Does a language-derived representation warp near-boundary discriminability (within-category pairs harder, cross-category easier) relative to V-JEPA, in a way a matched non-linguistic control does not? |
| **lineage** | Whorf, *Language, Thought, and Reality* (MIT Press, 1956) (VERIFIED). Winawer et al., "Russian blues reveal effects of language on color discrimination", *PNAS* 104 (2007) 7780-7785 (VERIFIED). |
| **operationalization** | Content with a known cross-lingual split (goluboy/siniy blue border). On cached features, measure near-boundary vs within-category discriminability for a frozen vision-language encoder vs V-JEPA. Whorf predicts a boundary warp in the language-derived space absent in V-JEPA. |
| **null** | The language-derived encoder shows no boundary-aligned warp beyond a non-linguistic second encoder (DINOv2) at matched dimensionality. |
| **control** | DINOv2 at matched rank as the no-language comparison; a random-rotation of the language space (verbal-interference analogue). |
| **tier** | needs-new-cache (needs a language-grounded encoder cache on the color-border stimuli) |
| **coverage** | Not covered. AT1/AL2/AL3 test alignability, never a category-boundary warp. |
| **gap** | The one place a text-free substrate makes Whorf directly testable is unposed; requires a language-native encoder cache. |
| **priority** | flagship |

### SEM-LANG-4 (medium) — Language as compression (rate-distortion)

| field | value |
|---|---|
| **position** | Language's value is lossy dimensionality reduction of perceptual states into transmissible codes, so forcing thought through a discrete bottleneck should HURT on high-perceptual-detail tasks and pay off only when detail is irrelevant. |
| **question** | At matched compute, does a discrete bottleneck degrade high-detail tasks more than abstract tasks, tracing a rate-distortion frontier rather than a uniform effect? |
| **lineage** | Zaslavsky, Kemp, Regier, Tishby, "Efficient compression in color naming and its evolution", *PNAS* 115 (2018) 7937-7942 (VERIFIED, Information Bottleneck). |
| **operationalization** | VQ/discrete-bottleneck shell vs continuous shell at matched compute across tasks varying in perceptual granularity; plot usability vs bits. Compression predicts a monotone detail-dependent penalty crossing zero when detail is irrelevant. i9_vq_rate_distortion is the existing lane. |
| **null** | Discrete and continuous shells tie at all detail levels (bottleneck is free), OR usability is flat in rate (i9 null). |
| **control** | Random-codebook and k-means codebooks at matched rate; matched-compute continuous shell. |
| **tier** | cpu-now |
| **coverage** | Partial: i9_vq_rate_distortion and p6/s4 test discrete vs continuous but not the detail-dependence. |
| **gap** | No task-granularity axis crossed with the bottleneck. |
| **priority** | medium |

### SEM-LANG-5 (high) — Language as scaffold (label-feedback, top-down)

| field | value |
|---|---|
| **position** | Language temporarily augments perception via top-down label-feedback and can be dialed up or down, so an added linguistic signal should sharpen near-threshold categorical decisions and vanish under a verbal-interference analogue. |
| **question** | Does injecting a label/language signal improve near-threshold categorization more than an equally informative non-linguistic cue, and does interference selectively remove the linguistic boost? |
| **lineage** | Lupyan and Bergen, "Linguistically Modulated Perception and Cognition: The Label-Feedback Hypothesis", *Frontiers in Psychology* 3:54 (2012) (VERIFIED). |
| **operationalization** | Add a label embedding as auxiliary shell input on hard-vs-easy discriminations; compare to a matched-information non-linguistic cue. Scaffold predicts a label-specific boost on hard cases, removable by a verbal-interference ablation but not a non-linguistic ablation. |
| **null** | The label cue helps no more than a matched-entropy non-linguistic cue, OR the boost is uniform across difficulty. |
| **control** | A non-linguistic cue matched on mutual information; a shuffled-label channel; difficulty-stratified. |
| **tier** | needs-new-cache (needs a label/text embedding cache aligned to the discrimination stimuli) |
| **coverage** | Not covered. No experiment injects a linguistic cue as a top-down modulator. |
| **gap** | The entire top-down label-feedback direction is absent (blind-spot 9 via Lupyan). |
| **priority** | high |

### SEM-LANG-6 (exploratory, OPEN) — Language as the medium of thought itself

| field | value |
|---|---|
| **position** | For a distinguished class of contents (abstract, cross-domain, meta), there is no non-linguistic vehicle, so those contents should be representable only through a symbolic shell and absent from the raw latent at any readout. |
| **question** | Is there a class of abstract/relational contents a symbolic shell can support but that are not decodable from the raw latent by ANY probe? |
| **lineage** | Carruthers, "The cognitive functions of language", *BBS* 25 (2002) 657-726 (VERIFIED). |
| **operationalization** | Curate candidate constitutively-linguistic contents (logical connectives, cross-domain relations). Test whether ANY probe recovers them from raw latents vs whether only a symbolic shell supports them. Hard to make non-vacuous (absence-of-decodability confounds with probe weakness). |
| **null** | Every content a symbolic shell supports is also decodable from raw latents above frozen-random (nothing is constitutively linguistic). |
| **control** | Frozen-random floor for the raw-latent probe; a capacity-matched non-symbolic shell; a strong positive control on the probe (mandatory, since proving un-decodability risks confusing constitutive absence with probe failure). |
| **tier** | open-philosophical |
| **coverage** | Not covered, hard to operationalize. |
| **gap** | The strongest "language is thought" claim has no clean test; flagged OPEN. |
| **priority** | exploratory |

### SEM-LANG-7 (flagship) — Grounding asymmetry (Bender-Koller)

| field | value |
|---|---|
| **position** | A form-only language system cannot in principle acquire reference, so a language-only encoder should fail a use/reference criterion on the same content that perceptually-grounded V-JEPA passes, exposing a form-vs-meaning gap. |
| **question** | On a use-based (not readout-based) reference criterion, does a form-only encoder fail where the grounded substrate succeeds, isolating grounding rather than raw information content? |
| **lineage** | Bender and Koller, "Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data", ACL 2020, aclanthology.org/2020.acl-main.463 (VERIFIED, the octopus argument). |
| **operationalization** | Define a use criterion (perturb a latent shape direction, check the shell's behavior changes as a referential state would). Compare a form-only text encoder vs V-JEPA on identical grounded content. Grounding predicts V-JEPA passes where the form-only encoder cannot. |
| **null** | The form-only encoder passes the use/reference criterion as well as V-JEPA, OR neither passes (criterion vacuous). |
| **control** | The DR4/s10 anti-self-deception guard: a frozen-random encoder must FAIL the use-test; matched information content so the gap is grounding, not bits. |
| **tier** | needs-new-cache (needs a text-encoder cache + DR4-style intervention lane on paired content) |
| **coverage** | Partial: DEEP_RESEARCH sec4 cites the debate; s1/s10 test grounding as decodability, which the octopus argument says is not the point. |
| **gap** | Bender-Koller is invoked as framing but never run as a use-asymmetry experiment. |
| **priority** | flagship |

### SEM-LANG-8 (high) — Conceptual-role internalism (anti-Bender-Koller)

| field | value |
|---|---|
| **position** | Meaning is fixed by internal relational structure, not external reference, so a form-only encoder's meaning is measurable purely from how its internal states relate, and it can match a grounded substrate on relational/inferential structure without grounding. |
| **question** | Does a form-only encoder recover the same inferential/relational structure (analogy, entailment geometry) as the grounded substrate, showing meaning-as-conceptual-role that needs no reference? |
| **lineage** | Piantadosi and Hill, "Meaning without reference in large language models", arXiv 2208.02957 (2022) (VERIFIED). |
| **operationalization** | Measure relational structure (RSA / kNN-topology of inferential relations) inside a form-only encoder vs V-JEPA on matched content. Internalism predicts a match on relational structure even while failing the reference use-test of SEM-LANG-7. Score by kNN-overlap under a permutation null (AT1-relrep), never global R^2. |
| **null** | The form-only encoder's relational structure ties a random-map-of-equal-rank floor, OR it matches V-JEPA only because both tie the random floor. |
| **control** | Random-map-of-equal-rank and shuffled-fit floors; an isotropic-Gaussian target at matched d,n. |
| **tier** | needs-new-cache (needs a text-encoder cache) |
| **coverage** | Partial: AL2-kNN/AT1-relrep are the right machinery but never applied to a form-only-vs-grounded contrast. |
| **gap** | The internalist and grounding positions form a decisive PAIR (form-only passes conceptual-role, fails use-reference); neither half is posed. |
| **priority** | high |

### SEM-LANG-9 (high) — Discrete-token reasoning is a bottleneck (Coconut)

| field | value |
|---|---|
| **position** | Forcing reasoning through serialized language tokens discards parallel latent structure, so continuous latent reasoning should match or beat token-serialized reasoning at matched compute on search-heavy problems. |
| **question** | At matched compute, does continuous-latent iterative reasoning beat a discrete-token-serialized reasoning shell on planning/search, or does serialization tie or win? |
| **lineage** | Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (Coconut), arXiv 2412.06769 (2024) (VERIFIED). |
| **operationalization** | Continuous-latent refiner vs discrete-VQ-serialized reasoning shell at matched compute on a planning task with real search depth. Coconut predicts continuous >= discrete, widening with branching factor. Must fix the p9/ex17 confound: match compute AND depth. |
| **null** | Continuous and discrete-serialized shells tie at matched compute, OR both tie the untied-depth control (the p9/s4 null). |
| **control** | Matched-compute AND matched-unrolled-depth untied control; a shuffle-the-latent-trace ablation (DR2). |
| **tier** | cpu-now |
| **coverage** | Partial: p9/s4 (REFUTED/tie) tested this on easy regimes where iteration tied depth; DR2/DR11 are the live lanes. |
| **gap** | Prior tests used regimes with no real search depth, exactly where Coconut predicts no gap; the branching-factor axis is unswept. |
| **priority** | high |

### SEM-LANG-10 (flagship) — A frozen LLM's hidden states are a distinct perspective

| field | value |
|---|---|
| **position** | A frozen LLM's hidden states are a genuinely different perspective from V-JEPA on the same content, not a relabeling: the two should carry decorrelated correctness information, licensing language as a distinct reasoning MODE (PR1). |
| **question** | On identical content describable in both video and text, do a frozen LLM's hidden states and V-JEPA's latents make decorrelated errors, so combining them beats the best single one at matched capacity? |
| **lineage** | Huh et al., "The Platonic Representation Hypothesis", ICML 2024, arXiv 2405.07987 (VERIFIED, the convergence prior). Its Aristotelian rebuttal (Groger, Wen, Brbic, arXiv 2602.14486, 2026, VERIFIED per corpus): global convergence dies under permutation null, local topology survives. |
| **operationalization** | WS1/PR1 across V-JEPA and a frozen LLM on parallel content: does cross-substrate DISAGREEMENT predict error better than either substrate's own confidence, with an invertible-remap vacuity guard. Distinct perspective => decorrelated errors; relabeling => LLM ties an invertible remap of V-JEPA. |
| **null** | The LLM's correctness signal is an invertible remap of V-JEPA's (agreement AUROC <= best single confidence; N2 remap guard fires). |
| **control** | Invertible-remap vacuity guard (WS1 N2); matched-capacity concat-MLP fusion floor; parallel content. |
| **tier** | needs-new-cache (needs an LLM-hidden-state cache on parallel content; the Qwen-textified cache is a LABEL-FREE PIXEL-DERIVED textification (color grid + brightest-cell position, already paired to the clips, lacking SHAPE which decodes at chance here), not full parallel LLM states on the same referents) |
| **coverage** | Partial: WS1 is specified for V-JEPA + DINOv2 (vision-vision); PR1 licenses reasoning modes; neither uses a LANGUAGE substrate as the second perspective. |
| **gap** | The core MoP claim that language is a distinct perspective is only tested vision-vs-vision. |
| **priority** | flagship |

### SEM-LANG-11 (high) — Pragmatics is where a linguistic perspective adds value

| field | value |
|---|---|
| **position** | The same latent should mean different things under different task contexts, and a routing shell can induce this context-sensitivity the way linguistic pragmatics does, unlike a fixed perceptual read. (EXTENSION of SEM-COMM-4 with the linguistic-vs-non-linguistic-cue contrast.) |
| **question** | Can the shell's read of a single fixed latent shift with routing/task context, and does a language-conditioned context induce this more than a non-linguistic context? |
| **lineage** | Frank and Goodman, "Predicting Pragmatic Reasoning in Language Games", *Science* 336 (2012) 998, the Rational Speech Act model (VERIFIED). |
| **operationalization** | One fixed latent under two task contexts (two routing prompts); measure whether the shell's decoded meaning shifts appropriately; test whether a linguistic context-cue induces more shift than a matched non-linguistic cue. |
| **null** | The shell's read of the fixed latent is invariant to context (delta < chance). |
| **control** | A non-linguistic context cue matched on information; a scrambled-context floor; guard that the shift is appropriate, not noise. |
| **tier** | cpu-now |
| **coverage** | Not covered (blind-spot 3). |
| **gap** | Pragmatic meaning is unposed even though the router is exactly the mechanism that could induce it. |
| **priority** | high |

### SEM-LANG-12 (medium) — Language as a module-integrator (Carruthers weak thesis)

| field | value |
|---|---|
| **position** | Language's distinctive function is combining outputs of otherwise-encapsulated perceptual modules, so a language-structured shell should improve CROSS-modal integration more than any single-currency task. |
| **question** | Does a language-structured integration shell beat an unstructured fusion shell specifically on cross-currency tasks while tying it on single-currency tasks? |
| **lineage** | Carruthers, "The cognitive functions of language", *BBS* 25 (2002) 657-726, the defended weaker thesis (VERIFIED). |
| **operationalization** | Symbolic/combinatorial fusion shell vs matched-capacity concat-MLP across single-currency and cross-currency (video+audio, video+text) tasks. Integrator role predicts a symbolic advantage ONLY on cross-currency binding. |
| **null** | The symbolic fusion shell ties the concat-MLP on cross-currency tasks, OR any advantage appears equally on single-currency tasks. |
| **control** | Matched-capacity unstructured concat-MLP; single-vs-cross-currency contrast; frozen-random second substrate. |
| **tier** | needs-new-cache (needs a second-currency cache paired to video, e.g. audio-video; wav2vec2 exists but is not paired on the same clips) |
| **coverage** | Partial: A14/WS1 test structured vs concat fusion but frame it as shared-code, never as the single-vs-cross-currency dissociation. |
| **gap** | The integrator hypothesis makes a dissociation prediction the workspace experiments do not isolate. |
| **priority** | medium |

### SEM-LANG-13 (medium) — Linguistic determinism (strong Whorf, expected to FAIL)

| field | value |
|---|---|
| **position** | Language DETERMINES available conceptual distinctions, so distinctions unlexicalized in a language substrate should be undecodable from a language-derived encoder even when plainly present in V-JEPA. The preregistered-to-fail pole. |
| **question** | Are perceptual distinctions a language leaves unlexicalized actually ABSENT (undecodable) from a language-derived encoder, or merely harder, while remaining fully decodable from the perceptual substrate? |
| **lineage** | The strong (determinism) reading of Whorf (1956) (VERIFIED), now rejected by consensus (SEP Whorfianism); the weak version survives per Winawer 2007. |
| **operationalization** | For a distinction unlexicalized in language L, test decodability from an L-derived encoder vs V-JEPA. Determinism predicts chance decoding from the L-encoder; weak-relativity predicts above-chance-but-degraded. This is the null we EXPECT to reject. |
| **null** | Preregistered expectation: the unlexicalized distinction IS decodable above chance from the language-derived encoder (determinism false; this null expected to hold). |
| **control** | V-JEPA decodability of the same distinction (the present-in-perception anchor); frozen-random floor; a distinction that IS lexicalized as the positive control. |
| **tier** | needs-new-cache (needs a language-grounded encoder cache) |
| **coverage** | Not covered. |
| **gap** | Pairing this with SEM-LANG-3 gives a graded relativity axis with a built-in floor; the pair distinguishes "language warps" from "language determines". |
| **priority** | medium |

---

## 4. CODE as a perspective

Code as a formal, executable, VERIFIABLE language of thought, distinct from natural language by having ground-truth semantics. **Verification-forced correction (applies to this whole section):** the domain brief tiered most of these cpu-now on a false premise. There is NO DSL, NO executor, and NO scene with count/relation/exists factors anywhere in the repository. The programmatic_reference cache is a 25-d vector = one-hot(shape,5) + one-hot(color,5) + 15 continuous nuisance columns (position, scale, rotation, velocity), i.e. exactly TWO crisp factor slots (shape, color), not the {shape, color, count, relation} the positions name. The pooled substrate factors shape from color off-ceiling (the 0.708 held-out result) but exposes nothing richer. Therefore every code position that assumes an executor scoring a shell-emitted program is split into (a) the TOOLING BUILD (DSL + executor + a count/relation cache), marked `needs-new-cache`, and (b) the actual test, marked `cpu-now-after-(a)`. None is cpu-now today, because the scorer the corpus's null-lane diagnosis calls for is still assumed, not instantiated.

### SEM-CODE-1 (flagship) — Executable code is the scorer ex18 lacked

| field | value |
|---|---|
| **position** | Code is a distinct language of thought because it has ground-truth semantics: a program runs and produces the right output or it does not. For a latent-native model, code is the only candidate perspective whose meaning is checkable without a human in the loop. |
| **question** | Can a tiny shell emit a program in a small DSL whose EXECUTION on the perceptual state is scored by ground truth, and does that execution-checkable channel carry a real correction signal where the non-executable verifier (ex18) did not? |
| **lineage** | Fodor 1975 / Fodor-Pylyshyn 1988 (VERIFIED via SEP); the executable-semantics distinction sharpened by RLVR code generation (CodeRL, Le et al., NeurIPS 2022, VERIFIED). |
| **operationalization** | (a) BUILD: a 5-primitive DSL (count, filter-by-attribute, compare, exists, project), an executor, AND a scene cache with count+relation factors (the current 5x5 shape-color cache cannot bind more than shape/color). (b) TEST: shell emits a short program executed against the true scene-program vector; metric = does execution-gated revision beat single-shot at matched compute AND beat a shuffled-executor control? |
| **null** | Execution-gated revision ties single-shot at matched compute, and the true executor ties a shuffled executor: the pooled substrate does not expose factor slots crisp enough for a discrete program to bind to (the p9/ex17/ex18 null generalized to the executable channel). |
| **control** | Shuffled-executor (program scored against a permuted scene, matched pass-rate); single-shot at matched compute; random-init same-arch ViT-L slots at matched 256px for the decode-to-slots step (NOT frozen_random_projection). |
| **tier** | needs-new-cache (DSL + executor + count/relation cache) THEN cpu-now |
| **coverage** | Gap. ex18/DR7/DR9 test a LEARNED verifier; nothing executes a symbolic program against ground-truth scene semantics. The programmatic_reference cache is a passive AT4 ceiling (shape+color only), never an executor. |
| **gap** | The entire executable-verification idea is untested, and the artifacts it needs (DSL, executor, richer cache) do not exist. |
| **priority** | flagship |

### SEM-CODE-2 (flagship) — An executable verifier may flip the null test-time-compute lane

| field | value |
|---|---|
| **position** | The null test-time-compute lane died because search/verify/revise needs a cheap reliable scorer and the perceptual verifier (ex18) was decorative. Executable code supplies that scorer, so the null lane may be a substrate-of-verification problem, not a reasoning-in-latents problem. |
| **question** | Does adding an EXECUTABLE verifier to the exact refine/verify loop that was null in ex17/ex18 flip the matched-compute verdict, attributable to verifiability rather than extra compute? |
| **lineage** | Chollet, "On the Measure of Intelligence", arXiv 1911.01547 (2019) (VERIFIED); RLVR unit-test reward (CodeRL, VERIFIED); the corpus 04_reasoning_program precondition (cheap reliable scorer + best branch not directly findable). |
| **operationalization** | (a) BUILD DSL+executor+richer cache (as SEM-CODE-1). (b) TEST: re-run ex17/ex18 with the executor replacing the trained verifier; shell proposes K candidate programs, each executed against ground truth, passing one kept. Budget TOTAL FLOPs including discarded candidates and execution. Metric: accuracy vs single-shot at matched total compute; verifier-AUROC of pass/fail vs true correctness. |
| **null** | Even with a perfect executor, execute-and-keep-best ties single-shot at matched total compute, because the single forward net already finds the passing program directly (precondition (b) fails), extending the ex17 gain=0.0 result to the executable regime. |
| **control** | Matched TOTAL FLOPs single-pass MLP; a NON-verifiable arm (keep a random candidate) to isolate the executor's contribution; task-difficulty calibration (D3) so the regime is not degenerate. |
| **tier** | needs-new-cache (DSL + executor + richer cache) THEN cpu-now |
| **coverage** | Gap. MP4/MP7/DR6/DR7 budget total compute but every scorer is a LEARNED verifier, never a ground-truth executor. |
| **gap** | No experiment gives the verify-revise loop a scorer reliable BY CONSTRUCTION; the null lane's cause (unreliable scorer vs no headroom) is unidentified, and these positions still assume the executor rather than building it. |
| **priority** | flagship |

### SEM-CODE-3 (high) — Program induction as a routable mode (PR1)

| field | value |
|---|---|
| **position** | Program induction makes errors decorrelated from the perceptual reactive mode, so it earns a router slot (PR1) even if it does not win alone. Its distinctive competence is exact held-out-combination extrapolation. |
| **question** | Do a program-induction mode and the reactive perceptual head make per-sample errors decorrelated enough that an oracle router beats the best single mode, on held-out compositional cells where the perceptual mode is weakest? |
| **lineage** | Ellis et al., DreamCoder, PLDI 2021 / *Phil Trans A* 2023 (VERIFIED); Chollet ARC (VERIFIED); PR1 error-disjointness criterion. |
| **operationalization** | (a) BUILD DSL+executor over decoded slots. (b) TEST: on the compositional_under_nuisance cache (5x5, diagonal held-out where V-JEPA scores 0.708), run reactive head vs a program-induction head; compute per-sample loss correlation and the oracle-selection bound over the best single mode, SPLIT by seen vs held-out cells. |
| **null** | Oracle-router gain over the best single mode is inside seed spread everywhere; the program mode adds nothing on held-out cells because the perceptual mode already factors shape from color. Modes redundant, not complementary. |
| **control** | Random-init same-arch ViT-L slots at matched 256px feeding both modes; shuffled per-sample-loss pairing; matched compute per mode. |
| **tier** | needs-new-cache (DSL + executor over slots) THEN cpu-now (the 5x5 cache suffices for the split, but the program mode still needs an executor that does not exist) |
| **coverage** | Gap. PR1 licenses reactive/refine/plan/verify; a program-induction mode is not among them. |
| **gap** | Code as a routable mode is absent from the mode taxonomy. |
| **priority** | high |

### SEM-CODE-4 (high) — Code exposure reshapes abstraction (DSL as inductive bias)

| field | value |
|---|---|
| **position** | A shell trained to emit programs should generalize compositionally BETTER than a shell trained on flat labels, because the DSL's typed/recursive structure is an inductive bias the flat head lacks. |
| **question** | Does a programmatic (DSL-structured) target reshape the shell so held-out-combination extrapolation improves over a flat-label shell of matched capacity, at matched compute? |
| **lineage** | Ellis 2021 library learning (VERIFIED); Fodor-Pylyshyn systematicity (VERIFIED); the code-improves-reasoning trend (Code-to-Think survey arXiv 2502.19411, PLAUSIBLE; individual MathCoder papers UNVERIFIED). |
| **operationalization** | (a) BUILD DSL+executor. (b) TEST: two matched-capacity shells over identical V-JEPA slots, shell F predicts a flat label, shell P predicts a typed expression tree whose execution yields the same label. Train on SEEN cells, evaluate held-out-combination. Metric: held-out P minus F at matched params/compute. |
| **null** | P ties F on held-out combinations: the DSL target adds no compositional bias beyond what the pooled substrate supplies (0.708 already substrate-carried), localizing compositionality to the substrate not the target format. |
| **control** | Matched-capacity flat shell; a SCRAMBLED-DSL arm (same tree target, types removed / primitives relabeled) to isolate typed structure; random-init-ViT slots at matched resolution. |
| **tier** | needs-new-cache (DSL + executor) THEN cpu-now |
| **coverage** | Gap. The corpus tests whether the SUBSTRATE factors, never whether the shell's OUTPUT format changes generalization. |
| **gap** | The "code reshapes abstraction" claim is untested. |
| **priority** | high |

### SEM-CODE-5 (high) — Code-as-planning / library learning (ex2 extension)

| field | value |
|---|---|
| **position** | ex2 latent planning is proto-programmatic: making the action sequence an EXPLICIT executable program should enable REUSE of a discovered sub-plan across tasks (library learning) that the implicit rollout cannot. |
| **question** | Does an explicit executable program beat the implicit MPC rollout at matched compute, and does the explicit form gain cross-task sub-plan reuse? |
| **lineage** | Ellis DreamCoder library-growth (PLDI 2021, VERIFIED); Chollet (VERIFIED). |
| **operationalization** | **Verification-forced caveat:** ex2 does NOT run on V-JEPA latent dynamics. `close_ex2_planning.py` uses `_true_dynamics_params` / `_step_true`, a hand-written synthetic 8-d-action toy with a small learned `_DynamicsModel`; it touches no V-JEPA latent. So "executed by the frozen predictor's action-conditioned rollout" describes machinery that does not exist. Two honest readings: (i) restrict the claim to the toy (then it says NOTHING about the substrate); (ii) mark the substrate-grounded version `needs-new-cache` (needs an action-conditioned V-JEPA rollout that is not built). The library-reuse metric is fine in principle but inherits the toy-vs-substrate caveat. |
| **null** | Explicit-program planning ties implicit MPC at matched compute AND the growing library gives no reuse advantage (task N+1 return with library == without). |
| **control** | Implicit MPC at matched compute (ex2 arm); action-shuffle (ex2 control); a NO-LIBRARY arm; matched TOTAL compute including library-search cost. |
| **tier** | needs-new-cache (action-conditioned V-JEPA rollout) for the substrate claim; the toy version is cpu-now but substrate-silent |
| **coverage** | Partial: ex2 planning is a surviving positive, but on a synthetic toy, not the substrate. Library learning has no analogue anywhere. |
| **gap** | Library learning is absent; ex2 is one-shot per task and toy-bound. |
| **priority** | high |

### SEM-CODE-6 (medium) — Program-length as a substrate-typing metric (Church-Turing)

| field | value |
|---|---|
| **position** | If the substrate is a genuine currency, any computable factor-transformation over its decoded state should be expressible as a program the shell can search for; factors that resist ALL short programs mark a real substrate entanglement bound distinct from a decodability ceiling. |
| **question** | Is there a factor-transformation decodable in principle but NOT expressible as any short program over decoded slots, and does the program-length threshold mark a bound distinct from a ceiling? |
| **lineage** | Church-Turing / computationalism (Turing 1936; Church 1936; SEP Church-Turing Thesis and CTM) with the Church-Turing FALLACY caveat (Copeland) so this is a HYPOTHESIS-source (VERIFIED); DreamCoder MDL priors (Ellis 2021, VERIFIED). |
| **operationalization** | (a) BUILD DSL+executor. (b) TEST: program search for the shortest program reproducing each factor, MDL cap, real V-JEPA slots vs random-init-ViT slots. A factor needing a much SHORTER program off V-JEPA slots is pre-factoring evidence; a factor needing an unboundedly long program off BOTH is an entanglement bound (AT4 can express it while the substrate cannot). |
| **null** | Every factor needs the same program length off V-JEPA and random-init slots (no pre-factoring), OR every factor is trivially short off both (ceiling). |
| **control** | Random-init same-arch ViT-L slots at matched 256px; AT4 programmatic_reference as the ceiling that CAN express the transform; MDL cap fixed across arms. |
| **tier** | needs-new-cache (DSL + executor + richer factors than shape/color) THEN cpu-now |
| **coverage** | Gap. Decodability is measured everywhere; program-length-to-solve is never measured. |
| **gap** | An MDL/program-search metric distinguishing entanglement bounds from ceilings does not exist. |
| **priority** | medium |

### SEM-CODE-7 (high) — Execution fixes meaning independently of convention (signalling)

| field | value |
|---|---|
| **position** | Code is where a private idiolect can be forced into a shared language, because a program's meaning is fixed by EXECUTION, not convention. Two shells that must agree on an executable program have a convention-independent shared semantics. (EXTENSION of SEM-COMM-2 with an executable channel.) |
| **question** | Can two independently seeded shells NEGOTIATE a shared DSL program through a channel to solve a referential task above a no-shared-history floor, where cross-seed probe transfer (p5/s5/y3) was at/below the frozen-random floor? |
| **lineage** | Lewis, *Convention* (1969, VERIFIED); Skyrms, *Signals* (2010, VERIFIED); RLVR execution-fixes-meaning (VERIFIED). |
| **operationalization** | (a) BUILD DSL+executor. (b) TEST: sender emits a DSL program for a target scene; receiver executes it against a candidate set and picks the referent. Both train jointly on referential success. Metric: accuracy above a no-shared-history floor and a frozen-random-init receiver. |
| **null** | Two shells fail to reach referential success above the no-shared-history floor even with an executable channel: the pooled substrate lacks slots stable enough across seeds for an executable convention to bind (the p5/s5/y3 result holds even when meaning is execution-fixed). |
| **control** | No-shared-history floor; frozen random-init receiver; matched-channel-capacity baseline (continuous vector of equal bits) to test whether DISCRETE executable structure, not raw bandwidth, carries the win. |
| **tier** | needs-new-cache (DSL + executor) THEN cpu-now |
| **coverage** | Gap. p5/s5/y3 test one shell's code stability; no sender-receiver signalling-game exists, and none uses execution to fix meaning. |
| **gap** | This is a sharper (executable) version of SEM-COMM-2, still requiring an executor that does not exist. |
| **priority** | high |

### SEM-CODE-8 (medium) — Recursion as a genuine capability-density gain

| field | value |
|---|---|
| **position** | Recursion and typed structure are what natural language and the perceptual substrate lack; a shell with a RECURSIVE DSL primitive should solve tasks with unbounded structure that any fixed-depth perceptual head cannot, marking recursion as capability-density gain, not unrolled depth. |
| **question** | On a task with unbounded recursive structure, does a shell with a recursive primitive beat a fixed-depth head at matched compute, attributable to recursion rather than more compute/depth? |
| **lineage** | Fodor-Pylyshyn productivity (VERIFIED); Chollet ARC recursion/iteration priors (VERIFIED); DreamCoder recursive primitives (VERIFIED); the corpus matched-compute discipline. |
| **operationalization** | (a) BUILD DSL with a recursive primitive + executor + a task with per-sample hidden recursion depth. (b) TEST: recursive shell R (fold/until to data-dependent depth) vs fixed-depth D. Metric: accuracy vs depth, R minus D, at matched AVERAGE and matched MAX compute. A win at matched-average vanishing at matched-max is the honest adaptive-allocation (MP5) framing. |
| **null** | R ties D at matched max AND matched average compute: recursion is unrolled depth with a halt (the y1/n9 no-fixed-point result extends to the recursive primitive). |
| **control** | Fixed-depth D at matched max AND average compute; a NON-recursive equal-primitive-count DSL arm; action/step-shuffle; noisy-TV guard if halting is uncertainty-gated. |
| **tier** | needs-new-cache (DSL + executor + recursive-structure task) THEN cpu-now |
| **coverage** | Gap. n9/y1 test implicit latent iteration (no convergence); nothing tests an explicit recursive primitive. |
| **gap** | Recursion as an executable primitive is absent. |
| **priority** | medium |

### SEM-CODE-9 (high) — Execute-and-check as the router's reliable-scorer mode (MP4)

| field | value |
|---|---|
| **position** | The verifiability of code builds the reliable-scorer precondition MP4 depends on: a router over reasoning primitives can only exploit error diversity if one mode carries a trustworthy self-assessment, and an execute-and-check mode is correct by construction, unlike every learned confidence signal that was null (e4, ex18). |
| **question** | When an execute-and-check mode is added to the MP4 router, does per-sample dispatch to it lift routed accuracy above the best fixed mode at matched mean compute, where confidence-gated routing over learned signals was null? |
| **lineage** | DreamCoder recognition-model routing (Ellis 2021, VERIFIED); Chollet (VERIFIED); MP4 synthesis-router thesis; e4 confidence-gating null; the reliable-scorer precondition. |
| **operationalization** | (a) BUILD DSL+executor. (b) TEST: MP4 router over {reactive, iterative-refine, latent-plan, execute-and-check-program}; the program mode returns a HARD pass/fail; router gates on it. Metric: routed accuracy vs best fixed mode at matched MEAN FLOPs (counting execution and discarded programs); ablate the program mode to confirm it is load-bearing. |
| **null** | Adding the mode does not lift routed accuracy above the best fixed mode: either it passes only on inputs reactive already solves (PR1 fails), or execution passes are uncorrelated with true correctness on the pooled substrate (executor cannot bind to crisp-enough slots). |
| **control** | Best fixed single mode at matched mean compute; equal-weight ensemble of the same modes (H2 control); the pass-signal REPLACED by a shuffled pass-signal; matched mean FLOPs including discarded work. |
| **tier** | needs-new-cache (DSL + executor) THEN cpu-now |
| **coverage** | Gap. MP4's mode set is {reactive, refine, plan, verify} where verify is LEARNED; the execute-and-check mode with a correct-by-construction pass signal is not in the set. |
| **gap** | The router has never been given a mode with a trustworthy self-assessment. |
| **priority** | high |

### SEM-CODE-10 (medium) — Designed executable code grounds where learned VQ did not

| field | value |
|---|---|
| **position** | A DSL whose primitives are DEFINED by execution against world factors should ground (track a world variable under intervention) where the learned VQ code did not (s1/s10 refuted), because grounding for a program is enforced by execution, not learned from co-occurrence. |
| **question** | Does a hand-designed DSL whose tokens are execution-defined predicates ground under intervention where the learned VQ codebook was PASS-VACUOUS? |
| **lineage** | Harnad 1990 (VERIFIED); s7 learned-vs-designed line; RLVR execution-defines-meaning (VERIFIED); s10 anti-self-deception meta-test. |
| **operationalization** | (a) BUILD a DSL of checkable predicates over ground-truth factors (is-red, count>2) + executor. (b) TEST: intervene on a world factor and test whether the corresponding token's execution-value tracks the intervention (use-based, not readout). Compare against the learned VQ code's s10-style score with the s10 PASS-VACUOUS check applied identically. |
| **null** | Designed DSL tokens track interventions no better than the learned VQ code once the s10 frozen-random control is applied: grounding-by-execution is vacuous on the pooled substrate (the s10 result is about the substrate's slots, not the code's origin). |
| **control** | The s10 anti-self-deception protocol applied identically; intervention vs null intervention; decoded slots from random-init-ViT at matched resolution. |
| **tier** | needs-new-cache (DSL + executor + a count>2 predicate needs a count factor the cache lacks) THEN cpu-now |
| **coverage** | Partial: s1/s7/s10/ex16 test learned codes (PASS-VACUOUS). Designed-executable code tested by intervention is untested. |
| **gap** | Grounding has only been tested for emergent codes by readout. |
| **priority** | medium |

### SEM-CODE-11 (exploratory) — Executable preconditions as principled abstention (normativity)

| field | value |
|---|---|
| **position** | A typed program carries its own preconditions (types, assertions), so an executable perspective can REFUSE to answer when preconditions fail, giving principled abstention the corpus's confidence signals (e4, null) could not. Normativity as a type/precondition violation. |
| **question** | Can a typed DSL program's precondition-failure separate a MISAPPLIED concept (type error) from mere low accuracy, achieving selective-prediction gains the learned-uncertainty gates could not? |
| **lineage** | Fodor-Pylyshyn typed compositional structure (VERIFIED); DreamCoder typed primitives (VERIFIED); the normativity blind spot (OPEN); e4 uncertainty-gating null. |
| **operationalization** | (a) BUILD a typed DSL + executor. (b) TEST: on OOD/malformed scenes, a program that type-check-fails ABSTAINS. Metric: selective-prediction risk-coverage using type-failure vs the best learned-confidence gate (ensemble disagreement, PR4), with a noisy-TV guard. A win is lower risk at matched coverage. |
| **null** | Type-failure abstention ties the learned-confidence gate, or fails the noisy-TV guard: the pooled substrate does not surface preconditions cleanly, so a type error is indistinguishable from generic low confidence (the OPEN status stands). |
| **control** | Best learned-confidence gate (ensemble disagreement, PR4) at matched coverage; noisy-TV guard; a SHUFFLED-type control. |
| **tier** | needs-new-cache (typed DSL + executor) THEN cpu-now |
| **coverage** | Gap. Normativity is OPEN with no operationalization; type-precondition-failure as abstention is unbuilt. |
| **gap** | No test separates misapplication from low accuracy. |
| **priority** | exploratory |

### SEM-CODE-12 (exploratory, OPEN) — Programs as the representation of negation/absence

| field | value |
|---|---|
| **position** | Programs are the natural representation of negation/counterfactual content (an if-not branch, a program computing what is NOT present), the blind spot pooled perception cannot address because it encodes only present positive attributes. |
| **question** | Can an executable predicate represent ABSENCE/negation (exists-no-red, count==0) supporting correct downstream behavior, where pooled substrate and every probe test only present positive attributes? |
| **lineage** | Fodor LoT negation as a Mentalese operator (VERIFIED via SEP); Goodman-Tenenbaum probabilistic-language-of-thought (CBMM Memo 010, 2014; Goodman et al. 2015, VERIFIED); the negation/absence blind spot (OPEN, blocked on dense video). |
| **operationalization** | (a) BUILD a DSL with negation predicates + executor + a count factor. (b) TEST: a shell that must act on ABSENCE succeeds above chance, and whether success is attributable to the substrate carrying absence-information vs the executor inverting present positives. Honest scope: this is the cpu-now pooled proxy; the full occlusion/permanence version is blocked on dense video (DR1) and stays OPEN. |
| **null** | The negation predicate cannot support above-chance absence-conditioned behavior beyond trivially inverting a present-attribute readout: absence is representable only as not-present, so the blind spot is not closed by an executable predicate on pooled latents. |
| **control** | A present-positive-only baseline; random-init-ViT slots; the honest caveat that the discriminating occlusion test is blocked on DR1, so a pooled null is a lower bound not a refutation. |
| **tier** | needs-new-cache (DSL + executor + count factor) for the proxy; open-philosophical / DR1-blocked for the full test |
| **coverage** | Gap (proxy) / OPEN (full test). n8/d2 permanence are the nearest miss and are pooling-bound. |
| **gap** | No experiment represents absence as a checkable predicate. |
| **priority** | exploratory |

---

## 5. MATHEMATICS as a perspective

Is mathematical cognition a distinct faculty or reducible to language/logic; can exposure to math make the latent plastic; the a priori question; geometric vs symbolic math; whether math aligns with the vision substrate above the frozen-random floor. **Verification-forced correction (applies to this whole section):** the domain brief tiered the numerosity/geometry/ordinal/magnitude positions cpu-now, but NO numerosity, geometry, dot-array, glyph, or collision-count cache exists, and no script generates one. Every one of these needs a stimulus set RENDERED and RUN THROUGH the frozen encoder, and the live-state encoder ban blocks producing it now. So the decoding cluster (SEM-MATH-1, -2, -7, -8, -9, -10, -11, -13) is retiered `needs-new-cache`. The random-init-ViT control column and the Kim-2021 warning remain correct and are genuine strengths; they simply cannot run until the caches exist. The alignment and dissociation positions (SEM-MATH-3, -4) can reuse the AL2 machinery but still need a math substrate that is not instantiated.

### SEM-MATH-1 (flagship) — Numerosity is a native perspective

| field | value |
|---|---|
| **position** | Approximate number (set cardinality in a clip) is decodable off-ceiling from pooled V-JEPA latents, because number is a core-knowledge percept the way motion is. |
| **question** | Does approximate numerosity decode from frozen V-JEPA better than a matched-resolution random-init ViT, or is number a "free" emergent axis the random control ALREADY carries (making the substrate claim vacuous)? |
| **lineage** | Dehaene, *The Number Sense* (Oxford UP, 1997); Spelke and Kinzler 2007 (VERIFIED). The doctrinal control: Nasr, Padmanabhan, Barbour, "Number detectors spontaneously emerge", *Science Advances* (2019); Kim et al., "Visual number sense in untrained deep neural networks", *Science Advances* (2021): number-selective units arise in UNTRAINED nets (VERIFIED). |
| **operationalization** | Cache pooled latents for clips of 1..8 objects with position/size/color/spacing as nuisance. Tiny probe reads numerosity. NON-VACUOUS control: the SAME probe on a matched-arch matched-resolution random-init ViT (NOT a square projection). Report real-minus-random delta and preregistered p. |
| **null** | Real V-JEPA gives no numerosity-decoding advantage over the random-init ViT at matched resolution and probe capacity (delta <= 0 or p >= 0.05). |
| **control** | Matched-arch matched-resolution random-init ViT (Kim-2021 predicts this control is unusually strong for number, so the bar is real). Secondary: cumulative-area / density decorrelated from count. |
| **tier** | needs-new-cache (numerosity 1..8 stimulus render + frozen-encoder pass) |
| **coverage** | Gap. No numerosity probe exists; the currencies atlas never lists number. |
| **gap** | The corpus has never asked whether COUNT is decodable, and never confronted the Kim-2021 warning that the random-init control may pass (the vacuous-control trap in numeric clothing). |
| **priority** | flagship |

### SEM-MATH-2 (flagship) — Geometry native, algebra not (asymmetry)

| field | value |
|---|---|
| **position** | V-JEPA is spatial-temporal, so Euclidean-geometric properties (parallelism, right angles, symmetry, collinearity) factor off-ceiling from the latent while purely symbolic/algebraic structure does not. |
| **question** | Do core-geometry properties decode above the random-init floor by a WIDER margin than symbolic-algebraic relations, i.e. a substrate-specific geometry-over-algebra asymmetry? |
| **lineage** | Dehaene, Izard, Pica, Spelke, "Core Knowledge of Geometry in an Amazonian Indigene Group", *Science* 311 (2006), PubMed 16424341; Izard, Pica, Spelke, Dehaene, "Flexible intuitions of Euclidean geometry", *PNAS* (2011) (VERIFIED). |
| **operationalization** | Two matched probe sets over frozen V-JEPA: (a) geometric-intruder clips; (b) matched-difficulty symbolic relations. Compare real-minus-random-init delta on (a) vs (b). Difficulty calibrated with the programmatic substrate so a null on (b) is "substrate too weak", not "test too hard". |
| **null** | The real-minus-random delta on geometric properties equals the delta on symbolic relations (no asymmetry beyond overall probe strength). |
| **control** | Random-init ViT for BOTH probe sets; the programmatic/handcrafted-descriptor ceiling to bound each factor's decodability. |
| **tier** | needs-new-cache (geometric-figure + symbolic-relation render + encoder pass) |
| **coverage** | Gap. "Geometry" in the corpus only ever means "any-high-dim geometry"; no mathematical-geometry factor is probed. |
| **gap** | The single sharpest V-JEPA-specific prediction has never been posed. |
| **priority** | flagship |

### SEM-MATH-3 (high) — Math is a distinct faculty from language (dissociation)

| field | value |
|---|---|
| **position** | The axes carrying mathematical structure are separable from those aligning with linguistic structure, mirroring expert math sparing the language network. |
| **question** | Is the math-predictive subspace statistically dissociable from the language-alignment subspace, above a random rank-matched split? |
| **lineage** | Amalric and Dehaene, "Origins of the brain networks for advanced mathematics in expert mathematicians", *PNAS* 113 (2016), DOI 10.1073/pnas.1603205113 (VERIFIED). |
| **operationalization** | Fit the math-predictive subspace (directions predicting number/geometry) and the direction of best alignment to a small frozen LLM's hidden states. Measure subspace overlap (principal angles) vs a null of two random rank-matched subspaces. Dissociation = math subspace LESS aligned to the LLM axis than chance-rank predicts. |
| **null** | The math subspace overlaps the language-alignment subspace no less than two random rank-matched subspaces (no dissociation). |
| **control** | Random rank-matched subspace pairs (NOT a square projection; the metric is principal-angle overlap, not R^2). |
| **tier** | needs-new-cache (needs the math factors from SEM-MATH-1/2 AND an LLM-hidden-state cache) |
| **coverage** | Gap. AL2 tests alignment generically; nothing tests a math-vs-language dissociation within one substrate. |
| **gap** | The "math as a separate currency" thesis is asserted but never operationalized as a within-substrate dissociation. |
| **priority** | high |

### SEM-MATH-4 (high) — Math fails to align with vision above the random floor

| field | value |
|---|---|
| **position** | A symbolic/programmatic math substrate and V-JEPA share only the trivial alignment a random full-rank map gives, so math cannot be exchanged into the visual currency. |
| **question** | Does a math substrate align with frozen V-JEPA above the random-map-of-equal-rank floor on local kNN topology, or does math-vision alignment die under the permutation null the way global convergence does? |
| **lineage** | Huh et al., Platonic Representation Hypothesis, ICML 2024 (VERIFIED); its Aristotelian rebuttal (Groger, Wen, Brbic, VERIFIED per corpus). Extended to the specific math-vs-vision pair. |
| **operationalization** | Build a math substrate (LLM-emitted programs / symbolic feature vectors for the same scenes) and V-JEPA latents. Score by kNN-overlap (local topology), NOT global R^2, per AT1-relrep. Control: random map of equal rank; permutation null. Positive = local kNN overlap survives permutation for math-vision. |
| **null** | Math-vision kNN-overlap does not exceed the random-equal-rank map under permutation (math is an isolated currency, no exchange rate). |
| **control** | Random-map-of-equal-rank plus label-permutation null; square-projection alignment forbidden. |
| **tier** | needs-new-cache (needs a math substrate instantiated; atlas 1.9 lists it only as a ceiling reference) |
| **coverage** | Partial: AL2 machinery exists but the math substrate is never instantiated. |
| **gap** | The exchange-rate question is posed for audio/DINOv2/CLIP but never for math, the currency most likely non-exchangeable. |
| **priority** | high |

### SEM-MATH-5 (high) — Math exposure makes the latent plastic

| field | value |
|---|---|
| **position** | A tiny shell trained on math structure reshapes how downstream non-math tasks use the latent, evidence of moldability rather than a bolt-on probe. |
| **question** | After math-structure training, does a HELD-OUT non-math task sharing latent geometry improve beyond a matched-capacity shell trained on a non-math objective? |
| **lineage** | Piaget genetic epistemology (VERIFIED framing); Lakoff and Nunez, *Where Mathematics Comes From* (Basic Books, 2000): math built by metaphorical extension of sensorimotor structure (VERIFIED). |
| **operationalization** | Substrate frozen (doctrine). Train shell A on a math-structure task (ordinal/magnitude comparison) and shell B on a matched-capacity non-math task. Freeze each learned readout; test transfer to a third held-out task. Plasticity = A's induced latent-use transfers where B's does not. |
| **null** | Math-trained shell A gives no transfer advantage over non-math shell B at matched capacity/compute (moldability indistinguishable from added readout). |
| **control** | Matched-capacity non-math shell B; random-init-shell floor. Honest scope: because the substrate is frozen, any transfer is shell-mediated, so the claim is bounded to SHELL plasticity, not substrate plasticity. |
| **tier** | needs-new-cache (needs the math task, which needs a math stimulus cache) then studio |
| **coverage** | Gap. Plasticity work (d4/d6/n5/e3/e4/b4 NEGATIVE; PR5 the live retry) is about substrate reopening, never math-content-driven reshaping of latent USE. |
| **gap** | The "plastic through math" thesis has no operationalization; the honest version can only test SHELL plasticity on a frozen substrate. |
| **priority** | high |

### SEM-MATH-6 (high) — Math is a priori / analytic (grounding ablation)

| field | value |
|---|---|
| **position** | Math relations are decodable from the frozen latent WITHOUT empirically grounded scene content (from noise or purely structural stimuli), so math is in the geometry of the space, not in the visual grounding. |
| **question** | Can a math relation (ordering, symmetry, equality) be decoded from ungrounded structural stimuli above the random-init floor, or does decodability collapse without grounding (the empiricist prediction)? |
| **lineage** | Kant, *Critique of Pure Reason* (1781/1787): mathematics as synthetic a priori (VERIFIED as a POSITION to test). The empiricist counter: Lakoff and Nunez (2000) predicts ungrounded decoding FAILS (VERIFIED). |
| **operationalization** | Two stimulus regimes: grounded (objects in scenes) vs ungrounded (abstract dot arrays / glyphs / structured noise). Decode the same relation from each. Kant-side: ungrounded still decodes above random-init. Empiricist: ungrounded collapses to the floor. Noisy-TV guard so structured noise is not just high-variance features. |
| **null** | Ungrounded-stimulus math decoding is at the random-init floor (no a-priori structure; decodability requires grounding). |
| **control** | Random-init ViT on the SAME ungrounded stimuli; noisy-TV guard. |
| **tier** | needs-new-cache (grounded + ungrounded stimulus render + encoder pass) |
| **coverage** | Gap. No probe distinguishes grounded from ungrounded stimuli. |
| **gap** | The a-priori question is central to "faculty vs readout" and cleanly operationalizable, yet absent. |
| **priority** | high |

### SEM-MATH-7 (medium) — The mental number line (ordinal magnitude axis)

| field | value |
|---|---|
| **position** | Magnitude is encoded ORDINALLY and monotonically (a 1D SNARC-like axis), so 3 is latent-between 2 and 4. |
| **question** | Is there a single latent direction along which numerosity projects MONOTONICALLY, above a random-init ViT, distinguishing a magnitude axis from mere count-classifiability? |
| **lineage** | Dehaene, Bossini, Giraux, "The mental representation of parity and number magnitude" (SNARC), *JEP: General* (1993); Dehaene (1992) triple-code analog magnitude (VERIFIED). |
| **operationalization** | Fit the best linear magnitude axis over 1..8; test MONOTONICITY (Spearman of projection vs true count) and the distance/ratio effect (confusability scales with numeric ratio). The claim is that the ORDERED axis beats the control, not just separability. |
| **null** | No monotonic magnitude axis beyond random-init (numbers separable but not ordered; no distance/ratio effect above control). |
| **control** | Random-init ViT magnitude-axis fit; a categorical-shuffle control (permuted count labels) to show ordering carries the signal. |
| **tier** | needs-new-cache (numerosity cache) |
| **coverage** | Gap. No ordinal/magnitude structure is probed; factors are treated categorically. |
| **gap** | Ordinality is the difference between a number READOUT and a number SENSE; the distance-effect signature is a cheap decisive discriminator never run. |
| **priority** | medium |

### SEM-MATH-8 (medium) — Arithmetic is compositional off-ceiling

| field | value |
|---|---|
| **position** | Latent quantities combine so a held-out sum/difference decodes as well as seen ones: the substrate supports a compositional number system, not per-count memorization. |
| **question** | Given latents for counts a and b, does a shell predict a+b (or a-b) combinations it never saw as well as seen ones, above the random-init floor and a nearest-count baseline? |
| **lineage** | Lakoff and Nunez (2000) arithmetic as metaphorical object-collection (VERIFIED); Kant's 7+5=12 as the canonical synthetic combination. Extends compositional_under_nuisance to a numeric operation. |
| **operationalization** | Cache latents for known counts; train on a subset of (a,b) pairs, test held-out. Held-out == seen accuracy = compositional. Control: random-init ViT; nearest-seen-count baseline; resolution-confound cleared per CM1. |
| **null** | Held-out combined-count accuracy falls to the nearest-count / random-init floor (memorization, the c-series-negative default). |
| **control** | Random-init ViT; nearest-seen-count baseline; the c-series evaporate-at-margin prior (c1/c9/s6 REFUTED) is the strong null to beat. |
| **tier** | needs-new-cache (counting-scene cache) |
| **coverage** | Adjacent: compositional_under_nuisance (attribute binding SUPPORTED off-ceiling) and the c-series (REFUTED). Neither tests numeric composition. |
| **gap** | Arithmetic composition is the untested middle case between passed attribute-binding and failed symbolic-analogy. |
| **priority** | medium |

### SEM-MATH-9 (medium) — Geometric transformation-equivariance (operator)

| field | value |
|---|---|
| **position** | The substrate represents geometric objects such that a transformation (rotation, reflection, scaling) acts as a smooth near-linear operation on the latent: geometric REASONING, not just recognition. |
| **question** | Does a fixed operator learned once map a figure's latent to its rotated counterpart across held-out figures, above a random-init floor, indicating group-structured transformations live in the substrate? |
| **lineage** | Dehaene et al. Core geometry (2006) and Izard et al. (2011): humans intuit Euclidean transformations and invariances (VERIFIED). |
| **operationalization** | Learn operator T (rotation-by-theta) on train pairs; apply to HELD-OUT figures, measure closeness of T(latent) to the true rotated latent vs (a) identity, (b) the operator learned on random-init ViT. Group check: T composed with itself approximates rotation-by-2theta. |
| **null** | The learned operator does not generalize to held-out figures beyond the random-init operator (no substrate-specific group structure). |
| **control** | Random-init ViT operator; identity baseline; a shuffled-pair operator (vacuous-transform guard). |
| **tier** | needs-new-cache (geometric-figure + rotated-pair render) then studio |
| **coverage** | Gap. c2 latent-analogy (REFUTED) was categorical, not a geometric group transformation; a2 is decodability. |
| **gap** | Whether the substrate supports geometric transformations as OPERATIONS is untested (the geometric analogue of the aboutness question). |
| **priority** | medium |

### SEM-MATH-10 (exploratory) — Math present only where it coincides with a perceptual invariant (anti-Tegmark)

| field | value |
|---|---|
| **position** | The Tegmark "reality is mathematical" intuition predicts the WRONG thing: purely formal mathematical structure with no perceptual correlate is NOT decodable above the random floor. |
| **question** | Is decodability of a math property PREDICTED by whether it has a perceptual correlate (number, area, symmetry decode; primality, parity-of-a-symbol do not), and does the formal-only property sit at the random-init floor? |
| **lineage** | Tegmark, "The Mathematical Universe", arXiv 0704.0646 (2007); *Our Mathematical Universe* (2014) (VERIFIED, taken as a position to test NOT endorse). The doctrinal prediction is the OPPOSITE of Tegmark. |
| **operationalization** | Rank math properties by perceptual-correlate strength (numerosity, symmetry high; primality, parity-of-a-rendered-digit low). Decode each from V-JEPA vs random-init. Prediction: real-minus-random delta correlates with perceptual-correlate strength and hits zero for formal-only properties. Noisy-TV guard against rendering artifacts. |
| **null** | Formal-only properties decode above the random-init floor (Tegmark-favorable). |
| **control** | Random-init ViT per property; the ordering test IS the result, so a per-property random baseline is mandatory. |
| **tier** | needs-new-cache (property battery render) |
| **coverage** | Gap. Nothing grades factors by perceptual-correlate strength. |
| **gap** | This separates "math as a currency" from "math as a coincidence of perception" and turns MUH into a graded falsifiable per-property delta. |
| **priority** | exploratory |

### SEM-MATH-11 (flagship) — Number is confounded by area/density (the s10 guard)

| field | value |
|---|---|
| **position** | Apparent numerosity decoding is carried by cumulative surface area / density / spatial-frequency, so the substrate has no abstract number, only low-level visual magnitude. |
| **question** | When area, density, and spatial frequency are decorrelated from count, does numerosity decoding survive above the random-init floor, or collapse (revealing a low-level magnitude confound)? |
| **lineage** | The number literature (Dehaene 1997; Spelke-Kinzler 2007) is contested on this confound; the untrained-network results (Nasr 2019; Kim 2021, VERIFIED) are criticized as area/density artifacts. This operationalizes that critique. |
| **operationalization** | Regenerate stimuli under the congruent/incongruent design (area and count anti-correlated on half the trials). Decode count from V-JEPA; the abstract-number claim requires accuracy to hold on INCONGRUENT trials above random-init. The anti-self-deception meta-test for number (s10 analogue). |
| **null** | Numerosity decoding collapses to chance / random-init floor on incongruent trials (no abstract number; all magnitude confound). |
| **control** | Random-init ViT on incongruent trials; the congruent-minus-incongruent gap is the confound estimate; handcrafted area/density descriptor as the low-level reference. |
| **tier** | needs-new-cache (congruent/incongruent numerosity render) |
| **coverage** | Gap. This is the load-bearing PASS-VACUOUS guard for any numerosity result and does not exist because no numerosity probe exists. |
| **gap** | Without this guard, a positive SEM-MATH-1 result is uninterpretable; it is the number-domain anti-self-deception discipline. |
| **priority** | flagship |

### SEM-MATH-12 (high) — Ordinality/sequence is a temporal-native math perspective

| field | value |
|---|---|
| **position** | Because V-JEPA is temporal-predictive, ORDINAL structure (first/second/third, monotone sequences) is a more native math perspective than static cardinality, and decodes off-ceiling from the time axis specifically. |
| **question** | Does ordinal/sequence structure decode from the TEMPORAL structure of V-JEPA latents (and degrade under time-shuffling) more than from any single frame, above the random-init floor? |
| **lineage** | Dehaene (1992) analog magnitude is inherently ordinal; Piaget seriation (genetic epistemology, VERIFIED). Ties to AT3 and blind-spot 8. |
| **operationalization** | Cache temporally-extended clips of an ordered sequence (increasing count / progressing magnitude). Decode ordinal position from the full temporal latent vs single frames; ablate by SHUFFLING time (AT3). Ordinal-is-temporal = decoding drops under time-shuffle AND beats single-frame, above random-init. |
| **null** | Ordinal decoding is unaffected by time-shuffling and no better than single-frame (ordering not carried by temporal structure), the AT3-time-decodable-not-meaningful prior. |
| **control** | Random-init ViT; time-shuffled ablation; single-frame baseline. |
| **tier** | needs-new-cache (ordered-sequence clip render + encoder pass) |
| **coverage** | Adjacent: AT3 tests time-decodability, not whether ordinal MATH content rides on time. |
| **gap** | The one math position that plays to V-JEPA's native currency; a positive would be the strongest "math as native perspective" evidence. |
| **priority** | high |

### SEM-MATH-13 (medium) — Conservation of number (the concept vs the classifier)

| field | value |
|---|---|
| **position** | The substrate has abstract number only if count decodes invariantly across rearrangement/occlusion/viewpoint, the Piagetian conservation criterion. |
| **question** | Does a numerosity read trained on one arrangement generalize to a DIFFERENT arrangement of the same count, above the random-init floor and above a configuration-memorizing baseline? |
| **lineage** | Piaget number-conservation (genetic epistemology, VERIFIED). Reframes decodability into a use/invariance criterion (blind-spots 1 and 5). |
| **operationalization** | Train on arrangement A; test on arrangement B of matched counts (spread out, clustered, count-preserving occlusion). Conservation = cross-arrangement accuracy near within-arrangement, above random-init. Control: random-init ViT cross-arrangement; a configuration-memorizing (nearest-config) anti-memorization guard. |
| **null** | Cross-arrangement accuracy drops to the configuration-memorizing / random-init floor (a count classifier, no conserved number concept). |
| **control** | Random-init ViT cross-arrangement; nearest-configuration baseline; area/density held constant so conservation is not read off a preserved low-level magnitude. |
| **tier** | needs-new-cache (multi-arrangement numerosity render) |
| **coverage** | Gap. No conservation test of number exists; p2 is the nearest idea but on categorical labels with no frozen-random control. |
| **gap** | Conservation is the developmental gold standard for HAVING a number concept; it converts the number question into the corpus's preferred invariance-with-non-vacuous-control frame. |
| **priority** | medium |

---

## 6. PHYSICS and intuitive physics as a perspective

Is predictive physical intuition (the JEPA objective) a distinct currency from symbolic/formal physics; does the frozen substrate already encode dynamics, affordances, forces decodably and usably; can intuitive physics ground the symbolic perspectives; can we do physics by intervening in the latent. **Verification-forced correction (applies to this whole section):** NO physics, collision, projectile, trajectory, mass, or dynamics cache exists anywhere in `data/cache` (only the shape-color clipset, its random-init twin, handcrafted descriptors, single-frame V-JEPA, and the 25-d programmatic reference). Every physics-decoding position needs a parametric physics clipset RUN THROUGH the frozen encoder, and (a) the clips are not cached, (b) the encoder cannot be run (live-state ban). So the decoding cluster (SEM-PHYS-1, -2, -3, -5, -6, -7, -8, -9) is retiered `needs-new-cache`. Positions the brief already marked needs-new-cache (SEM-PHYS-4, -11, -12) keep that tier. The random-init-ViT control column and the map-invariance guard remain correct.

### SEM-PHYS-1 (flagship) — Intuitive vs formal physics are distinct currencies

| field | value |
|---|---|
| **position** | Intuitive (predictive-perceptual) and formal (symbolic-equational) physics are two DISTINCT currencies, not two readouts of one representation. V-JEPA's masked-latent-prediction is native to the first and structurally blind to the second. |
| **question** | Does the frozen latent carry a physical variable (landing point, time-to-contact) usable by a shell WITHOUT carrying the SYMBOLIC parameters (g, v0, angle), so a symbolic-physics encoder and V-JEPA make DECORRELATED errors on the same prediction? |
| **lineage** | Lake, Ullman, Tenenbaum, Gershman, "Building Machines That Learn and Think Like People", *BBS* 2017, arXiv 1604.00289 (VERIFIED). Kubricht, Holyoak, Lu, "Intuitive Physics", *TiCS* 2017 (VERIFIED). Garrido et al., "Intuitive physics understanding emerges from self-supervised pretraining on natural videos", arXiv 2502.11831 (2025) (VERIFIED). |
| **operationalization** | Two frozen encoders over the same cached projectile/collision clips: V-JEPA vs a symbolic-physics "encoder" = ground-truth generative parameters through a random-init fixed MLP at matched rank. Train a shell on each to predict landing point. Score PR1-style error-decorrelation. Control: symbolic encoder replaced by a random-init MLP on SHUFFLED parameters (destroys structure, keeps rank/compute). |
| **null** | The V-JEPA-shell and symbolic-shell error streams are no more decorrelated than V-JEPA-shell vs the shuffled-parameter control (no distinct symbolic currency). |
| **control** | Symbolic encoder = random-init fixed MLP on SHUFFLED parameters at matched rank/compute; V-JEPA vs frozen-random-projection-of-V-JEPA to bound the readout ceiling. |
| **tier** | needs-new-cache (parametric projectile/collision clipset + encoder pass) |
| **coverage** | Gap. PR1 error-decorrelation exists but never with a symbolic-physics currency; the atlas asserts V-JEPA is dynamics-native but never contrasts it against a formal-physics feature stream. |
| **gap** | "Formal physics" has never been operationalized as a currency, nor asked whether it decorrelates from V-JEPA's intuitive physics. |
| **priority** | flagship |

### SEM-PHYS-2 (flagship) — The substrate is already a physics engine (dynamics variables)

| field | value |
|---|---|
| **position** | Video-prediction pretraining installs a probabilistic forward dynamics model, so physical state variables (velocity, time-to-contact, support/stability) are latently present, not merely appearance features. |
| **question** | Are continuous DYNAMICS variables decodable from the frozen latent ABOVE a random-init-ViT of matched architecture and resolution, i.e. is the dynamics content bought by PRETRAINING rather than architecture? |
| **lineage** | Battaglia, Hamrick, Tenenbaum, "Simulation as an engine of physical scene understanding", *PNAS* 2013, DOI 10.1073/pnas.1306572110 (VERIFIED). Hamrick, Battaglia, Tenenbaum, CogSci 2011 (VERIFIED). Garrido et al. 2025 (VERIFIED). |
| **operationalization** | Cached parametric physics clips with ground-truth velocity, TTC, stability. Ridge/shallow-MLP probe from pooled V-JEPA. Non-vacuous control: SAME probe on random-init same-arch ViT-L at matched 256px (NOT a square projection). **Map-invariance guard:** report both a linear probe (invariant to invertible maps, so expected to tie a frozen-random projection) AND a metric that is NOT map-invariant so the delta is not forced to 0.000. |
| **null** | Frozen V-JEPA decodes velocity/TTC/stability no better than a random-init same-arch ViT-L at matched resolution (any apparent physics is generic linear separability). |
| **control** | Random-init same-arch ViT-L at matched 256px; tuned linear probe on raw downsampled pixels as the floor. |
| **tier** | needs-new-cache (physics clipset + encoder pass) |
| **coverage** | Partial. substrate_vs_random_init_vit establishes nuisance-INVARIANCE (p=0.029) but on STATIC shape, not dynamics. a1_affordance_decode touches action-relevance but ties frozen-random. |
| **gap** | The most V-JEPA-native claim (video pretraining installs a forward physics model) is untested against the random-init control on actual dynamics variables. |
| **priority** | flagship |

### SEM-PHYS-3 (high) — V-JEPA's physics is noisy-Newtonian, not veridical

| field | value |
|---|---|
| **position** | V-JEPA should track the probabilistic, perceptually-noisy structure of human intuitive physics (systematic biases, uncertainty scaling with complexity) rather than exact analytic mechanics. |
| **question** | Does a shell reproduce the SIGNATURE BIASES of human intuitive physics rather than an exact simulator's errors, i.e. does its error PROFILE match noisy-Newton better than analytic Newton? |
| **lineage** | Sanborn, Mansinghka, Griffiths, "Reconciling intuitive physics and Newtonian mechanics for colliding objects", *Psychological Review* 2013 (VERIFIED, "noisy Newton"). Hamrick et al., "Inferring mass in complex scenes by mental simulation", *Cognition* 2016, PubMed 27592412 (VERIFIED). |
| **operationalization** | Cached two-body collision clips with parametric mass ratios and controlled perceptual-noise. Shell predicts relative mass. Fit exact-Newton and noisy-Newton to the residuals; model comparison by held-out likelihood. Control: same fit on a shell over random-init-ViT (does the noisy-Newton signature require pretraining). |
| **null** | The shell's residuals are fit equally well (or better) by exact Newton, and any noisy-Newton advantage is present identically on random-init-ViT (probe noise, not a substrate prior). |
| **control** | Shell over random-init same-arch ViT-L; exact-Newton as the competing generative model the noisy account must beat. |
| **tier** | needs-new-cache (collision clipset with noise manipulation) |
| **coverage** | Gap. Nothing fits a cognitive-physics error model to shell residuals. |
| **gap** | No test asks what KIND of physics the substrate does (probabilistic-biased vs analytic). |
| **priority** | high |

### SEM-PHYS-4 (flagship) — Intuitive physics is the grounding floor (symbol drives percept)

| field | value |
|---|---|
| **position** | The perceptual-physical currency is the one to which symbolic currencies must be tied back, because it is the only one whose tokens have non-arbitrary causal contact with the world. Grounding flows TOWARD it. |
| **question** | Can a symbolic-physics statement (a code/equation for "higher mass") DRIVE a correct expectation in the frozen latent (top-down: symbol predicts which clip is physically consistent), above a no-shared-history floor? |
| **lineage** | Spelke, Kinzler, "Core knowledge", *Developmental Science* 2007 (VERIFIED). Spelke, Core Knowledge and Natural Number (College de France, VERIFIED). Lake et al. 2017 (VERIFIED). |
| **operationalization** | Needs a SECOND encoder: an LLM/CLIP-aligned symbolic head over the same physics clips. Referential-expectation: given a symbolic token (mass A > mass B) and two candidate frozen-latent clips (one consistent, one violation), does an alignment map from symbolic space pick the consistent clip. Control: random-map-of-equal-rank (AL2 alignment-artifact control) and a frozen-random-init receiver. Score by top-down selection accuracy above the shuffled-pairing floor, NOT R2. |
| **null** | A symbolic token drives perceptual expectation no better than a random map of equal rank into the frozen latent (no genuine top-down grounding, only statistical co-occurrence a full-rank map fits). |
| **control** | Random-map-of-equal-rank plus frozen-random-init receiver at matched channel capacity. |
| **tier** | needs-new-cache (physics clipset + symbolic encoder) |
| **coverage** | Gap (blind-spot 9). AL2/AL3 test statistical alignment; grounding-direction untested. |
| **gap** | The thesis that intuitive physics grounds the symbolic currencies is asserted (Spelke-style) but never operationalized as a directional use-based test. The physics-domain analog of the program's deepest blind spot. |
| **priority** | flagship |

### SEM-PHYS-5 (high) — The latent encodes affordances (Gibsonian)

| field | value |
|---|---|
| **position** | Perception already carries what actions a scene permits (a surface affords support, a gap affords passing-through), distinct from raw appearance. |
| **question** | Is action-relevant affordance (can-support, can-pass-through, will-collide) decodable ABOVE a random-init-ViT AND ABOVE a plain appearance baseline, i.e. does the substrate expose affordances as such rather than relabeled appearance? |
| **lineage** | Gibson, *The Ecological Approach to Visual Perception* (1979) (VERIFIED). Ahn et al., SayCan (2022) (VERIFIED) as the ML affordance-grounding analog. |
| **operationalization** | Cached clips labeled with affordances. Probe from frozen latent. Two controls: (a) random-init same-arch ViT-L; (b) a tuned APPEARANCE baseline (DINOv2 / category) so affordance is not just appearance. Map-invariance guard as in SEM-PHYS-2. |
| **null** | Affordance decodability from V-JEPA ties a random-init ViT and/or a tuned appearance baseline (the a1_affordance_decode outcome). |
| **control** | Random-init same-arch ViT-L at matched resolution AND a tuned static-appearance baseline (DINOv2 / category), so affordance must beat both. |
| **tier** | needs-new-cache (affordance-labeled clipset, esp. DYNAMIC affordances like will-collide; encoder pass) |
| **coverage** | Tested and REFUTED-ish: a1_affordance_decode fails to exceed shuffle floor or ties frozen-random, but was STATIC and single-control. |
| **gap** | The existing a1 test never dissociated affordance from appearance, and never tested DYNAMIC affordances where video pretraining should help. A cleaner rerun could still bite. |
| **priority** | high |

### SEM-PHYS-6 (high) — Physics by intervening in the latent (controllability)

| field | value |
|---|---|
| **position** | Because the substrate is a forward dynamics model, perturbing a physical-state direction should produce a downstream prediction consistent with that physical change, giving a controllability/intervention handle. |
| **question** | If we perturb the decoded velocity direction and roll forward, does the trajectory change in the PHYSICALLY CORRECT direction (faster perturbation, farther landing) more than perturbing a random matched-norm direction? |
| **lineage** | Friston free-energy / active inference (VERIFIED, FEP; "A free energy principle for a particular physics", arXiv 1906.10184). Battaglia et al. 2013 (VERIFIED). |
| **operationalization** | Frozen substrate + tiny latent dynamics head. Decode the velocity direction, add k*(velocity-direction), roll forward, measure whether the predicted outcome shifts monotonically and correctly with k. Control: perturb a RANDOM direction of matched norm; a physical direction should produce a correct, monotone, larger effect (a DR4-style counterfactual-match tested for physical correctness). |
| **null** | Perturbing the decoded velocity direction changes the rolled-forward prediction no more, and no more correctly, than a random matched-norm direction (a readout, not an intervention handle). |
| **control** | Random latent direction of matched norm; DR4 causal-intervention counterfactual-match as the existing nearest lane. |
| **tier** | needs-new-cache (physics clipset for the decode step + a dynamics head over V-JEPA; note ex2's dynamics head is a synthetic toy, not V-JEPA) |
| **coverage** | Partial. ex2 is a promoted positive (on a synthetic toy) and DR4 tests counterfactual match, but neither tests whether a PHYSICAL direction is a correct-signed intervention handle. |
| **gap** | The controllability-of-physics claim has never been tested as a signed, physically-correct intervention against a random-direction control. |
| **priority** | high |

### SEM-PHYS-7 (high) — Intuitive physics is fragmented into Spelke core signatures

| field | value |
|---|---|
| **position** | Intuitive physics is fragmented into core signatures (cohesion, continuity, contact, no-action-at-a-distance) rather than a single simulator; which the substrate passes is diagnostic of what pretraining installs. |
| **question** | Does the substrate support a VIOLATION-OF-EXPECTATION signal SELECTIVELY for some core principles but not others, above a random-init-ViT baseline? |
| **lineage** | Spelke, Kinzler 2007 (VERIFIED). Baillargeon violation-of-expectation (Margoni, Surian, Baillargeon, *Psychological Review* 2024, VERIFIED). Garrido et al. 2025 already uses VoE on JEPA for permanence/shape (VERIFIED). |
| **operationalization** | Cached matched possible/impossible clip pairs, one per principle. Surprise = frozen predictor's latent-prediction error on impossible vs possible. Per-principle VoE score. Control: random-init same-arch ViT-L predictor; a real signature shows a principle-selective profile the random-init lacks. |
| **null** | Impossible clips produce no more surprise than possible clips beyond the random-init baseline, OR the surprise is principle-nonselective (generic novelty). |
| **control** | Random-init same-arch ViT-L predictor; a low-level pixel-change/optical-flow surprise baseline (impossible clip must not simply have more raw motion). |
| **tier** | needs-new-cache (possible/impossible clip-pair render + predictor pass) |
| **coverage** | Partial and bound. d2/n8 permanence touch VoE but are pooling-bound and single-principle. No principle-selective battery exists. |
| **gap** | VoE has only been run on permanence, pooling-bound; the Spelke-style multi-principle selectivity profile is unposed. |
| **priority** | high |

### SEM-PHYS-8 (high) — Time as causal content (before/after, cause/effect)

| field | value |
|---|---|
| **position** | Physical MEANING lives in temporal structure: before/after, cause/effect, and event boundaries are the substrate's intrinsic physical content, so destroying time destroys physics-as-content. |
| **question** | Does the substrate encode the DIRECTION and CAUSAL ORDER of a physical event, such that a shell can read causal order, and does this collapse under time-shuffling more than a matched non-temporal baseline? |
| **lineage** | Friston active inference (VERIFIED); Spelke contact/causality (VERIFIED); Michotte launching-effect as the classical source; V-JEPA's temporal-predictive objective (Garrido et al. 2025, VERIFIED). |
| **operationalization** | Cached collision/causal clips played forward vs reversed, with cause-effect order manipulated. Shell reads time-arrow and causal order. AT3 time-axis ablation: correct vs shuffled/reversed time. Controls: random-init same-arch ViT-L AND a single-frame DINOv2 baseline (no time axis, must fail, proving the signal is temporal). |
| **null** | Time-shuffling degrades causal-order readout no more than it degrades a static-appearance readout (the substrate treats events as a bag of frames). |
| **control** | Static single-frame DINOv2 baseline (must fail on causal order) plus random-init same-arch ViT-L; AT3 time-axis ablation. |
| **tier** | needs-new-cache (forward/reversed causal clipset + encoder pass) |
| **coverage** | Partial. AT3/AL3 test time-decodability, not causal order or time-arrow as physical MEANING (blind-spot 8 specialized to physics). |
| **gap** | The substrate's most native currency has never been probed for causal-order semantics, only decodability. |
| **priority** | high |

### SEM-PHYS-9 (medium) — Invariant physical quantities (mass ratio, momentum)

| field | value |
|---|---|
| **position** | The substrate carries frame-and-viewpoint-invariant conserved/relational quantities (mass ratio, relative momentum), and it is these invariants, not surface kinematics, that constitute genuine physical understanding. |
| **question** | Is a viewpoint-and-nuisance-INVARIANT physical relation decodable stably across camera viewpoint and appearance changes, above a random-init-ViT, i.e. does the substrate expose physical INVARIANTS rather than viewpoint-tied kinematics? |
| **lineage** | Sanborn et al. 2013 noisy Newton (VERIFIED); Battaglia et al. 2013 (VERIFIED). Connects to the one landed positive (nuisance-invariance, p=0.029). |
| **operationalization** | Cached collision clips with fixed physics from multiple viewpoints/appearances. Probe relative-mass / momentum-conservation; measure cross-viewpoint stability (train A, test B). Control: random-init same-arch ViT-L. **a2 guard:** score cross-viewpoint TRANSFER by kNN-topology overlap, NOT probe R2, since a shared linear probe cancels an invertible map and forces a 0.000 delta. |
| **null** | Relative-mass/momentum decodability does not transfer across viewpoint better on V-JEPA than random-init-ViT (viewpoint-tied kinematics), and any tie is a forced consequence of the shared-probe map (the a2 vacuous-tie). |
| **control** | Random-init same-arch ViT-L at matched resolution; the a2 cross-viewpoint-transfer result as the cautionary vacuous-tie precedent (score by kNN-overlap). |
| **tier** | needs-new-cache (multi-viewpoint collision clipset) |
| **coverage** | Partial and cautionary. substrate_vs_random_init_vit shows nuisance-invariance for STATIC shape; a2 is a guaranteed-vacuous tie. Neither tests a physical INVARIANT. |
| **gap** | The nuisance-invariance positive has never been extended from static shape to a physical INVARIANT, scored by non-map-invariant topology. |
| **priority** | medium |

### SEM-PHYS-10 (medium) — Relational physics is pooling-bound

| field | value |
|---|---|
| **position** | Genuinely relational multi-object physics (support chains, occlusion-with-persistence, tool-mediated force transfer) requires a spatial/object index the pooled interface destroys, so the substrate's physics is single-object and present-tense; the ceiling is the interface, not the mechanism. |
| **question** | Does multi-object relational physics decode from the POOLED latent, and if it fails, does the same content survive on a DENSE token interface, isolating pooling as the binding constraint? |
| **lineage** | Spelke core knowledge multi-object relations (VERIFIED); the pooling-bottleneck framing is in-corpus (dense_vs_pooled.json). Battaglia et al. 2013 stack-toppling (VERIFIED). |
| **operationalization** | Cached multi-object support/tool clips with relational ground truth. Probe from (a) pooled V-JEPA and (b) dense per-token features. Control: random-init same-arch ViT-L at each interface. Fork: pooled ties random-init but dense beats it => pooling binds; dense also ties => the mechanism is the ceiling. Guard: relational content must be genuinely non-additive (dense_vs_pooled showed single spatial factors already survive pooling). |
| **null** | Multi-object relational physics decodes no better on dense than pooled (and both tie random-init), so pooling is NOT the binding constraint and the substrate simply lacks relational physics. |
| **control** | Random-init same-arch ViT-L at both interfaces; dense_vs_pooled.json single-factor result as the precedent that additive factors already survive pooling. |
| **tier** | needs-new-cache (multi-object physics clipset AND dense V-JEPA 2.1 weights, not on HF, models.yaml available=false; DR1 blocks occlusion) |
| **coverage** | Partial and blocked. dense_vs_pooled shows single factors survive pooling (weakening the naive story); dense weights blocked; DR1 blocks occlusion. |
| **gap** | The decisive test is gated on the 2.1 dense interface; the single-factor result already undercuts the naive version. Honest status: real but gated. |
| **priority** | medium |

### SEM-PHYS-11 (exploratory, OPEN) — Counterfactual physics (what did not happen)

| field | value |
|---|---|
| **position** | Negation/counterfactual physics (the impossible branch, the absent force) is beyond the frozen substrate because pooled video encodes only present positive dynamics; genuine physical thought must represent absence, and V-JEPA cannot. |
| **question** | Can any shell represent a COUNTERFACTUAL physical outcome (what would have happened if the wall were absent) distinctly from the actual outcome, above a baseline with access only to the actual clip, on real bound video with occlusion? |
| **lineage** | Lake et al. 2017 causal model-building (VERIFIED); Baillargeon VoE (VERIFIED). Blind-spot 4. |
| **operationalization** | Requires REAL bound video with occlusion (blocked on DR1). Counterfactual task: present the actual clip to an intervention point, ask the shell to predict the counterfactual (wall removed) outcome; compare to a DR4 counterfactual-match criterion. Control: a shell with access only to the actual outcome must not match the counterfactual. Honest status: on the pooled present-tense substrate this is likely NOT operationalizable (absence has no pooled signature); mark the pooled version OPEN and gate the real version on DR1. |
| **null** | The shell's counterfactual prediction is no better than a baseline that only saw the actual outcome (it does not represent the absent branch). |
| **control** | Actual-outcome-only shell (no intervention variable); DR4 counterfactual-match as the nearest lane. |
| **tier** | needs-new-cache (real bound occlusion video, blocked on DR1) / open-philosophical for the pooled version |
| **coverage** | Gap (blind-spot 4). n8/d2 are the nearest miss, pooling-bound. |
| **gap** | Physical thought about absence is the honestly-OPEN frontier; the pooled substrate has no signature for what did not happen. |
| **priority** | exploratory |

### SEM-PHYS-12 (medium) — Formal physics overrides intuitive physics (arbitration)

| field | value |
|---|---|
| **position** | Formal physics is a learned symbolic overlay that can OVERRIDE and sometimes CONTRADICT intuitive physics (impetus/naive-physics misconceptions), so the two currencies can disagree and a mature system needs ARBITRATION. |
| **question** | On stimuli where naive physics gives the WRONG answer a formal rule corrects, does the shell reproduce the INTUITIVE error, and can a formal-symbolic head arbitrate to override it, beating either currency alone? |
| **lineage** | McCloskey, "Intuitive Physics", *Scientific American* 1983, and naive theories of motion (VERIFIED, impetus misconceptions). Kubricht et al. 2017 (VERIFIED). Connects to the WS-series arbitration lanes. |
| **operationalization** | Cached misconception-diagnostic clips (impetus-eliciting trajectories) with both intuitive-wrong and formally-correct labels. Shell predicts outcome; measure whether it matches the intuitive-WRONG answer. Then add a formal-symbolic head and a WS3 arbiter; test whether arbitration beats each single currency. Controls: random-init-ViT for the intuitive arm; a shuffled-symbolic head at matched compute for the formal arm. |
| **null** | The shell does not reproduce the intuitive misconception (it is correct or random, not human-like-wrong), OR arbitration yields no gain over the better single currency at matched compute. |
| **control** | Random-init same-arch ViT-L for the intuitive arm; shuffled-symbolic head at matched compute; WS3 arbitration; the better-single-currency as the baseline to beat. |
| **tier** | needs-new-cache (misconception clipset + symbolic-physics encoder) |
| **coverage** | Gap. WS1/WS3 test two-encoder arbitration for correctness generally, never on misconception content where the currencies systematically DISAGREE. |
| **gap** | The strongest evidence V-JEPA has HUMAN intuitive physics would be reproducing human misconceptions; and the strongest case for MoP arbitration is where currencies conflict. Both unposed. |
| **priority** | medium |

---

## 7. PLASTICITY and DEVELOPMENT across modalities

Can a frozen-substrate + tiny-shell MoP be molded the way a child's brain is; is there a developmental order; are there per-modality critical periods; does cross-modal transfer occur; is moldability substrate-deep or shell-only. These positions are mostly cpu-now: they run on the shell over EXISTING cached latents with proxy tasks, so they do not need new stimulus caches, though the cross-MODALITY ones use perspective-proxy tasks (spatial/label/symbolic) that are constructible from the shape-color cache.

### SEM-PLAS-1 (flagship) — The frozen substrate caps plasticity at shell-deep (reshaping vs reweighting)

| field | value |
|---|---|
| **position** | Exposure can only re-weight a fixed feature basis, never reshape the latent organization, so "moldable as a child's brain" is false at the substrate level by construction and true (if at all) only at the shell. |
| **question** | When the shell is molded, does the ORGANIZATION the downstream reader sees change (new separating structure a linear reader of the raw latent could not already find), or does molding only re-weight directions already present? |
| **lineage** | Quartz and Sejnowski, "The neural basis of cognitive development: A constructivist manifesto", *BBS* 20 (1997) 537-556 (VERIFIED, neural constructivism, the anti-frozen limit). Kumar, Raghunathan, Jones, Ma, Liang, "Fine-Tuning can Distort Pretrained Features", ICLR 2022, arXiv 2202.10054 (VERIFIED). |
| **operationalization** | Frozen V-JEPA latents. Train shell A on a modality task, freeze it. Measure whether A's readable structure is reachable by ANY linear map of the raw latent: fit a fresh linear probe directly to the same targets and compare. Non-vacuous control: a frozen-random-init-ViT substrate at matched resolution and matched shell capacity; any "shell helps" effect must exceed random features + same shell. |
| **null** | Shell-molded readable structure is fully recoverable by a direct linear probe of the raw frozen latent (delta within seed spread, not above the random-init-ViT + same-shell control). Molding is reweighting, not reshaping. |
| **control** | Frozen-random-init-ViT at matched resolution and shell capacity, plus a direct linear-probe-of-raw-latent baseline; never a square/full-rank projection (which would trivially preserve all structure and fake reshaping). |
| **tier** | cpu-now |
| **coverage** | Partially/obliquely: PR2 tests real-vs-random adaptation speed and BWT; p1 shows zero purity gain over frozen-random. No experiment isolates reshaping-vs-reweighting as the plasticity ceiling per se. |
| **gap** | The corpus asserts the frozen substrate cannot be reshaped as a caveat but never operationalizes "molding = reweighting not reshaping" as a preregistered testable ceiling. |
| **priority** | flagship |

### SEM-PLAS-2 (flagship) — There is a developmental ORDER across perspectives

| field | value |
|---|---|
| **position** | Physical/spatial grounding must be acquired first and bootstraps language, then math/code; out-of-order presentation yields worse final competence at matched total data, mirroring child development. |
| **question** | At matched total exposure and shell capacity, does a spatial-then-language-then-symbolic ordering beat a shuffled/reversed ordering on final multi-modality competence, and does any order effect survive under a frozen-random substrate? |
| **lineage** | Elman, "Learning and development in neural networks: the importance of starting small", *Cognition* 48 (1993) 71-99 (VERIFIED). Karmiloff-Smith, *Beyond Modularity* (MIT Press, 1992) and *TiCS* 2 (1998) 389-398 (VERIFIED). |
| **operationalization** | Three cached-latent shell tasks as perspective proxies: (a) spatial/physical, (b) label/category, (c) symbolic/relational. Train one shell sequentially in several orders at matched total updates; measure final joint competence. Control: same orderings on frozen-random-init-ViT; matched total data and matched LR-integral so order is not confounded with exposure. |
| **null** | Final competence is order-invariant at matched total data and LR-integral, and any residual order effect appears equally under frozen-random features (a generic continual-SGD trajectory effect, not developmental). |
| **control** | Frozen-random-init-ViT through identical orderings; matched total updates and LR-integral; shuffled-order arm. |
| **tier** | cpu-now |
| **coverage** | Adjacent but not this: d6_sensitive_window (early-vs-late same evidence, refuted); d5/b3/b7/d7 (difficulty curricula, mixed). None test cross-MODALITY ordering. |
| **gap** | The corpus tests order-within-one-modality and refutes it, but never the cross-perspective developmental sequence the thesis names. |
| **priority** | flagship |

### SEM-PLAS-3 (flagship) — Cross-modal transfer is real and asymmetric

| field | value |
|---|---|
| **position** | Molding the shell on one perspective (code/structure) improves competence in another (math), above a matched-compute no-transfer baseline, so the perspectives share transferable structure. |
| **question** | Does shell pretraining on perspective X reduce the data/updates to reach criterion on perspective Y, beyond a matched-compute shell that saw X-shaped noise or an unrelated task? |
| **lineage** | Scherer, Siddiq, Sanchez Viveros, "The cognitive benefits of learning computer programming: A meta-analysis", *Journal of Educational Psychology* 2019 (VERIFIED, g=0.47 far transfer, g=0.75 near). Karmiloff-Smith representational redescription (VERIFIED). |
| **operationalization** | Code-like task (compose rules over attribute tokens) and math-like task (arithmetic/ordering over the same attributes) as shell tasks on cached latents. Arm 1: pretrain shell on code-task, measure updates-to-criterion on math-task. Arm 2 (matched-compute control): pretrain SAME shell SAME updates on a scrambled-label code-task. Transfer = Arm1 faster than Arm2. Control: repeat on frozen-random-init-ViT. |
| **null** | Updates-to-criterion are the same whether the shell was pretrained on the real source or a compute-matched scrambled version, and any gap is matched under frozen-random features. |
| **control** | Matched-compute scrambled-structure pretraining arm; frozen-random-init-ViT arm; never a from-scratch arm that simply saw less compute. |
| **tier** | cpu-now |
| **coverage** | Not directly. d4_transfer_matrix (unwired) tests decodability transfer, not a molded-shell speeding a second modality; c2 (REFUTED) tests analogy as arithmetic. |
| **gap** | The single most specific "does learning code help math" question has no experiment. |
| **priority** | flagship |

### SEM-PLAS-4 (high) — Per-modality critical periods (Achille deficit-and-restore)

| field | value |
|---|---|
| **position** | For a perspective there is a window during shell training when exposure permanently sets competence; a deficit inside the window cannot be recovered by later exposure. |
| **question** | Does a modality-specific deficit (withhold/corrupt X for an early window, then restore) produce permanent competence loss vs never-deprived, with recovery depending on onset/length, and is the effect ABSENT on frozen-random features? |
| **lineage** | Achille, Rovere, Soatto, "Critical Learning Periods in Deep Networks", ICLR 2019, arXiv 1711.08856 (VERIFIED). Hensch, *Nature Reviews Neuroscience* 6 (2005) 877-888 (VERIFIED). Werker and Tees 1984 (VERIFIED). |
| **operationalization** | Cached-latent shell on a stream mixing perspectives. Deprivation arm: corrupt/withhold X for steps [t0, t0+w], then restore; compare final X-competence to never-deprived at matched total X-exposure. Sweep onset and width. Measure the Fisher-trace (diagnostics/fisher_trace.py exists) for the rise-then-fall signature. Control: identical protocol on frozen-random-init-ViT. |
| **null** | Deprivation leaves no permanent deficit once total X-exposure is matched, the Fisher trace decays monotonically, and any deficit is reproduced on frozen-random features. |
| **control** | Frozen-random-init-ViT arm; never-deprived matched-total-exposure arm; the Fisher-trace shape as an internal falsifier (monotone decay = no window). |
| **tier** | cpu-now |
| **coverage** | e3_plasticity / n5_fisher_reopen (REFUTED: Fisher trace decays monotonically); these test a single generic task, not a MODALITY-SPECIFIC deficit-and-restore. PR5 is the live retry but single-stream. |
| **gap** | The corpus refuted generic critical periods but never ran the Achille deficit-and-restore PER MODALITY. |
| **priority** | high |

### SEM-PLAS-5 (high) — Molding is redescription-limited (frozen-latent information ceiling)

| field | value |
|---|---|
| **position** | The shell can make IMPLICIT frozen structure EXPLICIT but cannot add information the latent does not contain, so plasticity is bounded above by the latent's mutual information with the target and is re-formatting, never enrichment. |
| **question** | Across increasing shell capacity/exposure, does molded competence asymptote at the linear-decodability ceiling of the raw latent (redescription only), or exceed the raw latent's information (enrichment)? |
| **lineage** | Karmiloff-Smith, *Beyond Modularity* (1992), representational redescription re-formats existing implicit knowledge, does not add sensory information (VERIFIED). |
| **operationalization** | Estimate an upper bound on task-relevant information in the raw latent (high-capacity regularized cross-validated nonlinear probe). Train shells of increasing capacity/exposure; check whether any exceeds that bound. Falsifier: a shell EXCEEDS the ceiling (enrichment, or more likely a leak to hunt). Control: same ceiling estimate on frozen-random-init-ViT. |
| **null** | Molded competence asymptotes at (never exceeds) the raw latent's information ceiling; plasticity is redescription with no enrichment. |
| **control** | Raw-latent high-capacity nonlinear probe as the ceiling; frozen-random-init-ViT ceiling estimate; regularized cross-validated estimation to avoid overfit inflating the ceiling. |
| **tier** | cpu-now |
| **coverage** | Implicit in the corpus framing ("upper-bounded by how much task-relevant detail the frozen latent preserves") but never operationalized as a measured ceiling shells respect. |
| **gap** | Redescription-vs-enrichment is the precise mechanism-level plasticity limit and is untested. |
| **priority** | high |

### SEM-PLAS-6 (high) — Starting small helps in the shell (Elman capacity-growth)

| field | value |
|---|---|
| **position** | Constraining early shell capacity (short horizon, low-rank head, small context) then relaxing yields better final competence and retention than full capacity throughout, at matched final compute. |
| **question** | Does a capacity-growth schedule beat full-capacity-throughout on final competence and forgetting at matched final compute, and does it survive on frozen-random features? |
| **lineage** | Elman 1993 (VERIFIED). Newport, "Maturational constraints on language learning", *Cognitive Science* 14 (1990) 11-28 (VERIFIED). Skeptic anchor: Wu, Dyer, Neyshabur, "When Do Curricula Work?", ICLR 2021 (VERIFIED, curricula help mainly under limited time or noisy data). |
| **operationalization** | Shell with a growable bottleneck: start low-rank/short-horizon/small-context, relax to full; compare to full-capacity-throughout at matched TOTAL updates and FINAL capacity. Endpoints: final accuracy and backward transfer. Control: frozen-random-init-ViT arm; match final compute so the schedule is not just fewer effective parameters early. |
| **null** | Capacity-growth changes only convergence speed, not final competence or retention, at matched final compute, and any gain is reproduced under frozen-random features. |
| **control** | Full-capacity-throughout arm at matched total updates and final capacity; frozen-random-init-ViT arm; permute which axis grows (rank vs horizon vs context). |
| **tier** | cpu-now |
| **coverage** | Named in the corpus (vol2 A4 "starting small", flagged build-now) but no run exists; e3 tested plasticity scheduling not capacity growth. |
| **gap** | Elman's is the most famous ML-echo developmental result, explicitly planned (A4) but never executed. |
| **priority** | high |

### SEM-PLAS-7 (high) — Multi-perspective forgetting is the moldability bottleneck

| field | value |
|---|---|
| **position** | Because the frozen substrate cannot absorb change, all interference lands on the tiny shell, so multi-perspective moldability is bottlenecked by shell forgetting, not substrate capacity. |
| **question** | As perspectives are added sequentially, does backward transfer degrade faster in the shell-only regime than a matched-capacity system would, and does the sparse/gated architecture mitigate it on real latents? |
| **lineage** | Dohare et al., "Loss of plasticity in deep continual learning", *Nature* 632 (2024) 768-774 (VERIFIED, continual backprop restores plasticity). McCloskey and Cohen catastrophic interference (classic). Hensch 2005 (VERIFIED). |
| **operationalization** | Sequential stream of perspective-proxy tasks on cached latents. Track backward transfer and effective rank / dead-unit fraction (ex13_long_stream tracks effective_rank). Compare dense shell vs sparse/gated shell (e7) vs continual-backprop reinit. Control: frozen-random-init-ViT; param-matched AND activation-sparsity-matched dense head (the PR3 control) so sparsity-the-mechanism is isolated from sparsity-the-capacity-cut. |
| **null** | Sparse/gated ties a param-matched, activation-sparsity-matched dense shell on backward transfer on real latents, and continual-backprop reinit adds nothing over tuned baseline. |
| **control** | Param-matched AND activation-sparsity-matched dense head (PR3 control); frozen-random-init-ViT substrate; tuned constant/decay LR baseline before crediting reinit. |
| **tier** | cpu-now |
| **coverage** | Strongly adjacent: e7_sparse (provisional positive on synthetic, pending real-latent + frozen-random control), PR3 (registry-only), ex13, ex15. But framed as generic continual learning, not multi-PERSPECTIVE interference. |
| **gap** | The machinery exists but was never framed as the bottleneck on multi-perspective moldability, nor run on REAL latents with the perspective-proxy stream. Closing PR3 on real latents is the concrete step. |
| **priority** | high |

### SEM-PLAS-8 (high) — Symbol-to-percept top-down molding (bidirectional plasticity)

| field | value |
|---|---|
| **position** | A frozen substrate blocks true (bidirectional) plasticity: a perspective can be read OUT but cannot drive a perceptual EXPECTATION back in, so communication is one-directional and the developmental mechanism by which language reshapes perception is structurally blocked. |
| **question** | Can a shell trained to map a symbolic cue to a target latent region then use that cue to improve a downstream perceptual-prediction task (top-down benefit), or is the benefit strictly bottom-up? |
| **lineage** | Karmiloff-Smith redescription (cross-system codes become bidirectionally accessible, VERIFIED). Vygotsky (language reshapes perception). Quartz and Sejnowski 1997 (VERIFIED). Blind-spot 9. |
| **operationalization** | Two shells: an encoder-side reader (percept to symbol) and a generator-side predictor (symbol to expected-latent). Test whether conditioning the perceptual-prediction shell on the symbol REDUCES prediction error on held-out clips vs an unconditioned matched-capacity predictor. Control: frozen-random-init-ViT; a shuffled-symbol arm. Honest caveat: a fully bidirectional test may need a second (text) encoder cache; the within-substrate version is runnable now. |
| **null** | Symbol conditioning does not reduce perceptual-prediction error beyond a matched-capacity symbol-ignoring predictor (or beyond a shuffled-symbol control); molding is bottom-up only. |
| **control** | Shuffled-symbol arm; matched-capacity symbol-ignoring predictor; frozen-random-init-ViT substrate. |
| **tier** | cpu-now (within-substrate) / needs-new-cache (cross-encoder version) |
| **coverage** | Not covered as top-down. LLM/CLIP rows treat grounding as bottom-up; blind-spot 9 OPEN. |
| **gap** | Bidirectional molding is exactly how language molds a child's perception; the top-down direction is untested. |
| **priority** | high |

### SEM-PLAS-9 (medium) — Plasticity closure is nothing but LR annealing

| field | value |
|---|---|
| **position** | Any stabilize-then-specialize dynamic in the shell is fully explained by LR decay because there is no representation to protect, so "critical-period closure" has zero developmental content here. |
| **question** | Does any staged closure schedule produce a retention/specialization benefit a tuned monotonic LR-decay baseline at matched LR-integral does not, or is closure exactly annealing? |
| **lineage** | Hensch 2005 (VERIFIED): biological closure is ACTIVE (inhibitory maturation, PNN brakes), the exact contrast; if the shell shows only passive decay, closure is content-free. Corpus A4/6.1 note. |
| **operationalization** | Staged-plasticity closure (per-module freezing / rigidity term) vs a tuned cosine/monotonic LR-decay baseline with matched LR-integral, on retention AND specialization (does module identity localize by task). A positive requires a benefit annealing at equal integral cannot reproduce (e.g. task-localized module specialization). Control: frozen-random-init-ViT. |
| **null** | Staged closure ties tuned decay at matched LR-integral on retention and shows no task-localized specialization beyond decay; closure is annealing. |
| **control** | Tuned monotonic-decay baseline at matched LR-integral; frozen-random-init-ViT; module-identity localization as the specialization falsifier. |
| **tier** | cpu-now |
| **coverage** | e3_plasticity (REFUTED: staged ties constant/cosine LR); the corpus already reads this as an LR trick. But the SPECIALIZATION half (does closure localize modules by perspective) was never separated from retention. |
| **gap** | e3 refuted the retention benefit; it never tested whether closure induces perspective-localized module SPECIALIZATION. |
| **priority** | medium |

### SEM-PLAS-10 (medium) — Surprise-gated reopening cannot beat the noisy-TV confound

| field | value |
|---|---|
| **position** | The epistemic (reducible) surprise signal is confounded with aleatoric (irreducible) latent noise, so any reopening either fails to fire or chases noise; biological novelty-triggered plasticity has no frozen-substrate analog. |
| **question** | Can a distributional (disagreement-based) surprise signal gate shell reopening so it allocates more plasticity to genuinely reducible novelty than to a noisy-TV distractor, or does it replicate the e4 conflation? |
| **lineage** | Hensch 2005 (VERIFIED); Schmidhuber/Pathak intrinsic-motivation and the noisy-TV problem; corpus e4_neuromod negative (point prediction-error gating amplifies error on noise 30/30). |
| **operationalization** | Stream with a reducible-novelty partition and an irreducible-noise (noisy-TV) partition over the frozen latent. Gate plasticity on ENSEMBLE DISAGREEMENT (distributional epistemic signal, ensemble.py exists) rather than point error. Metric: LR-integral allocated to reducible vs noise. Control: ungated arm and frozen-random-init-ViT; a positive requires strictly more reducible-partition allocation than ungated AND passing the noisy-TV guard. |
| **null** | The disagreement gate allocates plasticity no more selectively than an ungated arm and fails the noisy-TV guard (chases irreducible noise), replicating the e4 conflation. |
| **control** | Ungated arm; noisy-TV reducible-vs-irreducible split as the built-in guard; frozen-random-init-ViT. |
| **tier** | cpu-now |
| **coverage** | e4_neuromod (REFUTED, point-error gate), PR4 / mop_dr12 (registry-only). The DISTRIBUTIONAL disagreement gate the corpus prescribes as the fix has not been run through the noisy-TV guard. |
| **gap** | e4 refuted POINT-error gating; the distributional disagreement gate has never been tested against the noisy-TV guard. |
| **priority** | medium |

### SEM-PLAS-11 (medium) — The substrate carries an amortized developmental prior

| field | value |
|---|---|
| **position** | V-JEPA's progressive-resolution / temporal-predictive pretraining already bakes in the "starting small" benefit, so a molded shell over it needs LESS developmental scheduling than a shell over random features; the substrate is where the real developmental work already happened. |
| **question** | Does a shell over frozen V-JEPA reach criterion with less scheduling/capacity growth than the same shell over frozen-random features, i.e. is the developmental benefit already amortized into the pretrained substrate? |
| **lineage** | Quartz and Sejnowski 1997 (VERIFIED); Elman 1993 (VERIFIED); substrate_vs_random_init_vit (SUPPORTED, p=0.029). |
| **operationalization** | Ablate the shell's developmental scheduling (capacity growth on/off) crossed with substrate (real V-JEPA vs frozen-random-init-ViT at matched resolution). Position predicts scheduling helps LESS on the real substrate. Non-vacuous control: the frozen-random-init-ViT arm IS the control; the vacuity trap is avoided because a full-rank projection preserves information but not the pretrained INDUCTIVE BIAS. |
| **null** | Developmental scheduling gives the same benefit over the real substrate as over frozen-random features (the substrate carries no amortized developmental prior). |
| **control** | Frozen-random-init-ViT at matched resolution/architecture; scheduling-on vs scheduling-off cross; the interaction term (substrate x scheduling) is the load-bearing statistic. |
| **tier** | cpu-now |
| **coverage** | substrate_vs_random_init_vit shows nuisance-invariance; PR2 tests adaptation speed; neither crosses substrate WITH developmental scheduling. |
| **gap** | No experiment asks whether the pretrained substrate REDUCES the shell's need for scheduling (substrate-as-amortized-development); reframes the frozen limitation as a strength. |
| **priority** | medium |

### SEM-PLAS-12 (high) — Deep plasticity required (the boundary of moldability)

| field | value |
|---|---|
| **position** | There exists a class of perspective tasks (needing feature structure absent from the frozen latent) on which no shell at any capacity reaches competence; identifying that class draws the exact boundary of what "moldable" can mean. |
| **question** | Is there a constructible task, decodable from a lightly-fine-tuned encoder but not from any shell over the frozen encoder, that cleanly separates deep plasticity (needed) from shell plasticity (sufficient)? |
| **lineage** | Quartz and Sejnowski 1997 (VERIFIED); Kumar et al. 2022 (VERIFIED, fine-tuning sometimes strictly needed); Karmiloff-Smith 1998 (VERIFIED). |
| **operationalization** | Construct candidate tasks whose target depends on structure the frozen latent under-represents (fine-grained bound-attribute distinctions pooled V-JEPA collapses). Show a high-capacity shell plateaus below criterion. HONEST STATUS: the clean comparison arm (a lightly fine-tuned encoder) VIOLATES the frozen constraint and the live-state encoder ban, so the fine-tuned arm is OPEN / deferred; what is runnable now is the frozen-side half (the shell plateau on a pooling-bound task). Control: frozen-random-init-ViT plateau; sweep shell capacity to saturation to rule out capacity starvation. |
| **null** | For every constructed task, a sufficiently large shell over the frozen latent reaches criterion; no task requires deep plasticity (shell-only molding is sufficient). |
| **control** | Shell-capacity sweep to saturation; frozen-random-init-ViT floor; the fine-tuned-encoder comparison arm honestly deferred (violates frozen + live-state ban). |
| **tier** | needs-new-cache (the fine-tuned-encoder comparison; the frozen-side plateau half is cpu-now) |
| **coverage** | Adjacent: DOCTRINE_SYNTHESIS notes a compositional/binding ceiling (real bound video blocked on DR1); n8 is pooling-bound. No experiment is FRAMED as drawing the shell-vs-deep boundary. |
| **gap** | The deep-vs-shallow boundary is the core question, only implicit in scattered pooling-bound results; the fine-tuned comparison is legitimately deferred. |
| **priority** | high |

### SEM-PLAS-13 (medium) — Moldability is currency-distance-shaped, not a scalar

| field | value |
|---|---|
| **position** | Different perspectives sit at different depths relative to V-JEPA's native visual-temporal currency, so language and math (far) are shell-shallow-moldable at best while spatial/physical (near) have any headroom; the moldability profile is currency-distance-shaped. |
| **question** | Does the shell's gain-over-frozen-random scale with the perspective's DISTANCE from V-JEPA's native currency, producing a monotone moldability-vs-distance profile? |
| **lineage** | 06_currencies framework; Karmiloff-Smith domain-relevant starting states (1998, VERIFIED); Platonic Representation Hypothesis (Huh et al. 2024) for why some symbolic structure may still align. |
| **operationalization** | Order the perspective-proxy tasks by an independent distance-from-native measure (how much of the target is linearly present in raw V-JEPA vs raw random features). Plot shell gain-over-frozen-random against that distance. Control: frozen-random-init-ViT gives the per-task floor defining the gain axis; validate the distance measure against an independent proxy (AL2 alignment R^2). Guard: score by kNN-overlap / gain-over-floor, never raw R^2. |
| **null** | Shell moldability is unrelated to the perspective's distance from the native currency (flat, or all tasks tie the random floor). |
| **control** | Per-task frozen-random-init-ViT floor; an independently-validated distance-from-native measure; kNN/gain scoring not raw R^2. |
| **tier** | studio (needs the cross-modality proxy tasks and, for far perspectives, caches the corpus lacks) |
| **coverage** | The currencies atlas builds the distance machinery but scores alignment/decodability, not a moldability-vs-distance profile. |
| **gap** | The corpus never joins the currency-distance atlas to the plasticity experiments; reframes "moldable across modalities" as a graded profile. |
| **priority** | medium |

---

## 8. META-SEMANTICS

What meaning IS (reference vs use vs inferential/conceptual role), understanding vs memorization, the unity or incommensurability of thought across perspectives, agreement-as-truth-signal, the private-language argument. **Verification-forced deduplication:** the two central flagships of this layer, the use-vs-decode criterion (Wittgenstein PI 243 + Brandom + Dennett) and the two-agent signalling game (Lewis + Lazaridou), were RESTATED across the Communication, Meta-Semantics, and Code briefs (three copies of the use criterion, four of the signalling game). They are ONE design each, not seven. The canonical versions are SEM-COMM-1 (use) and SEM-COMM-2 (two-agent); this section CROSS-REFERENCES them rather than restating them, and adds only the genuinely distinct meta-semantic positions (inscrutability, inferential role, agreement-as-truth, unity-vs-alignment, understanding-as-extrapolation, theory-ladenness). This is the single biggest doctrine-fit correction in the set: the catalogue's instruction is to EXTEND, and the same design restated per-domain is the opposite of extension.

### SEM-META-1 (flagship) — Meaning is use, not reference [CANONICAL: see SEM-COMM-1]

The "meaning is use, test by intervention not readout" position is the SAME design as SEM-COMM-1 (use-difference, not a readout), with the SAME steering/behavioral-delta operationalization and the SAME 3d-vacuity guard (nonlinear consuming shell, output-ablation for "unconsulted", confirm the behavioral-delta metric is not invariant to an invertible linear map before trusting any nonzero delta). Not restated as an independent flagship. Lineage: Wittgenstein PI (1953, VERIFIED); Brandom (1994, VERIFIED). Tier cpu-now. Coverage: gap, blind-spot 1. **Cross-reference SEM-COMM-1.**

### SEM-META-2 (flagship) — Communication is a two-agent negotiated act [CANONICAL: see SEM-COMM-2]

The "meaning is fixed by sender-receiver coordination" position is the SAME design as SEM-COMM-2 (referential game, frozen-random-receiver + matched-channel-capacity floors). The Davidson triangulation lineage (Davidson, *Subjective, Intersubjective, Objective*, Oxford UP, 2001, VERIFIED) is an additional grounding but the operationalization and controls are identical. Not restated. **Cross-reference SEM-COMM-2.** The Code-domain executable-channel variant is SEM-CODE-7; the arbitrary-convention (cross-pair drop) variant is SEM-META-9 below, which IS distinct (different signature: cross-pair unintelligibility, not cross-seed match).

### SEM-META-3 (flagship) — Reference/aboutness is not decodability [CANONICAL: see SEM-COMM-3]

The Fregean/Millikan "a state is ABOUT X only if its use tracks X counterfactually" position is the consumer-side intervention already canonicalized as SEM-COMM-3. Additional lineage: Frege, "Über Sinn und Bedeutung" (1892), sense vs reference (VERIFIED); Millikan biosemantics (1989, VERIFIED). Not restated. **Cross-reference SEM-COMM-3.**

### SEM-META-4 (high) — Meaning is inferential/conceptual role

| field | value |
|---|---|
| **position** | A factor's content is fixed by what the shell can INFER from it and what licenses it, not by any world-object it points at, so understanding = correct downstream inferential behavior. |
| **question** | Does a latent factor support the systematic downstream inferences its putative concept licenses (if X is-a square then X has-corners), above a shell with the same decodable factor but no inferential articulation? |
| **lineage** | Brandom, *Making It Explicit* (1994), inferentialism (VERIFIED). Sellars, "Empiricism and the Philosophy of Mind" (1956), the space of reasons (VERIFIED). |
| **operationalization** | Shell trained on base task A. Test whether its representation of factor X licenses a held-out inference task B (count-of-corners from shape) with FEW examples, above a control shell given X as an explicit decoded scalar but no learned inferential structure. |
| **null** | The inferentially-articulated shell transfers to B no better than the explicit-decoded-scalar shell (delta = 0). |
| **control** | Explicit-decoded-scalar shell (information-matched, role-stripped); random-init-ViT features; not a projection. |
| **tier** | cpu-now |
| **coverage** | Gap. Inferentialism is untested; c-series tests latent arithmetic, not inferential licensing. |
| **gap** | The corpus never asks whether a representation carries inferential role, only whether a scalar is present. |
| **priority** | high |

### SEM-META-5 (high) — Reference is inscrutable / translation is indeterminate (gavagai)

| field | value |
|---|---|
| **position** | A shell's latent code admits multiple mutually-incompatible readings all fitting the behavior equally, so a single decoded label overclaims what the code means. |
| **question** | Do two shells at identical task success carve a factor into behaviorally-indistinguishable but geometrically-incompatible partitions (rabbit vs undetached rabbit-parts), and is the apparent shared meaning an artifact of the experimenter's chosen readout? |
| **lineage** | Quine, *Word and Object* (MIT Press, 1960), indeterminacy of translation and inscrutability of reference, the gavagai case (VERIFIED). |
| **operationalization** | Train N shells to equal accuracy. For a target factor, fit the best decodable partition per shell; measure whether cross-shell partitions AGREE on held-out AMBIGUOUS stimuli (where whole-object vs object-part readings diverge) above a permutation null. Control: matched random-map-of-equal-rank per shell (AT1-relrep, kNN-overlap not R^2); agreement above that floor = genuine shared reference, at that floor = inscrutability confirmed. |
| **null** | Cross-shell partitions agree on ambiguous held-out stimuli no more than random-equal-rank maps (reference inscrutable). |
| **control** | Random-map-of-equal-rank scored by kNN-overlap (never square projection); permutation null across shells. |
| **tier** | needs-new-cache (ambiguous held-out stimuli where distinct readings diverge do not exist; the current cache is unambiguous shape-color) |
| **coverage** | Partial. p5/s5/y3 test code stability (REFUTED, idiolect) but not the divergent-reading gavagai construction. |
| **gap** | No experiment builds the ambiguous stimuli where readings diverge; stability tests cannot detect inscrutability invisible on the training distribution. |
| **priority** | high |

### SEM-META-6 (high) — Agreement across perspectives is a truth signal (triangulation)

| field | value |
|---|---|
| **position** | When two independent frozen encoders converge on a correctness judgment, that convergence is more diagnostic of the world than either alone, above an agreement-by-chance floor. |
| **question** | Does cross-encoder AGREEMENT predict ground-truth correctness better than either encoder's own CONFIDENCE, above a shuffled-agreement baseline, making inter-perspective convergence an epistemic warrant? |
| **lineage** | Davidson triangulation, objectivity from convergence of two agents on a shared world (2001, VERIFIED). Global Workspace structure borrowed (Baars/Dehaene, VERIFIED) as architecture. |
| **operationalization** | WS1 read as epistemology: two frozen encoders, a central gate, on cached clips with ground truth. Test whether AGREEMENT(enc1,enc2) predicts correctness above max(conf1,conf2) and above a shuffled-agreement (mispaired-clip) null. Control: shuffled agreement + single-encoder confidence-only baseline; random-init second encoder to show the warrant needs a genuinely informative second perspective. |
| **null** | Cross-encoder agreement predicts correctness no better than single-encoder confidence at matched calibration (agreement carries no extra truth signal). |
| **control** | Shuffled-agreement null + confidence-only baseline + random-init second encoder; not a projection. |
| **tier** | cpu-now (V-JEPA + DINOv2 caches exist) |
| **coverage** | Partial. WS1 (agreement-vs-confidence) is the central-gate experiment but framed operationally, not as a triangulation truth-warrant. |
| **gap** | WS1 tests whether agreement helps a gate; it never tests whether agreement is a TRUTH signal above confidence, the Davidsonian claim the framework leans on. |
| **priority** | high |

### SEM-META-7 (flagship) — Understanding is extrapolation, not interpolation (the CM1 closure)

| field | value |
|---|---|
| **position** | Genuine concept possession shows in held-out-COMBINATION generalization where seen-equals-heldout accuracy, distinguishing a factored representation from memorized conjunctions. |
| **question** | Does the substrate's held-out-combination accuracy EQUAL its seen-combination accuracy on non-additively bound attributes at non-ceiling difficulty, the within-arm signature of factoring that no control is needed to interpret? |
| **lineage** | Fodor and Pylyshyn, "Connectionism and Cognitive Architecture" (1988), systematicity/productivity as the mark of genuine competence (VERIFIED). Wittgenstein rule-following (1953, VERIFIED). |
| **operationalization** | Directly extends compositional_under_nuisance (already SUPPORTED descriptively: shape factors from color, held-out 0.708 = seen, p=1.2e-12). The gate-clearing rerun (CM1): swap the resolution-confounded random-pixel control for a random-init-ViT-L at MATCHED resolution, multi-seed, to make the substrate attribution non-descriptive. The heldout-equals-seen equality is within-arm and needs no control. |
| **null** | Held-out-combination accuracy falls below seen-combination accuracy (representation memorizes conjunctions, does not factor). |
| **control** | Random-init-ViT-L at matched resolution, multi-seed (replaces the resolution-confounded random-pixel arm); never a projection. |
| **tier** | studio |
| **coverage** | Covered, needs closure. compositional_under_nuisance.json is the program's first off-ceiling compositional positive; its control is resolution-confounded (descriptive only). CM1 is the named gate-clearing rerun. |
| **gap** | The substrate-specific attribution is not yet gate-clearing (single seed, resolution-confounded control); the within-arm equality survives but the between-arm claim does not. |
| **priority** | flagship |

### SEM-META-8 (high) — Perception is theory-laden (the given is a myth)

| field | value |
|---|---|
| **position** | There is no given, so any structure a probe recovers may be contributed by the probe, not read off a bare substrate, which is why readout-based meaning claims are epistemically empty. |
| **question** | Is the nonlinear structure a probe recovers contributed by the probe rather than the encoder (identical gain on real and on a genuinely non-linear-image control)? |
| **lineage** | Sellars, "Empiricism and the Philosophy of Mind" (1956), the Myth of the Given (VERIFIED). Theory-ladenness of observation (Hanson/Kuhn-adjacent, UNVERIFIED as a single canonical source; Sellars is load-bearing). |
| **operationalization** | p10_theory_ladenness found nonlinear probe gain identical on real and frozen-random (probe-contributed). The doctrinal upgrade: replace the VACUOUS frozen-random-PROJECTION (full-rank invertible) with random-init-ViT features that are NOT a linear image of the latents, so a probe cannot absorb the map. If gain persists on real over that genuine control, structure is encoder-given; if it ties, probe-given (Sellars vindicated). |
| **null** | Nonlinear probe gain on real substrate does not exceed gain on random-init-ViT features at matched resolution (structure is probe-contributed). |
| **control** | Random-init-ViT-L at matched resolution (a genuine non-linear-image control), NOT the vacuous square projection that invalidated p10's original tie. |
| **tier** | studio |
| **coverage** | Covered but invalidated. p10 tied against the vacuous frozen-random-projection (that control could not have shown a gap); the position survives but its original test must be re-run against a real control. |
| **gap** | The strongest theory-ladenness evidence (p10) rests on the vacuous control; it was never actually tested against a non-invertible baseline. |
| **priority** | high |

### SEM-META-9 (high) — A private protocol is an arbitrary convention (cross-pair drop)

| field | value |
|---|---|
| **position** | A private protocol between two MoP instances, if one exists, is a CONVENTION selected by joint success and could have been otherwise; its hallmark is arbitrariness-plus-stability under a shared channel, not recurrence of the same code across seeds. The corpus refuted the wrong thing: idiolect-recurrence, not negotiated convention. |
| **question** | Do two co-adapting shells converge to a STABLE but ARBITRARY protocol (different pairs settle on different codes, each mutually intelligible) rather than a seed-universal code (already refuted) or an unstable idiolect? |
| **lineage** | Lewis, *Convention* (1969), arbitrary-but-stable coordination equilibria, multiple equilibria (VERIFIED). Skyrms (2010) different stable signalling equilibria (VERIFIED). Wittgenstein private-language argument reinterpreted: a two-agent shared practice escapes the objection (VERIFIED). |
| **operationalization** | Run the SEM-COMM-2 game across many independent PAIRS. Measure (a) within-pair mutual intelligibility and (b) cross-pair intelligibility (swap a receiver from pair i onto sender from pair j: convention predicts a DROP). A genuine convention is high within-pair, low cross-pair, stable within a pair over time. |
| **null** | Either cross-pair intelligibility equals within-pair (seed-universal, not a negotiated convention, consistent with p5/s5/y3) OR within-pair intelligibility is unstable (an idiolect that never conventionalizes). |
| **control** | Cross-pair swap (the arbitrariness test) and frozen-random-receiver floor (the stability-vs-noise test). A pair has a convention only if within-pair beats the random floor AND exceeds cross-pair intelligibility beyond seed spread. |
| **tier** | cpu-now |
| **coverage** | Reframes p5/s5/y3 (private language, refuted as seed-recurrence). Those tested the Platonic hallmark (cross-seed MATCH); this tests the Lewisian hallmark (cross-pair DROP). Distinct question, distinct signature. |
| **gap** | The corpus falsified code-recurrence but never tested negotiated arbitrary convention, which is what "two MoP instances develop a private protocol" actually means. |
| **priority** | high |

### SEM-META-10 (high) — Translation is structure-mapping, not vector arithmetic

| field | value |
|---|---|
| **position** | A productive cross-perspective mapping transfers a SYSTEM of relations (higher-order structure), so analogy-as-latent-offset failing (c2) does not refute analogy-as-structure-mapping. |
| **question** | Does a relational SYSTEM in one content domain map onto the same system in a held-out domain, scored by preservation of higher-order relational consistency rather than offset-vector transfer? |
| **lineage** | Gentner, "Structure-Mapping: A Theoretical Framework for Analogy", *Cognitive Science* (1983), the systematicity principle (VERIFIED). |
| **operationalization** | Cached clips instantiating a relation R (A-contains-B, A-larger-than-B) across two disjoint object sets. Train a shell to read R on set 1, test on set 2, scoring structural consistency (one-to-one relation preservation) not attribute decode. Control: object-appearance-matched shuffle that preserves objects but destroys the relation (Gentner's appearance-match); random-init substrate. A structure-mapper beats the appearance-shuffle; a feature-matcher does not. |
| **null** | Relational-system transfer to held-out objects does not exceed the appearance-preserving relation-shuffled control (no structure mapping). |
| **control** | Appearance-matched relation-shuffle + random-init-ViT; not a projection. |
| **tier** | needs-new-cache (relation-instantiating clips, e.g. containment/ordering across object sets, do not exist) |
| **coverage** | Gap. c2_latent_analogy (REFUTED) tested analogy as offset arithmetic; it never tested higher-order relational-system preservation (blind-spot 7). |
| **gap** | The corpus refuted the wrong operationalization of analogy; Gentner's systematicity was never the tested construct. |
| **priority** | high |

### SEM-META-11 (medium) — The private-language argument, constructive corollary

| field | value |
|---|---|
| **position** | A shell's code with no public criterion of correctness (recurs across seeds, or is shareable) is an idiolect not a language; the corpus showed the code is private, but the constructive corollary is that a public criterion can only arise through two-agent use. |
| **question** | Can a code that fails the solo recurrence test become public through negotiation (the private-language argument predicts negotiation is REQUIRED for correctness)? |
| **lineage** | Wittgenstein, *Philosophical Investigations* (1953), the private-language argument (VERIFIED). |
| **operationalization** | Pair the refuted solo recurrence (p5/s5/y3) with SEM-COMM-2 / SEM-META-9: test whether a code that fails solo recurrence nonetheless becomes public through negotiation. Control: frozen-random floor for recurrence; random-init receiver for shareability. |
| **null** | A solo shell's code recurs across seeds / transfers to a second shell no better than the frozen-random floor (code private, meaning-bearing status denied). |
| **control** | Frozen-random floor (recurrence) + random-init receiver (transfer); never a square projection for the transfer metric. |
| **tier** | cpu-now |
| **coverage** | Covered but one-sided. p5/s5/y3 REFUTED shared code three ways; this reframes the negative as CONFIRMING Wittgenstein and links it to whether negotiation manufactures the missing public criterion. |
| **gap** | The corpus showed the code is private but never tested the constructive corollary that a public criterion arises only through two-agent use. |
| **priority** | medium |

### SEM-META-12 (high) — One thought across perspectives vs incommensurable correlation

| field | value |
|---|---|
| **position** | There is one modality-independent conceptual content expressible across perspectives, vs the incommensurabilist claim that a visual thought and a symbolic thought are different thoughts that merely correlate. |
| **question** | Is there shared content such that a task learned in the visual perspective transfers to the symbolic perspective BETTER than their mere statistical alignment predicts, evidencing one content rather than two aligned ones? |
| **lineage** | Fodor amodal LoT (1975; Fodor-Pylyshyn 1988) vs Barsalou grounded modality-specific cognition (1999) (both VERIFIED); Davidson, "On the Very Idea of a Conceptual Scheme" (1974), against incommensurable schemes (VERIFIED). |
| **operationalization** | On paired vision+symbol cached data, contrast (H_unity) a shared bottleneck lets a task trained on the visual side zero-shot the symbolic side, vs (H_alignment) transfer is fully accounted for by the AL2 alignment map. Test = task-transfer through a frozen shared bottleneck MINUS transfer predicted by the fitted alignment map alone. Control: alignment-map-only prediction is the floor (residual transfer above it = shared content); random-init encoders bound the trivial-correlation baseline. |
| **null** | Cross-perspective task transfer does not exceed what the fitted statistical alignment map already predicts (no shared content beyond correlation; incommensurabilist null holds). |
| **control** | Alignment-map-only transfer prediction + random-init-encoder correlation floor; never a square projection. |
| **tier** | needs-new-cache (paired vision+symbol data on the same referents) |
| **coverage** | Gap. AL2 tests alignment, never whether transfer EXCEEDS alignment (the unity-vs-correlation cut). |
| **gap** | The corpus can measure alignment but has never separated "one thought in two perspectives" from "two correlated thoughts". |
| **priority** | high |

### SEM-META-13 (medium) — Indexicality / perspective-binding [CANONICAL: see SEM-COMM-6]

The "a latent may encode a POINT OF VIEW (ego/allo)" position is the same design as SEM-COMM-6, with the Perry-1979 essential-indexical lineage (VERIFIED) and the random-init-ViT off-ceiling control. Not restated. **Cross-reference SEM-COMM-6.** Tier needs-new-cache (multi-viewpoint pose-labeled clips).

### SEM-META-14 (high) — Temporal / narrative meaning (event boundaries as content) [CANONICAL: see SEM-PHYS-8]

The "event boundaries function as semantic UNITS, not decodable time-stamps" position shares its operationalization with SEM-PHYS-8 (causal content) and SEM-MATH-12 (ordinal-temporal). Lineage: Zacks and Tversky, "Event structure in perception and conception", *Psychological Bulletin* 127(1) (2001) 3-21, DOI 10.1037/0033-2909.127.1.3 (VERIFIED, exact title confirmed; was UNVERIFIED in the brief, now VERIFIED). Barsalou 1999 (VERIFIED). Operationalization: segment the latent trajectory into events, test whether boundaries predict held-out sub-event semantics above a shuffled-time AND random-boundary control. **Verification-forced retier:** the brief marked this cpu-now, but it needs (a) temporally-extended clips with annotated event boundaries and (b) a V-JEPA forward-prediction-error track over time (the predictor pass touches the banned encoder lane). No event-boundary annotation exists in the tree. Retiered `needs-new-cache`. The shuffled-time and random-boundary controls are sound. Coverage: gap (blind-spot 8). Priority high.

### SEM-META-15 (exploratory, OPEN) — Normativity: a concept can be wrong [see SEM-COMM-8]

The "a concept can be misapplied, not merely low-accuracy" position is the honestly-OPEN normativity gap, canonicalized as SEM-COMM-8 with the Millikan/Brandom/Wittgenstein lineage (all VERIFIED). The Meta-semantics candidate operationalization (do the shell's errors cluster on rule-violating cases vs scatter, scored by error-structure not error-rate, against a label-shuffled noise floor) is a second candidate alongside the SEM-COMM-8 signalling-game candidate; both are marked OPEN because construct validity (does error-structure capture genuine normativity vs just a harder distribution) is unresolved on a frozen substrate. **Cross-reference SEM-COMM-8.** Tier open-philosophical. Priority exploratory.

---

## 9. COVERAGE-AND-GAP MAP

Every position vs whether we test it today (with the experiment id) vs the gap. This is the scoring table for the audit. "Canonical" rows point to the primary id and are not double-counted in Section 10.

| id | tested today (experiment) | verdict today | gap |
|---|---|---|---|
| SEM-COMM-1 | none (DR4 tests counterfactual-match, not use) | untested | no use-vs-decode metric exists |
| SEM-COMM-2 | none (p5/s5/y3 test one shell) | untested | no sender-receiver game |
| SEM-COMM-3 | none (DR4 nearest) | untested | consumer-semantics unoperationalized |
| SEM-COMM-4 | none | untested | pragmatics unposed |
| SEM-COMM-5 | s1/ex16/s10 (readout only) | REFUTED as readout | drive-direction untested |
| SEM-COMM-6 | a2 (decodability) | decodability only | perspective-binding untested |
| SEM-COMM-7 | WS1/A4 (arbitration frame) | design exists | bus framing untested |
| SEM-COMM-8 | ex18/DR9 (correction existence) | REFUTED (existence) | norm-specificity OPEN |
| SEM-LANG-1 | c1/s6/c9, p6/s4/s7 | REFUTED / ceilinged | off-ceiling regime missing |
| SEM-LANG-2 | p1 | REFUTED (labels) | symbolic-shell-vs-raw-latent head-to-head missing |
| SEM-LANG-3 | none | untested | language-encoder cache missing |
| SEM-LANG-4 | i9, p6/s4 | partial | detail-dependence axis missing |
| SEM-LANG-5 | none | untested | top-down label-feedback absent |
| SEM-LANG-6 | none | untested (hard) | constitutive-vs-probe-weakness OPEN |
| SEM-LANG-7 | s1/s10 (decodability) | framing only | use-asymmetry never run |
| SEM-LANG-8 | AL2/AT1 (machinery) | machinery only | form-only-vs-grounded contrast missing |
| SEM-LANG-9 | p9/s4, DR2/DR11 | REFUTED (easy regime) | branching-factor axis unswept |
| SEM-LANG-10 | WS1 (vision-vision), PR1 | vision-vision only | LLM-as-second-perspective missing |
| SEM-LANG-11 | none | untested | pragmatics via router unposed |
| SEM-LANG-12 | A14/WS1 | shared-code frame | single-vs-cross dissociation missing |
| SEM-LANG-13 | none | untested (expected-fail pole) | language-encoder cache missing |
| SEM-CODE-1 | ex18/DR7/DR9 (learned verifier) | REFUTED (learned) | DSL+executor+richer cache absent |
| SEM-CODE-2 | MP4/MP7/DR6/DR7 (learned scorer) | null lane, unidentified cause | executor never instantiated |
| SEM-CODE-3 | PR1 (other modes) | untested | program-induction mode absent |
| SEM-CODE-4 | none | untested | DSL-target-vs-flat missing |
| SEM-CODE-5 | ex2 (synthetic toy) | positive on TOY, substrate-silent | V-JEPA rollout + library learning absent |
| SEM-CODE-6 | none (AT4 as ceiling) | untested | program-length metric absent |
| SEM-CODE-7 | p5/s5/y3 (one shell) | untested | executable signalling game absent |
| SEM-CODE-8 | n9/y1 (implicit iteration) | REFUTED (implicit) | recursive primitive absent |
| SEM-CODE-9 | MP4 (learned verify) | untested | execute-and-check mode absent |
| SEM-CODE-10 | s1/s7/s10/ex16 (learned code) | PASS-VACUOUS (learned) | designed-code intervention absent |
| SEM-CODE-11 | e4 (learned gate) | REFUTED (learned) | typed-precondition abstention absent |
| SEM-CODE-12 | n8/d2 (pooling-bound) | pooling-bound | negation predicate absent; full test DR1-blocked |
| SEM-MATH-1 | none | untested | numerosity cache absent |
| SEM-MATH-2 | none | untested | geometry cache absent |
| SEM-MATH-3 | AL2 (generic alignment) | untested | math-vs-language dissociation absent |
| SEM-MATH-4 | AL2 (machinery) | untested | math substrate never instantiated |
| SEM-MATH-5 | d4/d6/n5/e3/e4/b4 (substrate reopening) | NEGATIVE (substrate) | math-content-driven latent-use reshaping absent |
| SEM-MATH-6 | none | untested | grounded-vs-ungrounded stimulus cache absent |
| SEM-MATH-7 | none | untested | magnitude-axis / distance-effect absent |
| SEM-MATH-8 | compositional_under_nuisance (attribute) | adjacent | numeric composition absent |
| SEM-MATH-9 | c2 (categorical), a2 (decodability) | REFUTED (categorical) | geometric operator absent |
| SEM-MATH-10 | none | untested | perceptual-correlate-strength grading absent |
| SEM-MATH-11 | none | untested | area/density guard absent (blocks SEM-MATH-1) |
| SEM-MATH-12 | AT3 (time-decodability) | adjacent | ordinal-as-temporal-content absent |
| SEM-MATH-13 | p2 (categorical, no frozen-random) | adjacent | number conservation absent |
| SEM-PHYS-1 | PR1 (reasoning modes) | untested | symbolic-physics currency never instantiated |
| SEM-PHYS-2 | substrate_vs_random_init_vit (static) | SUPPORTED (static shape), p=0.029 | dynamics variables untested |
| SEM-PHYS-3 | none | untested | cognitive-physics error-model fit absent |
| SEM-PHYS-4 | AL2/AL3 (statistical) | untested | grounding-direction absent |
| SEM-PHYS-5 | a1_affordance_decode | REFUTED-ish (static, single-control) | dynamic affordance + appearance dissociation absent |
| SEM-PHYS-6 | ex2 (toy), DR4 | partial (toy/counterfactual) | signed physical intervention absent |
| SEM-PHYS-7 | d2/n8 (permanence, bound) | pooling-bound (single principle) | multi-principle VoE battery absent |
| SEM-PHYS-8 | AT3/AL3 (decodability) | decodability only | causal-order/time-arrow content absent |
| SEM-PHYS-9 | substrate_vs_random_init_vit (static), a2 (vacuous tie) | static / vacuous-tie | physical invariant, kNN-scored absent |
| SEM-PHYS-10 | dense_vs_pooled (single factor) | single-factor survives pooling | relational multi-object test gated on 2.1 dense |
| SEM-PHYS-11 | n8/d2 (pooling-bound) | pooling-bound | counterfactual absence; DR1-blocked, OPEN |
| SEM-PHYS-12 | WS1/WS3 (correctness) | untested on misconception | misconception + arbitration absent |
| SEM-PLAS-1 | PR2, p1 (oblique) | untested (as ceiling) | reshaping-vs-reweighting ceiling absent |
| SEM-PLAS-2 | d6, d5/b3/b7/d7 (within-modality) | REFUTED (within-modality) | cross-modality ordering absent |
| SEM-PLAS-3 | d4 (unwired), c2 | untested | matched-compute cross-modality transfer absent |
| SEM-PLAS-4 | e3/n5 (generic) | REFUTED (generic) | per-modality deficit-and-restore absent |
| SEM-PLAS-5 | framing only | untested | measured information ceiling absent |
| SEM-PLAS-6 | A4 (planned) | unrun | capacity-growth schedule absent |
| SEM-PLAS-7 | e7/PR3/ex13/ex15 | provisional positive (synthetic) | real-latent multi-perspective retention absent |
| SEM-PLAS-8 | none | untested | top-down molding absent |
| SEM-PLAS-9 | e3 (retention half) | REFUTED (retention) | specialization half absent |
| SEM-PLAS-10 | e4/PR4/dr12 | REFUTED (point-error) | distributional gate through noisy-TV absent |
| SEM-PLAS-11 | substrate_vs_random_init_vit, PR2 | partial | substrate x scheduling interaction absent |
| SEM-PLAS-12 | pooling-bound results | adjacent | shell-plateau + deferred fine-tuned arm |
| SEM-PLAS-13 | AT1/AL2 (alignment) | machinery only | moldability-vs-distance profile absent |
| SEM-META-1 | canonical SEM-COMM-1 | see SEM-COMM-1 | see SEM-COMM-1 |
| SEM-META-2 | canonical SEM-COMM-2 | see SEM-COMM-2 | see SEM-COMM-2 |
| SEM-META-3 | canonical SEM-COMM-3 | see SEM-COMM-3 | see SEM-COMM-3 |
| SEM-META-4 | none | untested | inferential role absent |
| SEM-META-5 | p5/s5/y3 (stability) | partial | ambiguous-reading stimuli absent |
| SEM-META-6 | WS1 (operational) | partial | truth-warrant framing untested |
| SEM-META-7 | compositional_under_nuisance | SUPPORTED descriptively | CM1 gate-clearing rerun pending |
| SEM-META-8 | p10 (vacuous control) | INVALIDATED (vacuous control) | re-run vs non-invertible baseline |
| SEM-META-9 | p5/s5/y3 (cross-seed) | reframes negative | cross-pair-drop signature absent |
| SEM-META-10 | c2 (offset) | REFUTED (offset) | structure-mapping construct absent |
| SEM-META-11 | p5/s5/y3 | one-sided negative | constructive corollary absent |
| SEM-META-12 | AL2 (alignment) | machinery only | transfer-exceeds-alignment cut absent |
| SEM-META-13 | canonical SEM-COMM-6 | see SEM-COMM-6 | see SEM-COMM-6 |
| SEM-META-14 | canonical SEM-PHYS-8 / SEM-MATH-12 | see those | event-boundary annotation absent |
| SEM-META-15 | canonical SEM-COMM-8 | see SEM-COMM-8 | see SEM-COMM-8 |

---

## 10. PRIORITY TEST LIST

The flagship positions ranked, split by tier, that the audit holds the plans against. Canonical-duplicate ids (SEM-META-1/2/3/13/15) are folded into their SEM-COMM primaries and not listed twice.

### Flagships, cpu-now (runnable today on cached features, no new modality)

1. **SEM-COMM-1** — use-vs-decode intervention (the load-bearing definition of communication). Guard: nonlinear shell, output-ablation, map-invariance check.
2. **SEM-COMM-2** — two-agent referential game (the missing sender-receiver success test). Floors: frozen-random receiver, matched-channel-capacity, mismatched-history.
3. **SEM-COMM-3** — aboutness as consumer-side intervention (feature-specific signature vs generic damage).
4. **SEM-PLAS-1** — reshaping-vs-reweighting plasticity ceiling (is molding shell-deep only).
5. **SEM-PLAS-2** — cross-modality developmental order (spatial then language then symbolic).
6. **SEM-PLAS-3** — cross-modal transfer (does learning code help math), matched-compute scrambled-structure control.
7. **SEM-META-9** — arbitrary-convention cross-pair-drop (the Lewisian hallmark p5 never tested).

### Flagships, studio (more compute, no new modality)

8. **SEM-META-7 (CM1)** — understanding-as-extrapolation gate-clearing rerun (random-init-ViT-L at matched resolution, multi-seed) to convert compositional_under_nuisance from descriptive to gate-clearing.
9. **SEM-META-8** — theory-ladenness re-run against a non-invertible control (fix the vacuous-projection tie that invalidated p10).
10. **SEM-COMM-5** — symbol-to-percept drive on V-JEPA's own predictor (needs the predictor pass).

### Flagships, needs-new-cache (blocked on a stimulus set or second encoder the corpus lacks; the live-state encoder ban blocks producing these now)

11. **SEM-LANG-3** — weak-Whorf category-boundary warp (needs a language-grounded encoder cache).
12. **SEM-LANG-7 + SEM-LANG-8** — the Bender-Koller / Piantadosi-Hill decisive PAIR (form-only fails use-reference, passes conceptual-role) (needs a text-encoder cache).
13. **SEM-LANG-10** — LLM hidden states as a distinct perspective (needs an LLM-hidden-state cache on parallel content).
14. **SEM-MATH-1 + SEM-MATH-11** — numerosity decode AND its area/density anti-self-deception guard (needs a numerosity 1..8 cache; the guard blocks interpreting the decode).
15. **SEM-MATH-2** — geometry-over-algebra asymmetry (needs geometric-figure + symbolic-relation caches).
16. **SEM-PHYS-1** — intuitive-vs-formal physics decorrelation (needs a physics clipset + symbolic-parameter encoder).
17. **SEM-PHYS-2** — the substrate as a physics engine (dynamics variables) (needs a physics clipset).
18. **SEM-PHYS-4** — intuitive physics as the grounding floor, symbol drives percept (needs physics clipset + symbolic encoder).
19. **SEM-CODE-1 + SEM-CODE-2** — executable program scored against ground truth, and whether it flips the null test-time-compute lane (needs a DSL + executor + a count/relation cache; NONE exists today).

### Open-philosophical (not cleanly operationalizable on a frozen substrate yet, marked honestly, held to the "do not fake a norm" discipline)

20. **SEM-COMM-8 / SEM-META-15** — normativity, a concept can be misapplied (candidate operationalizations offered but construct validity unresolved).
21. **SEM-LANG-6** — language as the constitutive medium of thought (proving un-decodability confounds with probe weakness).
22. **SEM-PHYS-11 / SEM-CODE-12** — counterfactual/negation content on the pooled present-tense substrate (the discriminating occlusion test is DR1-blocked).

---

## 11. HONEST LIMITS

### What a FROZEN substrate structurally cannot answer

1. **Deep plasticity (SEM-PLAS-1, SEM-PLAS-12).** A frozen substrate can only be re-weighted, not reshaped. Any "moldable as a child's brain" claim is, at the substrate level, false by construction; the honest residual is SHELL plasticity, bounded above by the frozen latent's mutual information with the target (SEM-PLAS-5, redescription not enrichment). The one test that would separate deep-plasticity-required from shell-sufficient (SEM-PLAS-12) needs a lightly fine-tuned encoder as its comparison arm, which VIOLATES the frozen constraint AND the live-state encoder ban, so that arm is legitimately deferred, not run.

2. **Symbol-to-percept drive without a second encoder (SEM-COMM-5, SEM-PLAS-8, SEM-PHYS-4).** The generative/top-down direction of grounding, and true bidirectional molding, need either V-JEPA's own predictor run (touches the encoder lane) or a second symbolic encoder. The within-substrate half of SEM-PLAS-8 is runnable now; the cross-encoder half is not.

3. **Counterfactual / negation / absence (SEM-PHYS-11, SEM-CODE-12, blind-spot 4).** The pooled present-tense substrate has no signature for what did NOT happen. The pooled proxies are lower bounds, not refutations; the discriminating occlusion/permanence test needs real bound video with occlusion, blocked on DR1.

4. **Normativity / correctness-of-a-concept (SEM-COMM-8, SEM-META-15, blind-spot 5).** Whether error-STRUCTURE captures genuine norm-violation vs merely a harder distribution is philosophically unresolved on a frozen substrate. Marked OPEN; the discipline is to report OPEN if the shuffled-correction / label-shuffled control is not beaten, never to fabricate a norm.

### What needs modality caches we have NOT built (and cannot build now under the encoder ban)

The verification pass confirmed the corpus caches are: the 5x5 shape-color bound-nuisance clipset (`bound_nuisance_v1`, 200 clips), its random-init ViT-L twin, handcrafted descriptors, the 25-d programmatic reference (shape + color one-hots + 15 nuisance columns; NO count/relation/exists), a real-encoder clipset, single-frame V-JEPA, and Qwen-textified / wav2vec2 sonified caches. Consequently:

- **No DSL, no executor, no count/relation scene.** Every Code-domain flagship (SEM-CODE-1/2, and the whole cluster) that assumed an executor scoring a shell-emitted program is retiered `needs-new-cache-and-tooling`. The corpus's own null-lane diagnosis is that no reliable scorer was ever instantiated; these positions still do not instantiate one, they assume it. The `bound_nuisance_v1` cache exposes exactly two crisp slots (shape, color), which the substrate already factors off-ceiling (0.708); it cannot bind count or relation.

- **No physics clipset.** All physics-decoding flagships (SEM-PHYS-1/2/4, and the dynamics/VoE/causal-order cluster) need parametric physics clips run through the frozen encoder; the clips are absent and the encoder lane is live.

- **No numerosity/geometry/ordinal stimulus set.** All math-decoding flagships (SEM-MATH-1/2/11, and the cluster) need rendered stimuli run through the encoder; absent and blocked.

- **No paired language encoder on identical content.** The relativity (SEM-LANG-3/13), grounding-asymmetry PAIR (SEM-LANG-7/8), and LLM-as-perspective (SEM-LANG-10) positions need a language-grounded or LLM-hidden-state cache on parallel referents; the Qwen-textified cache is a LABEL-FREE PIXEL-DERIVED textification (color grid + brightest-cell position, already paired to the clips, lacking SHAPE which decodes at chance here), not full parallel LLM states on the same clips.

- **No event-boundary annotation.** The temporal-meaning positions (SEM-META-14, SEM-PHYS-8, SEM-MATH-12) need annotated event/causal-order boundaries and a predictor-error track; both absent.

- **ex2 is a synthetic toy, not the substrate.** `close_ex2_planning.py` runs on `_true_dynamics_params` / `_step_true`, an 8-d-action hand-written environment with a small learned `_DynamicsModel`; it touches no V-JEPA latent. The surviving ex2 planning positive is a positive about a toy MPC, not about the substrate. Code-as-planning and physics-controllability claims that lean on it (SEM-CODE-5, SEM-PHYS-6) inherit this toy-vs-substrate caveat; the substrate-grounded versions need an action-conditioned V-JEPA rollout that is not built.

### The doctrinal spine, restated

The two flagships that carry the whole layer (SEM-COMM-1 use-vs-decode, SEM-COMM-2 two-agent negotiation) are the ONLY places the program has a definition of communication distinct from "a probe decodes" or "an alignment map fits". They are cpu-now and they carry the non-vacuous controls (frozen-random receiver, decode-only floor, matched-channel-capacity) that keep them out of the vacuous-projection trap. Everything else in this catalogue either extends those two, closes a named blind spot, or is honestly marked OPEN. The audit should hold every plan against whether it respects the two doctrinal spines: a random-init encoder at matched resolution (never a square projection) as the substrate control, and USE (intervention / negotiation / behavioral delta) rather than readout as the meaning criterion, with the map-invariance guard so the use-metric does not reinherit the vacuity trap in semantic clothing.
