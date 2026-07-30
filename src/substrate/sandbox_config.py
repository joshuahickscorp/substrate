"""Frozen authority for the Substrate Tangible Sandbox R2 campaign.

R2 is a tangible-artifact experiment, not a continuation of the synthetic
architecture tournament.  The selected Genesis II material and its limitations
are inherited verbatim.  This module contains only pre-outcome policy: no
measured candidate result is encoded here.
"""

from __future__ import annotations

from typing import Any

PROGRAM = "substrate-tangible-sandbox-r2"
CORPUS = "STSC-1"
CORPUS_VERSION = "1.0.0-r2"
ACTIVATION = False
UNQUALIFIED_NOUS = False

PARENT_MERGE_COMMIT = "31f5925751c28cc8677f691362d30db5fe73f8b0"
PARENT_READY_TAG = "substrate-cognitive-material-ii-ready"
PARENT_TERMINAL_TAG = "substrate-cognitive-material-ii-terminal"
PARENT_SELECTED_MATERIAL = "L1_associative_monolith"
PARENT_CLASSIFICATION = "cognitive_material_genesis_ii_complete"
PARENT_STATUS = "compositional_advantage_unproven"
PARENT_READINESS = "tangible_sandbox_ready"

IMPLEMENTATION_BRANCH = "agent/substrate-tangible-sandbox"
TERMINAL_BRANCH = "agent/substrate-tangible-sandbox-terminal"
PREFLIGHT_TAG = "substrate-tangible-sandbox-preflight"
READY_TAG = "substrate-tangible-sandbox-ready"
TERMINAL_TAG = "substrate-tangible-sandbox-terminal"
CANDIDATE_TAG = "substrate-tangible-developmental-candidate-1"

GIB = 1024**3
MINIMUM_DISK_FLOOR_BYTES = 50 * GIB
DISK_FLOOR_FRACTION = 0.20
CORE_MINIMUM_ACQUISITION_BYTES = 60 * GIB
CORE_PREFERRED_ACQUISITION_BYTES = 100 * GIB
SEED_MAXIMUM_ACQUISITION_BYTES = 25 * GIB

SESOI = 0.05
CONFIDENCE = 0.95
POWER_TARGET = 0.90
PLANNING_ETA_HOURS = 36
EXPECTED_ETA_HOURS = (32, 44)
CONTINGENCY_ETA_HOURS = (48, 60)
LONGITUDINAL_HOURS = 24

REQUIRED_ARMS = (
    "L1_full",
    "L1_no_development",
    "fresh_model",
    "full_transcript_replay",
    "summary_replay",
    "strong_retrieval",
    "conventional_memory_agent",
    "project_state_database",
    "stateless_router",
    "direct_strongest_model",
    "best_of_n_direct_model",
    "S2",
    "oracle",
)

REQUIRED_PUBLIC_FLOOR = {
    "web": "WebArena-Verified Hard or preregistered balanced subset >=96",
    "software_engineering": "SWE-bench Verified target 50, minimum 25",
    "long_term_memory": "LongMemEval-V2 small or LongMemEval",
    "tool_agent_user": "tau2-bench text across >=3 domains",
    "gui_or_embodied": "AndroidWorld or another admitted GUI/embodied lane",
}

OUTCOMES = {
    "A": {
        "classification": "tangible_developmental_cognitive_material_candidate",
        "readiness": "publication_ready",
    },
    "B": {
        "classification": "tangible_sandbox_complete",
        "status": "practical_advantage_unproven",
        "readiness": "publication_ready",
    },
    "C": {
        "classification": "terminal_tangible_sandbox_null",
    },
}

OUTCOME_A_GATES = (
    "effect_at_least_sesoi",
    "confidence_lower_above_zero",
    "replication_positive",
    "hidden_composition_positive",
    "continuity_passing",
    "resource_parity",
    "zero_mutation_survivors",
    "counterfeits_rejected",
    "independent_verification",
    "clean_clone",
)

OUTCOME_C_RESERVED_FOR = (
    "critical environment, evaluator, dataset, platform dependency, or "
    "integrity requirement cannot be made valid after bounded repair"
)

ACQUISITION_STATES = (
    "DISCOVERED",
    "LICENSED",
    "RESERVED",
    "DOWNLOADING",
    "DOWNLOADED",
    "HASHED",
    "EXTRACTED",
    "PREPROCESSED",
    "VALIDATED",
    "COMPLETE",
    "QUARANTINED",
    "GATED",
    "REFUSED",
)

ACQUISITION_POOLS = {
    "network": {"initial_workers": 4, "role": "source-limited resumable downloads"},
    "hash": {"initial_workers": 4, "role": "SHA-256 and ETag validation"},
    "extraction": {"initial_workers": 2, "role": "archive-safe extraction"},
    "preprocessing": {"initial_workers": 8, "role": "shard-safe transforms"},
}

STSC_ROOTS = (
    "builder_visible",
    "executor_visible",
    "evaluator_only",
    "publication_safe",
)

STSC_SPLITS = (
    "construction",
    "canary",
    "pilot",
    "principal",
    "replication",
    "hidden_composition",
    "publication_demo",
)

STSC_FAMILIES = (
    "longitudinal_software_project",
    "document_workspace",
    "browser_and_knowledge_work",
    "desktop_control",
    "android_control",
    "tool_agent_user_interaction",
    "long_term_memory",
    "multimodal_incident_reconstruction",
    "video_and_motion",
    "audio_and_speech",
    "spatial_and_3d_body",
    "active_perception",
    "human_style_teaching",
    "model_and_tool_replacement",
    "negative_transfer",
    "compound_publication_project",
)

CANARIES = (
    "C01_source_license_recorded",
    "C02_checksum_mismatch_detected",
    "C03_partial_download_resumes",
    "C04_duplicate_download_avoided",
    "C05_evaluator_only_data_inaccessible",
    "C06_public_benchmark_gold_task_passes",
    "C07_known_bad_action_fails",
    "C08_L1_persistent_state_changes_return_task",
    "C09_transcript_replay_distinct",
    "C10_no_development_distinct",
    "C11_wrong_history_clean",
    "C12_model_replacement_preserves_goal_and_state",
    "C13_document_evaluator_checks_contents",
    "C14_code_evaluator_runs_hidden_tests",
    "C15_browser_evaluator_checks_environment_state",
    "C16_video_consumes_frames",
    "C17_audio_consumes_waveforms",
    "C18_3d_consumes_scene_and_body_state",
    "C19_active_perception_has_cost",
    "C20_active_perception_oracle_has_headroom",
    "C21_teaching_precedes_test_outcome",
    "C22_false_teaching_rejected_or_scoped",
    "C23_checkpoint_restores_exact_owned_state",
    "C24_model_contexts_clear_before_restore",
    "C25_baseline_equal_tools_and_model_budget",
    "C26_task_id_leakage_detected",
    "C27_answer_leakage_detected",
    "C28_activation_remains_false",
)

MUTATIONS = (
    "answer_in_filename",
    "answer_in_task_metadata",
    "evaluator_file_exposed",
    "train_seed_reused_in_hidden",
    "public_result_used_for_selection",
    "transcript_credited_as_development",
    "summary_receives_hidden_state",
    "wrong_history_receives_credit",
    "fresh_receives_L1_memory",
    "baseline_fewer_tools",
    "baseline_lower_model_budget",
    "active_perception_free_correct_view",
    "teaching_contains_final_answer",
    "future_correction_used_early",
    "video_reads_scene_labels",
    "audio_reads_event_labels",
    "3d_reads_hidden_coordinates",
    "code_skips_hidden_tests",
    "document_checks_name_only",
    "checkpoint_omits_goals",
    "checkpoint_omits_environment",
    "model_replacement_resets_identity",
    "external_activation_true",
    "collector_accepts_incomplete_run",
)

COUNTERFEITS = (
    "file_name_oracle",
    "task_id_lookup",
    "history_key_lookup",
    "transcript_memorizer",
    "summary_oracle",
    "label_reading_media_adapter",
    "precompiled_custom_task_state_machine",
    "overpowered_arm",
)

REQUIRED_DELIVERABLES = (
    "SUBSTRATE_SANDBOX_PREFLIGHT.json",
    "SUBSTRATE_SANDBOX_HISTORICAL_IMMUTABILITY.json",
    "SUBSTRATE_SANDBOX_RESEARCH_AUTHORITY.json",
    "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json",
    "SUBSTRATE_SANDBOX_LICENSE_LEDGER.json",
    "SUBSTRATE_SANDBOX_ACQUISITION_PLAN.json",
    "SUBSTRATE_SANDBOX_ACQUISITION_RESULT.json",
    "SUBSTRATE_SANDBOX_DATA_MANIFEST.json",
    "SUBSTRATE_SANDBOX_DISK_PLAN.json",
    "SUBSTRATE_SANDBOX_PARALLELISM_POLICY.json",
    "SUBSTRATE_SANDBOX_MODEL_PANEL.json",
    "SUBSTRATE_SANDBOX_ENVIRONMENT_CATALOG.json",
    "SUBSTRATE_SANDBOX_ADAPTER_CONTRACT.json",
    "SUBSTRATE_SANDBOX_STSC1_SCHEMA.json",
    "SUBSTRATE_SANDBOX_STSC1_GENERATOR_AUTHORITY.json",
    "SUBSTRATE_SANDBOX_STSC1_SPLITS.json",
    "SUBSTRATE_SANDBOX_PUBLIC_BENCHMARK_PLAN.json",
    "SUBSTRATE_SANDBOX_BASELINE_AUTHORITY.json",
    "SUBSTRATE_SANDBOX_RESOURCE_PARITY.json",
    "SUBSTRATE_SANDBOX_GROK_AUTHORITY.json",
    "SUBSTRATE_SANDBOX_GROK_LEDGER.json",
    "SUBSTRATE_SANDBOX_CANARIES.json",
    "SUBSTRATE_SANDBOX_PILOT.json",
    "SUBSTRATE_SANDBOX_FAILURE_MATRIX.json",
    "SUBSTRATE_SANDBOX_FREEZE.json",
    "SUBSTRATE_SANDBOX_STATISTICAL_AUTHORITY.json",
    "SUBSTRATE_SANDBOX_PRINCIPAL_AUTHORITY.json",
    "SUBSTRATE_SANDBOX_PRINCIPAL_DAG.json",
    "SUBSTRATE_SANDBOX_PUBLIC_RESULTS.json",
    "SUBSTRATE_SANDBOX_CUSTOM_RESULTS.json",
    "SUBSTRATE_SANDBOX_REPLICATION.json",
    "SUBSTRATE_SANDBOX_HIDDEN_COMPOSITION.json",
    "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json",
    "SUBSTRATE_SANDBOX_TEACHING_RESULT.json",
    "SUBSTRATE_SANDBOX_MODEL_REPLACEMENT_RESULT.json",
    "SUBSTRATE_SANDBOX_MUTATION_REPORT.json",
    "SUBSTRATE_SANDBOX_COUNTERFEIT_REPORT.json",
    "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION.json",
    "SUBSTRATE_SANDBOX_CLEAN_CLONE.json",
    "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json",
    "SUBSTRATE_SANDBOX_FINAL_STATE.json",
    "SUBSTRATE_SANDBOX_TERMINAL_REPORT.md",
)

# Official-source state observed on 2026-07-29.  A source refresh resolves the
# live HEAD independently and stores both values.  Release pins are selected
# where an official stable release exists; otherwise a full commit is required.
OFFICIAL_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "browsergym",
        "official_url": "https://github.com/ServiceNow/BrowserGym",
        "git_url": "https://github.com/ServiceNow/BrowserGym.git",
        "observed_head": "9e779f087de9a65668b6974d11f9ce9816026e96",
        "selected_revision": "v0.14.3",
        "license": "Apache-2.0",
        "priority": "P0",
        "access": "ungated",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "webarena_verified",
        "official_url": "https://github.com/ServiceNow/webarena-verified",
        "git_url": "https://github.com/ServiceNow/webarena-verified.git",
        "observed_head": "6473f72db5dcefc97b5725b59e734504edc28a21",
        "selected_revision": "v1.2.3",
        "license": "Apache-2.0",
        "priority": "P1",
        "access": "ungated_code_and_dataset; environment_images_required",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "android_world",
        "official_url": "https://github.com/google-research/android_world",
        "git_url": "https://github.com/google-research/android_world.git",
        "observed_head": "3e50888527ef9f29b9157ecd537e408008bb1c85",
        "selected_revision": "3e50888527ef9f29b9157ecd537e408008bb1c85",
        "license": "Apache-2.0",
        "priority": "P1",
        "access": "ungated",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "swe_bench",
        "official_url": "https://github.com/SWE-bench/SWE-bench",
        "git_url": "https://github.com/SWE-bench/SWE-bench.git",
        "observed_head": "f7bbbb2ccdf479001d6467c9e34af59e44a840f9",
        "selected_revision": "f7bbbb2ccdf479001d6467c9e34af59e44a840f9",
        "license": "MIT code; source repositories retain their licenses",
        "priority": "P1",
        "access": "ungated_metadata; Docker images and repositories required",
        "redistribution_class": "local_evaluation_only",
    },
    {
        "source_id": "tau2_bench",
        "official_url": "https://github.com/sierra-research/tau2-bench",
        "git_url": "https://github.com/sierra-research/tau2-bench.git",
        "observed_head": "363133ada1936491fb5bcec33cd62c3518a99f65",
        "selected_revision": "v1.0.1",
        "license": "MIT",
        "priority": "P0",
        "access": "ungated_text_mode; model endpoints required",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "longmemeval_v2",
        "official_url": "https://github.com/xiaowu0162/LongMemEval-V2",
        "git_url": "https://github.com/xiaowu0162/LongMemEval-V2.git",
        "observed_head": "6f020ac2fc3275e46c706d3406e02c3ed79b7be2",
        "selected_revision": "6f020ac2fc3275e46c706d3406e02c3ed79b7be2",
        "license": "Apache-2.0 code; verify dataset card separately",
        "priority": "P0",
        "access": "ungated_public_small_tier; model endpoints required",
        "redistribution_class": "local_evaluation_only",
    },
    {
        "source_id": "longmemeval",
        "official_url": "https://github.com/xiaowu0162/LongMemEval",
        "git_url": "https://github.com/xiaowu0162/LongMemEval.git",
        "observed_head": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "selected_revision": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "license": "MIT code; verify cleaned dataset card separately",
        "priority": "P0",
        "access": "ungated",
        "redistribution_class": "local_evaluation_only",
    },
    {
        "source_id": "kubric",
        "official_url": "https://github.com/google-research/kubric",
        "git_url": "https://github.com/google-research/kubric.git",
        "observed_head": "61f2422c84bab75006df33c6989e0b483db3ccfe",
        "selected_revision": "61f2422c84bab75006df33c6989e0b483db3ccfe",
        "license": "Apache-2.0 code; asset licenses separate",
        "priority": "P0",
        "access": "ungated",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "procthor",
        "official_url": "https://github.com/allenai/procthor",
        "git_url": "https://github.com/allenai/procthor.git",
        "observed_head": "53d5bd4c8c96a699e6a615dc390abb670cc9d353",
        "selected_revision": "53d5bd4c8c96a699e6a615dc390abb670cc9d353",
        "license": "Apache-2.0",
        "priority": "P1",
        "access": "ungated",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "ai2thor",
        "official_url": "https://github.com/allenai/ai2thor",
        "git_url": "https://github.com/allenai/ai2thor.git",
        "observed_head": "24f79883b4889e3f0e6f4ae301808b9025872dfc",
        "selected_revision": "24f79883b4889e3f0e6f4ae301808b9025872dfc",
        "license": "Apache-2.0",
        "priority": "P1",
        "access": "ungated; Unity runtime assets required",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "osworld_v2",
        "official_url": "https://github.com/xlang-ai/OSWorld-V2",
        "git_url": "https://github.com/xlang-ai/OSWorld-V2.git",
        "observed_head": "1f413c3b5e2f12942e8d633630a48542f789b11e",
        "selected_revision": "v2026.06.24",
        "license": "Apache-2.0 code; task classes and complete assets gated",
        "priority": "P2",
        "access": "gated_task_classes_and_assets",
        "redistribution_class": "gated_optional",
    },
    {
        "source_id": "workarena",
        "official_url": "https://github.com/ServiceNow/WorkArena",
        "git_url": "https://github.com/ServiceNow/WorkArena.git",
        "observed_head": "a772230a94cf1caf4166b8ead3983f3b3786455b",
        "selected_revision": "a772230a94cf1caf4166b8ead3983f3b3786455b",
        "license": "code license in repository; ServiceNow instance terms separate",
        "priority": "P2",
        "access": "gated_ServiceNow_instances",
        "redistribution_class": "gated_optional",
    },
    {
        "source_id": "teach",
        "official_url": "https://github.com/alexa/teach",
        "git_url": "https://github.com/alexa/teach.git",
        "observed_head": "903191e256da866a603d1bbfb21db34e0874392d",
        "selected_revision": "903191e256da866a603d1bbfb21db34e0874392d",
        "license": "MIT code/weights; Apache-2.0 images; CDLA-Sharing-1.0 data",
        "priority": "P2",
        "access": "ungated",
        "redistribution_class": "redistributable_with_attribution",
    },
    {
        "source_id": "mle_bench",
        "official_url": "https://github.com/openai/mle-bench",
        "git_url": "https://github.com/openai/mle-bench.git",
        "observed_head": "507f92e1138bb6e40dac5c6ee7a6758e6424bf97",
        "selected_revision": "507f92e1138bb6e40dac5c6ee7a6758e6424bf97",
        "license": "MIT code; Kaggle competition terms per dataset",
        "priority": "P3",
        "access": "Kaggle_credentials_and_terms_required",
        "redistribution_class": "gated_optional",
    },
    {
        "source_id": "fsd50k",
        "official_url": "https://zenodo.org/records/4060432",
        "observed_head": None,
        "selected_revision": "Zenodo record 4060432 version 1.0",
        "license": "CC-BY corpus; per-clip CC0/CC-BY/CC-BY-NC/Sampling+",
        "priority": "P1",
        "access": "ungated",
        "redistribution_class": "filter_to_CC0_and_CC-BY_with_attribution",
    },
    {
        "source_id": "common_voice",
        "official_url": "https://datacollective.mozillafoundation.org/datasets",
        "observed_head": None,
        "selected_revision": "locale-specific release selected only after capacity admission",
        "license": "dataset-card-specific; current Common Voice releases commonly CC0",
        "priority": "P1",
        "access": "terms_apply; no_speaker_reidentification; no_rehosting",
        "redistribution_class": "local_evaluation_only",
    },
    {
        "source_id": "librispeech",
        "official_url": "https://www.openslr.org/12",
        "observed_head": None,
        "selected_revision": "SLR12",
        "license": "CC-BY-4.0",
        "priority": "P1",
        "access": "ungated; at most four concurrent connections",
        "redistribution_class": "redistributable_with_attribution",
    },
)

# Core binary acquisition is deliberately limited to public, directly
# downloadable archives with an official checksum. FSD50K is retained locally
# until clip-level licenses are filtered; LibriSpeech is CC-BY-4.0. The
# selection is 80.13 GiB and therefore falls inside the frozen 60-100 GiB
# preferred Core envelope before Docker layers and generated outputs.
CORE_BINARY_ASSETS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.eval_audio.z01",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.z01/content",
        "bytes": 3221225472,
        "md5": "3090670eaeecc013ca1ff84fe4442aeb",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.dev_audio.z05",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z05/content",
        "bytes": 3221225472,
        "md5": "81356521aa159accd3c35de22da28c7f",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.ground_truth.zip",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.ground_truth.zip/content",
        "bytes": 334701,
        "md5": "ca27382c195e37d2269c4c866dd73485",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.eval_audio.zip",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.zip/content",
        "bytes": 3037675767,
        "md5": "6fa47636c3a3ad5c7dfeba99f2637982",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.dev_audio.zip",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.zip/content",
        "bytes": 2306663327,
        "md5": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.dev_audio.z04",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z04/content",
        "bytes": 3221225472,
        "md5": "d088ac4e11ba53daf9f7574c11cccac9",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.metadata.zip",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.metadata.zip/content",
        "bytes": 6700838,
        "md5": "b9ea0c829a411c1d42adb9da539ed237",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.dev_audio.z02",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z02/content",
        "bytes": 3221225472,
        "md5": "8f9b66153e68571164fb1315d00bc7bc",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.doc.zip",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.doc.zip/content",
        "bytes": 6984,
        "md5": "3516162b82dc2945d3e7feba0904e800",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.dev_audio.z01",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z01/content",
        "bytes": 3221225472,
        "md5": "faa7cf4cc076fc34a44a479a5ed862a3",
    },
    {
        "source_id": "fsd50k",
        "filename": "FSD50K.dev_audio.z03",
        "url": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z03/content",
        "bytes": 3221225472,
        "md5": "1196ef47d267a993d30fa98af54b7159",
    },
    {
        "source_id": "librispeech",
        "filename": "dev-clean.tar.gz",
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "bytes": 337926286,
        "md5": "42e2234ba48799c1f50f24a7926300a1",
    },
    {
        "source_id": "librispeech",
        "filename": "dev-other.tar.gz",
        "url": "https://www.openslr.org/resources/12/dev-other.tar.gz",
        "bytes": 314305928,
        "md5": "c8d0bcc9cca99d4f8b62fcc847357931",
    },
    {
        "source_id": "librispeech",
        "filename": "test-clean.tar.gz",
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "bytes": 346663984,
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
    },
    {
        "source_id": "librispeech",
        "filename": "test-other.tar.gz",
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "bytes": 328757843,
        "md5": "fb5a50374b501bb3bac4815ee91d3135",
    },
    {
        "source_id": "librispeech",
        "filename": "train-clean-100.tar.gz",
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "bytes": 6387309499,
        "md5": "2a93770f6d5c6c964bc36631d331a522",
    },
    {
        "source_id": "librispeech",
        "filename": "train-clean-360.tar.gz",
        "url": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "bytes": 23049477885,
        "md5": "c0e676e450a7ff2f54aeade5171606fa",
    },
    {
        "source_id": "librispeech",
        "filename": "train-other-500.tar.gz",
        "url": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
        "bytes": 30593501606,
        "md5": "d1a0fd59409feb2c614ce4d30c387708",
    },
)


def disk_floor_bytes(capacity_bytes: int) -> int:
    """Return the R2 protected free-space floor for a filesystem."""

    return max(MINIMUM_DISK_FLOOR_BYTES, int(capacity_bytes * DISK_FLOOR_FRACTION))


def configuration() -> dict[str, Any]:
    """Machine-readable frozen policy used by authorities and verification."""

    return {
        "program": PROGRAM,
        "corpus": {"name": CORPUS, "version": CORPUS_VERSION},
        "activation": False,
        "unqualified_nous": False,
        "parent": {
            "merge_commit": PARENT_MERGE_COMMIT,
            "ready_tag": PARENT_READY_TAG,
            "terminal_tag": PARENT_TERMINAL_TAG,
            "selected_material": PARENT_SELECTED_MATERIAL,
            "classification": PARENT_CLASSIFICATION,
            "status": PARENT_STATUS,
            "readiness": PARENT_READINESS,
            "activation": False,
        },
        "branches": {
            "implementation": IMPLEMENTATION_BRANCH,
            "terminal": TERMINAL_BRANCH,
        },
        "tags": {
            "preflight": PREFLIGHT_TAG,
            "ready": READY_TAG,
            "terminal": TERMINAL_TAG,
            "candidate": CANDIDATE_TAG,
        },
        "disk": {
            "minimum_floor_bytes": MINIMUM_DISK_FLOOR_BYTES,
            "floor_fraction": DISK_FLOOR_FRACTION,
            "seed_maximum_bytes": SEED_MAXIMUM_ACQUISITION_BYTES,
            "core_minimum_bytes": CORE_MINIMUM_ACQUISITION_BYTES,
            "core_preferred_bytes": CORE_PREFERRED_ACQUISITION_BYTES,
        },
        "statistics": {
            "sesoi": SESOI,
            "confidence": CONFIDENCE,
            "power_target": POWER_TARGET,
        },
        "eta": {
            "planning_hours": PLANNING_ETA_HOURS,
            "expected_hours": list(EXPECTED_ETA_HOURS),
            "contingency_hours": list(CONTINGENCY_ETA_HOURS),
            "longitudinal_hours": LONGITUDINAL_HOURS,
        },
        "required_arms": list(REQUIRED_ARMS),
        "required_public_floor": dict(REQUIRED_PUBLIC_FLOOR),
        "outcomes": OUTCOMES,
        "outcome_a_gates": list(OUTCOME_A_GATES),
        "outcome_c_reserved_for": OUTCOME_C_RESERVED_FOR,
        "acquisition_states": list(ACQUISITION_STATES),
        "acquisition_pools": ACQUISITION_POOLS,
        "stsc_roots": list(STSC_ROOTS),
        "stsc_splits": list(STSC_SPLITS),
        "stsc_families": list(STSC_FAMILIES),
        "canaries": list(CANARIES),
        "mutations": list(MUTATIONS),
        "counterfeits": list(COUNTERFEITS),
        "required_deliverables": list(REQUIRED_DELIVERABLES),
        "official_sources": list(OFFICIAL_SOURCES),
    }
