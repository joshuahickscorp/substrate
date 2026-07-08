from mop.studio.memory_envelope import SCHEMA, MemorySampler, summarize_samples


def test_memory_sampler_records_summary_shape():
    sampler = MemorySampler("unit")
    sampler.sample("start")
    sampler.sample("end")
    summary = sampler.summary()
    assert summary["schema"] == SCHEMA
    assert summary["label"] == "unit"
    assert summary["n_samples"] == 2
    assert len(summary["samples"]) == 2
    assert "process_rss_gb" in summary


def test_summarize_samples_peak_and_min_available():
    samples = [
        {"process_rss_gb": 1.0, "system_available_gb": 5.0},
        {"process_rss_gb": 2.5, "system_available_gb": 3.0},
        {"process_rss_gb": 1.5, "system_available_gb": 4.0},
    ]
    summary = summarize_samples("manual", samples)
    assert summary["process_rss_gb"] == {"start": 1.0, "end": 1.5, "peak": 2.5}
    assert summary["system_available_gb"] == {"start": 5.0, "end": 4.0, "peak": 3.0}
