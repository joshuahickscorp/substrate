from mop.studio.encoder_scale_probe import SCHEMA, validate_probe_receipt


def _receipt() -> dict:
    return {
        "schema": SCHEMA,
        "config": {
            "sha256": "a" * 64,
            "revision": "b" * 40,
            "model_id": "fixture/vjepa",
        },
        "probe": {"mode": "forward", "device": "cpu"},
        "status": "passed",
        "explicit_out_of_memory": False,
        "hardware_limit_reached": False,
        "child": {
            "backend": "vjepa_hf",
            "parameters": 10,
            "trainable_parameters": 0,
            "model_class": "transformers.VJEPA2Model",
            "input_shape": [1, 64, 3, 256, 256],
            "input_layout": "B,T,C,H,W",
            "output_finite": True,
            "output_shape": [1, 1024],
        },
    }


def test_valid_real_forward_receipt():
    assert validate_probe_receipt(_receipt()) == []


def test_frozen_random_cannot_pass_as_real_forward():
    receipt = _receipt()
    receipt["child"]["backend"] = "frozen_random"
    assert "real vjepa_hf backend" in " ".join(validate_probe_receipt(receipt))


def test_timeout_or_kill_cannot_be_laundered_into_hardware_limit():
    receipt = _receipt()
    receipt["status"] = "timed-out"
    receipt["hardware_limit_reached"] = True
    assert "explicit allocator failure" in " ".join(validate_probe_receipt(receipt))


def test_passing_probe_requires_frozen_nonempty_realized_model():
    receipt = _receipt()
    receipt["child"]["trainable_parameters"] = 1
    receipt["child"]["parameters"] = 0
    problems = " ".join(validate_probe_receipt(receipt))
    assert "positive model parameter count" in problems
    assert "fully frozen model" in problems


def test_passing_forward_records_unambiguous_tensor_layout():
    receipt = _receipt()
    receipt["child"]["input_layout"] = "B,C,T,H,W"
    assert "B,T,C,H,W input contract" in " ".join(validate_probe_receipt(receipt))
