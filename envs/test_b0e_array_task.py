import json
import os
import shutil
import tempfile

import pytest

from b0e_array_task import HORIZON_BY_TIER, run


@pytest.fixture
def tmp_output_dir():
    d = tempfile.mkdtemp(prefix="b0e_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_smoke_easy_mlp_seed0_end_to_end(tmp_output_dir):
    """Real pool, real checkpoint, real _evaluate call -- shrunk to 3
    instances x 2 resets so it finishes in seconds, not the real 100x10.
    Exercises the exact code path a cluster task will run: pool fingerprint
    verification, fixed-subsample draw, untrained-baseline construction,
    checkpoint loading, both eval modes, the self-check, and the output
    schema -- against a real (tier, encoder, seed) that already has a
    checkpoint on disk from the Stage A sweep."""
    run("easy", "mlp", 0, array_index=10, output_dir=tmp_output_dir,
        n_eval_instances=3, n_resets=2, self_check_n_instances=2, self_check_n_resets=1)

    out_path = os.path.join(tmp_output_dir, "easy_mlp_seed0_b0e.json")
    assert os.path.exists(out_path)
    with open(out_path) as f:
        out = json.load(f)

    assert out["tier"] == "easy"
    assert out["encoder"] == "mlp"
    assert out["seed"] == 0
    assert out["horizon_used"] == HORIZON_BY_TIER["easy"]
    assert out["n_eval_instances"] == 3
    assert out["n_resets"] == 2
    assert out["eval_subsample_seed"] == 999

    for section in ("baseline", "checkpoint"):
        assert 0.0 <= out[section]["greedy_rate"] <= 1.0
        assert 0.0 <= out[section]["stochastic_rate"] <= 1.0
        sc = out[section]["self_check"]
        assert 0.0 <= sc["capped_rate"] <= 1.0
        assert 0.0 <= sc["uncapped_rate"] <= 1.0
        assert isinstance(sc["matched"], bool)

    assert out["checkpoint"]["reported_full_protocol_greedy"] is not None
    assert out["pool_provenance"]["tier"] == "easy"


def test_idempotent_skips_existing_output(tmp_output_dir, capsys):
    run("easy", "mlp", 0, array_index=10, output_dir=tmp_output_dir,
        n_eval_instances=3, n_resets=2, self_check_n_instances=2, self_check_n_resets=1)
    capsys.readouterr()
    run("easy", "mlp", 0, array_index=10, output_dir=tmp_output_dir,
        n_eval_instances=3, n_resets=2, self_check_n_instances=2, self_check_n_resets=1)
    captured = capsys.readouterr()
    assert "already complete" in captured.out


def test_hard_tier_self_check_never_reports_a_cap_it_didnt_apply(tmp_output_dir):
    """HARD's horizon is uncapped (40) by policy -- self_check should mark
    skipped_redundant_uncapped_run rather than silently running the same
    eval twice and reporting a trivially-true match."""
    run("hard", "mlp", 0, array_index=15, output_dir=tmp_output_dir,
        n_eval_instances=3, n_resets=2, self_check_n_instances=2, self_check_n_resets=1)
    with open(os.path.join(tmp_output_dir, "hard_mlp_seed0_b0e.json")) as f:
        out = json.load(f)
    assert out["horizon_used"] == 40
    assert out["baseline"]["self_check"]["skipped_redundant_uncapped_run"] is True
    assert out["checkpoint"]["self_check"]["skipped_redundant_uncapped_run"] is True


def test_easy_medium_self_check_actually_compares_two_horizons(tmp_output_dir):
    run("medium", "mlp", 0, array_index=15, output_dir=tmp_output_dir,
        n_eval_instances=3, n_resets=2, self_check_n_instances=2, self_check_n_resets=1)
    with open(os.path.join(tmp_output_dir, "medium_mlp_seed0_b0e.json")) as f:
        out = json.load(f)
    assert out["horizon_used"] == 15
    assert "skipped_redundant_uncapped_run" not in out["baseline"]["self_check"]
