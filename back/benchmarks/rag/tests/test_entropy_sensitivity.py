import math

import pytest

from core.entropy_sensitivity import (
    compute_topk_distribution_metrics,
    summarize_emitted_logprobs,
)


def test_equal_logits_have_known_entropy_and_full_vocab_probability():
    result = compute_topk_distribution_metrics([0.0, 0.0, 0.0, 0.0], top_ks=(1, 2, 4))
    assert result["entropy_k4_bits"] == pytest.approx(2.0)
    assert result["retained_mass_k4"] == pytest.approx(1.0)
    assert result["retained_mass_k2"] == pytest.approx(0.5)
    assert result["top1_prob_full"] == pytest.approx(0.25)
    assert result["top1_prob_local_k2"] == pytest.approx(0.5)
    assert result["logit_margin_top1_top2"] == pytest.approx(0.0)


def test_topk_entropy_and_retained_mass_are_bounded_and_monotone():
    result = compute_topk_distribution_metrics([4, 3, 2, 1, 0], top_ks=(1, 2, 4))
    assert result["entropy_k1_bits"] == pytest.approx(0.0)
    assert result["retained_mass_k1"] < result["retained_mass_k2"] < result["retained_mass_k4"]
    assert result["retained_mass_k4"] <= 1.0
    assert 0.0 <= result["entropy_k2_bits"] <= 1.0
    assert result["entropy_k4_bits"] <= math.log2(4)


def test_extreme_logits_use_stable_softmax_and_validate_input():
    result = compute_topk_distribution_metrics([10000.0, 9999.0, -10000.0], top_ks=(1, 2, 10))
    assert all(math.isfinite(value) for value in result.values())
    assert result["retained_mass_k10"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_topk_distribution_metrics([1.0, float("nan")])
    with pytest.raises(ValueError):
        compute_topk_distribution_metrics([1.0], top_ks=(0,))


def test_emitted_continuation_likelihood_is_length_normalized():
    result = summarize_emitted_logprobs([-1.0, -3.0, -2.0])
    assert result["n_tokens"] == 3
    assert result["mean_log_likelihood_nats_per_token"] == pytest.approx(-2.0)
    assert result["mean_nll_bits_per_token"] == pytest.approx(2.0 / math.log(2.0))


def test_mlx_token_telemetry_matches_full_vocabulary_reference():
    mx = pytest.importorskip("mlx.core")
    from src.llm_engine.mlx_impl import build_compact_tokens_info

    logits = [7.0 - (idx * 0.037) - ((idx % 9) * 0.013) for idx in range(128)]
    logit_array = mx.array(logits, dtype=mx.float32)
    logprobs = logit_array - mx.logsumexp(logit_array)
    chosen_token_id = 17

    actual = build_compact_tokens_info(
        [logprobs],
        [{"token_id": chosen_token_id, "token_text": "chosen"}],
        top_k_store=5,
        top_k_entropy=50,
    )[0]
    expected = compute_topk_distribution_metrics(logits)
    log_z = math.log(sum(math.exp(value - max(logits)) for value in logits)) + max(logits)

    field_map = {
        "entropy_k20_bits": "entropy_k20_bits",
        "entropy_k50_bits": "entropy_k50_bits",
        "entropy_k100_bits": "entropy_k100_bits",
        "retained_mass_k20": "retained_mass_k20",
        "retained_mass_k50": "retained_mass_k50",
        "retained_mass_k100": "retained_mass_k100",
        "top1_prob_full": "top1_prob_full",
        "top1_prob_local_k20": "top1_prob_local_k20",
        "top1_prob_local_k50": "top1_prob_local_k50",
        "top1_prob_local_k100": "top1_prob_local_k100",
        "logit_margin": "logit_margin_top1_top2",
    }
    for actual_key, expected_key in field_map.items():
        assert actual[actual_key] == pytest.approx(
            expected[expected_key], rel=2e-5, abs=2e-6
        ), actual_key
    assert actual["logprob"] == pytest.approx(logits[chosen_token_id] - log_z, abs=2e-6)
    assert actual["nll_bits_per_token"] == pytest.approx(-actual["logprob"] / math.log(2.0))
