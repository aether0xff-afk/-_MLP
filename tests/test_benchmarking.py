"""Speed/accuracy benchmark plumbing tests without asserting machine speed."""

from __future__ import annotations

import json

import numpy as np

from cubic_mlp.benchmarking import generate_hard_dataset, run_benchmark
from cubic_mlp.checkpoint import save_checkpoint
from cubic_mlp.model import CubicRootMLP, ModelConfig


def test_hard_dataset_is_finite_and_contains_expected_regimes() -> None:
    dataset = generate_hard_dataset(100, seed=77)

    assert dataset.coefficients.shape == (100, 4)
    assert dataset.roots.shape == (100, 3)
    assert np.all(np.isfinite(dataset.coefficients))
    assert np.all(np.isfinite(dataset.roots))
    assert np.sum(dataset.regimes == "near_multiple") == 50
    assert np.sum(dataset.regimes == "exact_multiple") == 20


def test_tiny_benchmark_writes_all_method_results(tmp_path) -> None:
    config = ModelConfig(hidden_sizes=(8, 6))
    model = CubicRootMLP(config)
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "benchmark.json"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "input_mean": [0.0, 0.0, 0.0],
            "input_std": [1.0, 1.0, 1.0],
        },
    )

    result = run_benchmark(
        checkpoint_path=checkpoint,
        output_path=output,
        samples=12,
        hard_samples=10,
        repeats=1,
        single_repeats=1,
        batch_size=8,
        device="cpu",
        seed=91,
    )

    methods = {
        "cardano_vectorized",
        "numpy_roots_loop",
        "mlp_batch",
        "mlp_newton4",
        "hybrid_mlp_cardano",
    }
    assert output.is_file()
    assert set(result["batch_speed"]) == methods
    assert set(result["general_accuracy"]) == methods
    assert set(result["hard_accuracy"]) == methods
    crossover = result["batch_crossover"]
    assert crossover["rows"]
    assert crossover["rows"][-1]["batch_size"] == 12
    assert set(crossover["first_measured_win"]) == {
        "mlp_vs_scalar_cardano",
        "mlp_vs_numpy_roots",
        "mlp_newton4_vs_scalar_cardano",
        "mlp_newton4_vs_numpy_roots",
        "mlp_vs_vectorized_cardano",
    }
    assert "within_tolerance_rate" in result["hard_accuracy"]["mlp_newton4"]
    assert all(
        record["median_seconds"] > 0.0
        for record in result["batch_speed"].values()
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["benchmark_scope"]["general_samples"] == 12
