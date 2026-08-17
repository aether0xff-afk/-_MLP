"""Fast tests for the trainable parts of the cubic-root approximator."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest
import torch

from cubic_mlp.checkpoint import load_checkpoint, save_checkpoint
from cubic_mlp.losses import (
    CubicRootLoss,
    normalized_polynomial_residual_loss,
    permutation_invariant_smooth_l1,
)
from cubic_mlp.model import CubicRootMLP, ModelConfig


def test_matching_loss_is_invariant_to_target_root_order() -> None:
    targets = torch.tensor(
        [
            [[-2.0, 0.0], [1.0, -2.0], [1.0, 2.0]],
            [[-1.0, 0.0], [0.5, 0.0], [3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    predictions = targets + torch.tensor(
        [[[0.10, 0.03], [-0.02, 0.04], [0.05, -0.01]]],
        dtype=torch.float32,
    )
    reference = permutation_invariant_smooth_l1(predictions, targets)

    for order in permutations(range(3)):
        permuted = targets[:, list(order), :]
        actual = permutation_invariant_smooth_l1(predictions, permuted)
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=1e-7)


def test_matching_loss_is_zero_for_an_exact_permutation() -> None:
    targets = torch.tensor(
        [[[-2.0, 0.0], [1.0, -2.0], [1.0, 2.0]]],
        dtype=torch.float32,
    )
    predictions = targets[:, [2, 0, 1], :]

    loss = permutation_invariant_smooth_l1(predictions, targets)

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0.0, atol=0.0)


def test_polynomial_residual_loss_is_zero_at_exact_zero_roots() -> None:
    predictions = torch.zeros((1, 3, 2), dtype=torch.float32)
    coefficients = torch.zeros((1, 3), dtype=torch.float32)

    loss = normalized_polynomial_residual_loss(predictions, coefficients)

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0.0, atol=0.0)


def test_model_output_shape_and_full_loss_gradients_are_finite() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(2026)
        model = CubicRootMLP(ModelConfig(hidden_sizes=(16, 12)))
        inputs = torch.randn(5, 3)
        targets = torch.randn(5, 3, 2)
        normalized_coefficients = torch.empty(5, 3).uniform_(-1.0, 1.0)

        predictions = model(inputs)
        total, components = CubicRootLoss()(
            predictions,
            targets,
            normalized_coefficients,
        )
        total.backward()

    assert predictions.shape == (5, 3, 2)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(total)
    assert set(components) == {"matching", "residual", "vieta"}
    assert all(torch.isfinite(value) for value in components.values())

    parameters = list(model.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_checkpoint_save_and_load_smoke(tmp_path) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(17)
        config = ModelConfig(hidden_sizes=(8, 6))
        model = CubicRootMLP(config)
        inputs = torch.tensor(
            [[0.25, -0.50, 0.75], [-0.10, 0.20, -0.30]],
            dtype=torch.float32,
        )
        expected = model(inputs).detach()

    checkpoint_path = tmp_path / "nested" / "smoke.pt"
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_config": config.to_dict(),
        "input_mean": [0.1, -0.2, 0.3],
        "input_std": [1.0, 2.0, 4.0],
    }
    save_checkpoint(checkpoint_path, payload)

    restored_model, restored_payload = load_checkpoint(checkpoint_path)
    actual = restored_model(inputs).detach()

    assert checkpoint_path.is_file()
    assert not restored_model.training
    assert restored_model.config == config
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        restored_payload["input_mean_array"],
        np.asarray(payload["input_mean"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        restored_payload["input_std_array"],
        np.asarray(payload["input_std"], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(input_dim=4),
        ModelConfig(output_dim=5),
        ModelConfig(activation="relu"),
    ],
)
def test_model_rejects_incompatible_public_configuration(config: ModelConfig) -> None:
    with pytest.raises(ValueError):
        CubicRootMLP(config)
