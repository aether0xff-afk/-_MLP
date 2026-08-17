"""순열 불변 지도 손실과 다항식 구조 보조 손실."""

from __future__ import annotations

from itertools import permutations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

ROOT_PERMUTATIONS = tuple(permutations(range(3)))


def permutation_invariant_smooth_l1(
    predictions: Tensor,
    targets: Tensor,
    *,
    beta: float = 0.05,
) -> Tensor:
    """세 근의 6가지 일대일 대응 중 오차가 가장 작은 것을 선택한다."""

    if predictions.shape != targets.shape or predictions.shape[-2:] != (3, 2):
        raise ValueError("predictions와 targets의 모양은 모두 (N, 3, 2)여야 합니다.")
    indices = torch.tensor(ROOT_PERMUTATIONS, device=targets.device)
    permuted_targets = targets[:, indices, :]
    expanded_predictions = predictions[:, None, :, :].expand_as(permuted_targets)
    element_losses = functional.smooth_l1_loss(
        expanded_predictions,
        permuted_targets,
        reduction="none",
        beta=beta,
    )
    assignment_losses = element_losses.mean(dim=(-1, -2))
    return assignment_losses.min(dim=1).values.mean()


def _complex_multiply(
    left_real: Tensor,
    left_imag: Tensor,
    right_real: Tensor,
    right_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def normalized_polynomial_residual_loss(
    predictions: Tensor,
    normalized_coefficients: Tensor,
) -> Tensor:
    """각 예측근을 식에 대입한 scale-invariant 잔차의 제곱 평균."""

    x = predictions[..., 0]
    y = predictions[..., 1]
    b = normalized_coefficients[:, 0, None]
    c = normalized_coefficients[:, 1, None]
    d = normalized_coefficients[:, 2, None]

    z2_real, z2_imag = _complex_multiply(x, y, x, y)
    z3_real, z3_imag = _complex_multiply(z2_real, z2_imag, x, y)
    value_real = z3_real + b * z2_real + c * x + d
    value_imag = z3_imag + b * z2_imag + c * y
    magnitude_squared = value_real.square() + value_imag.square()
    root_abs = torch.sqrt(x.square() + y.square() + 1e-16)
    denominator = (
        root_abs.pow(3)
        + b.abs() * root_abs.square()
        + c.abs() * root_abs
        + d.abs()
        + 1e-6
    )
    return (magnitude_squared / denominator.square()).mean()


def vieta_loss(predictions: Tensor, normalized_coefficients: Tensor) -> Tensor:
    """예측한 근 집합이 비에타 관계를 만족하도록 하는 보조 손실."""

    real = predictions[..., 0]
    imag = predictions[..., 1]
    b = normalized_coefficients[:, 0]
    c = normalized_coefficients[:, 1]
    d = normalized_coefficients[:, 2]

    sum1_real = real.sum(dim=1)
    sum1_imag = imag.sum(dim=1)

    p01 = _complex_multiply(real[:, 0], imag[:, 0], real[:, 1], imag[:, 1])
    p02 = _complex_multiply(real[:, 0], imag[:, 0], real[:, 2], imag[:, 2])
    p12 = _complex_multiply(real[:, 1], imag[:, 1], real[:, 2], imag[:, 2])
    sum2_real = p01[0] + p02[0] + p12[0]
    sum2_imag = p01[1] + p02[1] + p12[1]

    product01 = p01
    product_real, product_imag = _complex_multiply(
        product01[0], product01[1], real[:, 2], imag[:, 2]
    )

    first = ((sum1_real + b) / (1.0 + b.abs())).square() + sum1_imag.square()
    second = ((sum2_real - c) / (1.0 + c.abs())).square() + sum2_imag.square()
    third = ((product_real + d) / (1.0 + d.abs())).square() + product_imag.square()
    return (first + second + third).mean() / 3.0


class CubicRootLoss(nn.Module):
    def __init__(
        self,
        *,
        smooth_l1_beta: float = 0.05,
        residual_weight: float = 0.05,
        vieta_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.smooth_l1_beta = smooth_l1_beta
        self.residual_weight = residual_weight
        self.vieta_weight = vieta_weight

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        normalized_coefficients: Tensor,
        *,
        auxiliary_factor: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        matching = permutation_invariant_smooth_l1(
            predictions, targets, beta=self.smooth_l1_beta
        )
        residual = normalized_polynomial_residual_loss(
            predictions, normalized_coefficients
        )
        vieta = vieta_loss(predictions, normalized_coefficients)
        total = matching + auxiliary_factor * (
            self.residual_weight * residual + self.vieta_weight * vieta
        )
        return total, {
            "matching": matching.detach(),
            "residual": residual.detach(),
            "vieta": vieta.detach(),
        }
