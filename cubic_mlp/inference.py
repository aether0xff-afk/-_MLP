"""학습된 MLP 추론, 구조 정리, 선택적 Newton 보정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .cardano import (
    canonicalize_roots,
    cubic_discriminant_terms,
    normalize_cubic_coefficients,
    solve_cubic_cardano_batch,
)
from .checkpoint import load_checkpoint

# 구조 후처리는 float32 MLP 입출력에 적용되므로, 판별식 경계의 허용오차도
# float32 반올림 오차에 맞춘다. 이는 절대 기준이 아니라 두 판별식 항의 합에
# 곱해지므로, 작은 계수의 명백한 복소근을 실근으로 바꾸지 않는다.
STRUCTURE_DISCRIMINANT_TOL = 64.0 * np.finfo(np.float32).eps


@dataclass(frozen=True)
class PredictionResult:
    raw_roots: NDArray[np.complex128]
    structured_roots: NDArray[np.complex128]
    normalized_coefficients: NDArray[np.float64]
    variable_scale: float
    input_z_scores: NDArray[np.float64]


@dataclass(frozen=True)
class BatchPredictionResult:
    raw_roots: NDArray[np.complex128]
    structured_roots: NDArray[np.complex128]
    normalized_coefficients: NDArray[np.float64]
    variable_scales: NDArray[np.float64]
    input_z_scores: NDArray[np.float32]


@dataclass(frozen=True)
class HybridPredictionResult:
    roots: NDArray[np.complex128]
    mlp_roots: NDArray[np.complex128]
    fallback_mask: NDArray[np.bool_]
    normalized_residual_score: NDArray[np.float64]
    vieta_error_score: NDArray[np.float64]
    relative_discriminant: NDArray[np.float64]

    @property
    def fallback_rate(self) -> float:
        return float(np.mean(self.fallback_mask))


def enforce_real_cubic_structure(
    roots: ArrayLike,
    normalized_coefficients: ArrayLike,
    *,
    discriminant_tolerance: float = STRUCTURE_DISCRIMINANT_TOL,
) -> NDArray[np.complex128]:
    """판별식이 알려 주는 실근 수와 켤레복소근 구조만 강제한다."""

    values = np.asarray(roots, dtype=np.complex128).copy()
    if values.shape != (3,):
        raise ValueError("roots는 세 개여야 합니다.")
    coefficients = np.asarray(normalized_coefficients, dtype=np.float64)
    p, q, delta = cubic_discriminant_terms(coefficients)
    delta_scale = abs(float((q / 2.0) ** 2)) + abs(float((p / 3.0) ** 3))
    threshold = discriminant_tolerance * delta_scale

    if float(delta) > threshold:
        real_index = int(np.argmin(np.abs(values.imag)))
        pair_indices = [index for index in range(3) if index != real_index]
        real_root = complex(values[real_index].real, 0.0)
        pair_real = float(np.mean(values[pair_indices].real))
        pair_imag = float(np.mean(np.abs(values[pair_indices].imag)))
        values = np.asarray(
            [real_root, pair_real - 1j * pair_imag, pair_real + 1j * pair_imag]
        )
    else:
        values = values.real.astype(np.complex128)
    return canonicalize_roots(values)


def enforce_real_cubic_structure_batch(
    roots: ArrayLike,
    normalized_coefficients: ArrayLike,
) -> NDArray[np.complex128]:
    """배치의 각 근 집합에 :func:`enforce_real_cubic_structure`를 적용한다."""

    values = np.asarray(roots, dtype=np.complex128)
    coefficients = np.asarray(normalized_coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or coefficients.shape != (len(values), 3):
        raise ValueError("roots는 (N, 3), normalized_coefficients는 (N, 3)이어야 합니다.")
    p, q, delta = cubic_discriminant_terms(coefficients)
    delta_scale = np.abs((q / 2.0) ** 2) + np.abs((p / 3.0) ** 3)
    complex_pair = delta > STRUCTURE_DISCRIMINANT_TOL * delta_scale

    structured = values.real.astype(np.complex128)
    if np.any(complex_pair):
        rows = np.flatnonzero(complex_pair)
        selected = values[rows]
        real_indices = np.argmin(np.abs(selected.imag), axis=1)
        real_roots = selected[np.arange(len(rows)), real_indices].real
        pair_real = (selected.real.sum(axis=1) - real_roots) / 2.0
        pair_imag = (
            np.abs(selected.imag).sum(axis=1)
            - np.abs(selected[np.arange(len(rows)), real_indices].imag)
        ) / 2.0
        structured[rows] = np.column_stack(
            [
                real_roots.astype(np.complex128),
                pair_real - 1j * pair_imag,
                pair_real + 1j * pair_imag,
            ]
        )
    return canonicalize_roots(structured)


def batch_vieta_set_errors(
    normalized_coefficients: ArrayLike,
    scaled_roots: ArrayLike,
) -> NDArray[np.float64]:
    """무차원 근 배치의 비에타 계수 복원 오차 ``(N, 3)``을 반환한다."""

    coefficients = np.asarray(normalized_coefficients, dtype=np.float64)
    roots = np.asarray(scaled_roots, dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[1] != 3 or roots.shape != (len(coefficients), 3):
        raise ValueError("normalized_coefficients와 scaled_roots는 각각 (N, 3)이어야 합니다.")
    r1, r2, r3 = roots.T
    reconstructed = np.column_stack(
        [-(r1 + r2 + r3), r1 * r2 + r1 * r3 + r2 * r3, -(r1 * r2 * r3)]
    )
    return np.abs(reconstructed - coefficients) / (1.0 + np.abs(coefficients))


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다.")
    return resolved


class CubicMLPSolver:
    """체크포인트를 한 번만 읽고 많은 삼차식을 배치로 처리한다."""

    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/cubic_mlp.pt",
        *,
        device: str | torch.device = "auto",
    ) -> None:
        self.device = _resolve_device(device)
        self.model, self.payload = load_checkpoint(checkpoint_path, device=self.device)
        self.input_mean = self.payload["input_mean_array"]
        self.input_std = self.payload["input_std_array"]

    @torch.inference_mode()
    def predict_batch(
        self,
        coefficients: ArrayLike,
        *,
        batch_size: int = 65_536,
    ) -> BatchPredictionResult:
        """``(N,4)`` 계수를 MLP로 한꺼번에 추론한다."""

        if batch_size <= 0:
            raise ValueError("batch_size는 양수여야 합니다.")
        matrix = np.asarray(coefficients, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != 4 or len(matrix) == 0:
            raise ValueError("coefficients는 비어 있지 않은 (N, 4) 배열이어야 합니다.")

        normalized, scales = normalize_cubic_coefficients(matrix)
        normalized_float64 = np.asarray(normalized, dtype=np.float64)
        scales_float64 = np.asarray(scales, dtype=np.float64)
        model_inputs = normalized_float64.astype(np.float32)
        standardized = np.ascontiguousarray(
            (model_inputs - self.input_mean) / self.input_std,
            dtype=np.float32,
        )

        prediction_pairs = np.empty((len(matrix), 3, 2), dtype=np.float32)
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            tensor = torch.from_numpy(standardized[start:end]).to(self.device)
            prediction_pairs[start:end] = self.model(tensor).cpu().numpy()

        raw_scaled = (
            prediction_pairs[..., 0].astype(np.float64)
            + 1j * prediction_pairs[..., 1].astype(np.float64)
        )
        structured_scaled = enforce_real_cubic_structure_batch(
            raw_scaled, normalized_float64
        )
        return BatchPredictionResult(
            raw_roots=canonicalize_roots(raw_scaled * scales_float64[:, None]),
            structured_roots=canonicalize_roots(
                structured_scaled * scales_float64[:, None]
            ),
            normalized_coefficients=normalized_float64,
            variable_scales=scales_float64,
            input_z_scores=standardized,
        )

    def predict_hybrid(
        self,
        coefficients: ArrayLike,
        *,
        batch_size: int = 65_536,
        residual_threshold: float = 0.05,
        vieta_threshold: float = 0.05,
        zscore_threshold: float = 6.0,
        discriminant_relative_threshold: float = 1e-6,
        newton_steps: int = 3,
    ) -> HybridPredictionResult:
        """MLP 신뢰도가 낮은 행에만 카르다노 fallback을 적용한다."""

        from .metrics import normalized_residuals

        matrix = np.asarray(coefficients, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        prediction = self.predict_batch(matrix, batch_size=batch_size)
        scaled_roots = (
            prediction.structured_roots / prediction.variable_scales[:, None]
        )
        if newton_steps:
            scaled_roots = newton_polish_roots_batch_scaled(
                scaled_roots,
                prediction.normalized_coefficients,
                steps=newton_steps,
            )
        mlp_roots = scaled_roots * prediction.variable_scales[:, None]
        residuals = normalized_residuals(
            prediction.normalized_coefficients, scaled_roots
        )
        residual_score = np.median(residuals, axis=1)
        vieta_score = np.max(
            batch_vieta_set_errors(
                prediction.normalized_coefficients, scaled_roots
            ),
            axis=1,
        )
        out_of_distribution = (
            np.max(np.abs(prediction.input_z_scores), axis=1) > zscore_threshold
        )
        p, q, delta = cubic_discriminant_terms(
            prediction.normalized_coefficients
        )
        discriminant_scale = np.abs((q / 2.0) ** 2) + np.abs((p / 3.0) ** 3)
        relative_discriminant = np.divide(
            np.abs(delta),
            discriminant_scale,
            out=np.zeros_like(delta),
            where=discriminant_scale > 0.0,
        )
        ill_conditioned = (
            relative_discriminant < discriminant_relative_threshold
        )
        fallback_mask = (
            (residual_score > residual_threshold)
            | (vieta_score > vieta_threshold)
            | out_of_distribution
            | ill_conditioned
        )
        roots = mlp_roots.copy()
        if np.any(fallback_mask):
            roots[fallback_mask] = solve_cubic_cardano_batch(matrix[fallback_mask])
        return HybridPredictionResult(
            roots=roots,
            mlp_roots=mlp_roots,
            fallback_mask=fallback_mask,
            normalized_residual_score=residual_score,
            vieta_error_score=vieta_score,
            relative_discriminant=relative_discriminant,
        )

    def predict_polished_batch(
        self,
        coefficients: ArrayLike,
        *,
        batch_size: int = 65_536,
        newton_steps: int = 4,
    ) -> NDArray[np.complex128]:
        """MLP 배치 출력을 초기값으로 벡터화 Newton 보정을 적용한다."""

        prediction = self.predict_batch(coefficients, batch_size=batch_size)
        scaled = prediction.structured_roots / prediction.variable_scales[:, None]
        polished = newton_polish_roots_batch_scaled(
            scaled,
            prediction.normalized_coefficients,
            steps=newton_steps,
        )
        return polished * prediction.variable_scales[:, None]


@torch.inference_mode()
def predict_roots(
    coefficients: ArrayLike,
    *,
    checkpoint_path: str | Path = "artifacts/cubic_mlp.pt",
    device: str = "cpu",
) -> PredictionResult:
    solver = CubicMLPSolver(checkpoint_path, device=device)
    batch = solver.predict_batch(coefficients, batch_size=1)
    return PredictionResult(
        raw_roots=batch.raw_roots[0],
        structured_roots=batch.structured_roots[0],
        normalized_coefficients=batch.normalized_coefficients[0],
        variable_scale=float(batch.variable_scales[0]),
        input_z_scores=batch.input_z_scores[0].astype(np.float64),
    )


def newton_polish_roots(
    roots: ArrayLike,
    coefficients: ArrayLike,
    *,
    steps: int = 2,
) -> NDArray[np.complex128]:
    """MLP 예측을 초기값으로 하여 제한된 복소 Newton 보정을 적용한다."""

    values = np.asarray(roots, dtype=np.complex128).copy()
    a, b, c, d = np.asarray(coefficients, dtype=np.float64)
    if steps < 0:
        raise ValueError("steps는 0 이상이어야 합니다.")
    coefficient_scale = max(abs(a), abs(b), abs(c), abs(d))
    if coefficient_scale == 0.0:
        raise ValueError("모든 계수가 0일 수 없습니다.")
    for _ in range(steps):
        polynomial = ((a * values + b) * values + c) * values + d
        derivative = (3.0 * a * values + 2.0 * b) * values + c
        safe = np.abs(derivative) > 1e-12 * coefficient_scale
        correction = np.zeros_like(values)
        correction[safe] = polynomial[safe] / derivative[safe]
        maximum_step = 0.5 * (1.0 + np.abs(values))
        too_large = np.abs(correction) > maximum_step
        correction[too_large] *= maximum_step[too_large] / np.abs(correction[too_large])
        # Newton은 도함수가 작은 곳에서 오히려 멀어질 수 있다. 각 근마다
        # 잔차가 실제로 감소할 때까지 최대 8회 step을 절반으로 줄인다.
        accepted_values = values.copy()
        current_residual = np.abs(polynomial)
        trial_correction = correction.copy()
        pending = safe.copy()
        for _backtrack in range(9):
            candidate = values - trial_correction
            candidate_polynomial = ((a * candidate + b) * candidate + c) * candidate + d
            improved = pending & (np.abs(candidate_polynomial) < current_residual)
            accepted_values[improved] = candidate[improved]
            pending &= ~improved
            if not np.any(pending):
                break
            trial_correction[pending] *= 0.5
        values = accepted_values
    return canonicalize_roots(values)


def newton_polish_roots_batch_scaled(
    scaled_roots: ArrayLike,
    normalized_coefficients: ArrayLike,
    *,
    steps: int = 4,
) -> NDArray[np.complex128]:
    """무차원 MLP 근 배치를 잔차 감소형 Newton 방법으로 동시에 보정한다."""

    values = np.asarray(scaled_roots, dtype=np.complex128).copy()
    coefficients = np.asarray(normalized_coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or coefficients.shape != (len(values), 3):
        raise ValueError("scaled_roots와 normalized_coefficients는 각각 (N, 3)이어야 합니다.")
    if steps < 0:
        raise ValueError("steps는 0 이상이어야 합니다.")
    b = coefficients[:, 0, None]
    c = coefficients[:, 1, None]
    d = coefficients[:, 2, None]
    for _ in range(steps):
        polynomial = ((values + b) * values + c) * values + d
        derivative = (3.0 * values + 2.0 * b) * values + c
        safe = np.abs(derivative) > 1e-10
        correction = np.zeros_like(values)
        correction[safe] = polynomial[safe] / derivative[safe]
        maximum_step = 0.5 * (1.0 + np.abs(values))
        too_large = np.abs(correction) > maximum_step
        correction[too_large] *= maximum_step[too_large] / np.abs(correction[too_large])

        accepted = values.copy()
        current_residual = np.abs(polynomial)
        trial_correction = correction.copy()
        pending = safe.copy()
        for _backtrack in range(9):
            candidate = values - trial_correction
            candidate_polynomial = ((candidate + b) * candidate + c) * candidate + d
            improved = pending & (np.abs(candidate_polynomial) < current_residual)
            accepted[improved] = candidate[improved]
            pending &= ~improved
            if not np.any(pending):
                break
            trial_correction[pending] *= 0.5
        values = accepted
    return canonicalize_roots(values)


def polynomial_residuals(
    coefficients: ArrayLike,
    roots: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    values = np.asarray(roots, dtype=np.complex128)
    a, b, c, d = coefficient_array
    polynomial = ((a * values + b) * values + c) * values + d
    absolute = np.abs(values)
    denominator = (
        abs(a) * absolute**3
        + abs(b) * absolute**2
        + abs(c) * absolute
        + abs(d)
    )
    coefficient_scale = max(abs(a), abs(b), abs(c), abs(d))
    numerical_floor = (
        np.finfo(np.float64).eps * coefficient_scale * (1.0 + absolute) ** 3
    )
    denominator = np.maximum(denominator, numerical_floor)
    return np.abs(polynomial), np.abs(polynomial) / denominator


def vieta_set_error(
    coefficients: ArrayLike,
    roots: ArrayLike,
) -> NDArray[np.float64]:
    """세 출력이 중복 없이 하나의 근 집합인지 비에타 관계로 검사한다."""

    a, b, c, d = np.asarray(coefficients, dtype=np.float64)
    if a == 0.0:
        raise ValueError("a는 0이 아니어야 합니다.")
    values = np.asarray(roots, dtype=np.complex128)
    if values.shape != (3,):
        raise ValueError("roots는 세 개여야 합니다.")
    expected = np.asarray([b / a, c / a, d / a], dtype=np.complex128)
    reconstructed = np.asarray(
        [
            -np.sum(values),
            values[0] * values[1] + values[0] * values[2] + values[1] * values[2],
            -(values[0] * values[1] * values[2]),
        ]
    )
    return np.abs(reconstructed - expected) / (1.0 + np.abs(expected))
