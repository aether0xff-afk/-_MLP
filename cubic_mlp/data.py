"""카르다노 라벨을 사용하는 합성 학습 데이터 생성."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cardano import normalize_cubic_coefficients, solve_cubic_cardano_batch


@dataclass(frozen=True)
class DatasetConfig:
    """기본 12만 샘플(train/validation/test 합계) 데이터의 분포 설정."""

    coefficient_random_fraction: float = 0.35
    three_real_fraction: float = 0.25
    complex_pair_fraction: float = 0.25
    near_multiple_fraction: float = 0.10
    exact_multiple_fraction: float = 0.05
    coefficient_limit: float = 5.0
    root_unit_limit: float = 3.0
    min_log_scale: float = -1.0
    max_log_scale: float = 1.0
    near_gap_log_min: float = -4.0
    near_gap_log_max: float = -1.0

    def fractions(self) -> tuple[float, ...]:
        return (
            self.coefficient_random_fraction,
            self.three_real_fraction,
            self.complex_pair_fraction,
            self.near_multiple_fraction,
            self.exact_multiple_fraction,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedDataset:
    coefficients: NDArray[np.float64]
    roots: NDArray[np.complex128]
    regimes: NDArray[np.str_]


def coefficients_from_roots(roots: ArrayLike) -> NDArray[np.float64]:
    """세 근으로부터 비에타 관계를 이용해 monic 실수 계수를 만든다."""

    array = np.asarray(roots, dtype=np.complex128)
    was_vector = array.ndim == 1
    if was_vector:
        if array.shape != (3,):
            raise ValueError("근은 세 개여야 합니다.")
        array = array[None, :]
    elif array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("근 배열의 모양은 (3,) 또는 (N, 3)이어야 합니다.")

    r1, r2, r3 = array.T
    coefficients = np.column_stack(
        [
            np.ones(len(array), dtype=np.complex128),
            -(r1 + r2 + r3),
            r1 * r2 + r1 * r3 + r2 * r3,
            -(r1 * r2 * r3),
        ]
    )
    imaginary_error = np.max(np.abs(coefficients.imag), axis=1)
    scale = np.maximum(1.0, np.max(np.abs(coefficients.real), axis=1))
    if np.any(imaginary_error > 1e-10 * scale):
        raise ValueError("주어진 근들은 실수 계수 다항식을 만들지 않습니다.")
    result = coefficients.real.astype(np.float64)
    return result[0] if was_vector else result


def _allocate_counts(n_samples: int, fractions: tuple[float, ...]) -> list[int]:
    weights = np.asarray(fractions, dtype=np.float64)
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("데이터 구성 비율은 음수가 아니며 합이 1이어야 합니다.")
    raw = n_samples * weights
    counts = np.floor(raw).astype(int)
    remainder = n_samples - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts.tolist()


def _sample_scale(rng: np.random.Generator, count: int, config: DatasetConfig) -> NDArray[np.float64]:
    return 10.0 ** rng.uniform(config.min_log_scale, config.max_log_scale, count)


def _root_generated_coefficients(
    rng: np.random.Generator,
    count: int,
    regime: str,
    config: DatasetConfig,
) -> NDArray[np.float64]:
    if count == 0:
        return np.empty((0, 4), dtype=np.float64)

    scale = _sample_scale(rng, count, config)
    limit = config.root_unit_limit

    if regime == "three_real":
        roots = rng.uniform(-limit, limit, (count, 3)) * scale[:, None]
    elif regime == "complex_pair":
        real_root = rng.uniform(-limit, limit, count) * scale
        pair_real = rng.uniform(-limit, limit, count) * scale
        pair_imag = rng.uniform(0.10, limit, count) * scale
        roots = np.column_stack(
            [real_root, pair_real + 1j * pair_imag, pair_real - 1j * pair_imag]
        )
    elif regime == "near_multiple":
        center = rng.uniform(-limit, limit, count) * scale
        third = rng.uniform(-limit, limit, count) * scale
        gap = scale * 10.0 ** rng.uniform(
            config.near_gap_log_min, config.near_gap_log_max, count
        )
        roots = np.empty((count, 3), dtype=np.complex128)
        make_complex = rng.random(count) < 0.5
        roots[~make_complex] = np.column_stack(
            [
                third[~make_complex],
                center[~make_complex] - gap[~make_complex],
                center[~make_complex] + gap[~make_complex],
            ]
        )
        roots[make_complex] = np.column_stack(
            [
                third[make_complex],
                center[make_complex] + 1j * gap[make_complex],
                center[make_complex] - 1j * gap[make_complex],
            ]
        )
    elif regime == "exact_multiple":
        center = rng.uniform(-limit, limit, count) * scale
        third = rng.uniform(-limit, limit, count) * scale
        roots = np.column_stack([third, center, center]).astype(np.complex128)
        triple = rng.random(count) < 0.25
        roots[triple] = center[triple, None]
    else:
        raise ValueError(f"알 수 없는 데이터 regime: {regime}")

    return coefficients_from_roots(roots)


def generate_dataset_with_regimes(
    n_samples: int,
    *,
    seed: int = 20260816,
    config: DatasetConfig | None = None,
) -> GeneratedDataset:
    """계수와 카르다노 공식 라벨을 재현 가능하게 생성한다."""

    if n_samples <= 0:
        raise ValueError("n_samples는 양수여야 합니다.")
    config = config or DatasetConfig()
    rng = np.random.default_rng(seed)
    counts = _allocate_counts(n_samples, config.fractions())
    names = (
        "coefficient_random",
        "three_real",
        "complex_pair",
        "near_multiple",
        "exact_multiple",
    )

    coefficient_parts: list[NDArray[np.float64]] = []
    regime_parts: list[NDArray[np.str_]] = []
    for name, count in zip(names, counts, strict=True):
        if name == "coefficient_random":
            monic_tail = rng.uniform(
                -config.coefficient_limit,
                config.coefficient_limit,
                (count, 3),
            )
            coefficients = np.column_stack([np.ones(count), monic_tail])
        else:
            coefficients = _root_generated_coefficients(
                rng, count, name, config
            )
        coefficient_parts.append(coefficients)
        regime_parts.append(np.full(count, name, dtype="U24"))

    coefficients = np.concatenate(coefficient_parts, axis=0)
    regimes = np.concatenate(regime_parts, axis=0)
    permutation = rng.permutation(n_samples)
    coefficients = coefficients[permutation]
    regimes = regimes[permutation]

    # 중요: root-generated 부분도 원래 뿌리를 정답으로 재사용하지 않고 모든
    # 라벨을 이 카르다노 구현으로 다시 계산한다.
    roots = solve_cubic_cardano_batch(coefficients)
    return GeneratedDataset(coefficients=coefficients, roots=roots, regimes=regimes)


def generate_dataset(
    n_samples: int,
    *,
    seed: int = 20260816,
    config: DatasetConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """간단한 공개 API: ``(coefficients, Cardano roots)``를 반환한다."""

    dataset = generate_dataset_with_regimes(n_samples, seed=seed, config=config)
    return dataset.coefficients, dataset.roots


def make_model_arrays(
    coefficients: ArrayLike,
    roots: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    """원래 계수/근을 MLP 입력 3개와 출력 ``(3,2)``로 변환한다."""

    normalized, variable_scale = normalize_cubic_coefficients(coefficients)
    roots_array = np.asarray(roots, dtype=np.complex128)
    if roots_array.ndim != 2 or roots_array.shape != (len(normalized), 3):
        raise ValueError("roots의 모양은 (N, 3)이어야 합니다.")
    scaled_roots = roots_array / np.asarray(variable_scale)[:, None]
    targets = np.stack([scaled_roots.real, scaled_roots.imag], axis=-1)
    return (
        np.asarray(normalized, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(variable_scale, dtype=np.float64),
    )
