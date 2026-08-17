"""실수 계수 삼차방정식을 위한 수치적으로 보강한 카르다노 공식.

학습 라벨을 만드는 코드이므로 ``numpy.roots``를 호출하지 않는다. 세 실근인
casus irreducibilis에서는 복소 세제곱근 대신 카르다노 공식과 동치인 삼각함수
표현을 사용한다.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

# 판별식은 두 항의 상쇄로 0이 될 수 있다. 아래 값은 절대 오차가 아니라
# 두 항의 크기에 곱하는 상대 계수이며, float64 반올림 오차에 여유를 둔 값이다.
DEFAULT_DISCRIMINANT_TOL = 64.0 * np.finfo(np.float64).eps
DEFAULT_LEADING_TOL = 0.0


def _as_coefficient_matrix(coefficients: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    array = np.asarray(coefficients, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        if array.shape != (4,):
            raise ValueError("계수는 [a, b, c, d] 네 개여야 합니다.")
        array = array[None, :]
    elif array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("계수 배열의 모양은 (4,) 또는 (N, 4)여야 합니다.")
    if not np.all(np.isfinite(array)):
        raise ValueError("모든 계수는 유한한 실수여야 합니다.")
    return array, was_vector


def normalize_cubic_coefficients(
    coefficients: ArrayLike,
    *,
    leading_tol: float = DEFAULT_LEADING_TOL,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """계수를 monic/무차원 형태로 바꾸고 변수 복원 배율을 반환한다.

    ``x = scale * z``로 두어

    ``z^3 + Bn*z^2 + Cn*z + Dn = 0``

    을 만든다. 반환값은 ``([Bn, Cn, Dn], scale)``이며 각 정규화 계수의
    절댓값은 1 이하이다.
    """

    matrix, was_vector = _as_coefficient_matrix(coefficients)
    leading = matrix[:, 0]
    # 절대 크기가 아니라 행 안에서 선도계수가 차지하는 상대 크기로 판단한다.
    # 따라서 방정식 전체를 1e-100처럼 아주 작은 수로 곱해도 같은 입력이다.
    row_scale = np.max(np.abs(matrix), axis=1)
    if np.any(np.abs(leading) <= leading_tol * row_scale):
        raise ValueError("a가 0이거나 수치적으로 너무 작아 삼차방정식으로 볼 수 없습니다.")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        monic = matrix[:, 1:] / leading[:, None]
    if not np.all(np.isfinite(monic)):
        raise ValueError("선도계수로 나눈 계수비가 부동소수점 범위를 벗어납니다.")
    b, c, d = monic.T
    variable_scale = np.maximum.reduce(
        [
            np.abs(b),
            np.sqrt(np.abs(c)),
            np.cbrt(np.abs(d)),
        ]
    )
    # x^3=0만 배율 후보가 모두 0이다. 나머지는 작은 근을 가진 문제도 1로
    # 뭉개지 않고 고유 크기로 확대해, 정규화 공간의 경계까지 사용한다.
    variable_scale = np.where(variable_scale == 0.0, 1.0, variable_scale)
    normalized = np.column_stack(
        [
            b / variable_scale,
            (c / variable_scale) / variable_scale,
            ((d / variable_scale) / variable_scale) / variable_scale,
        ]
    )

    if was_vector:
        return normalized[0], variable_scale[0]
    return normalized, variable_scale


def cubic_discriminant_terms(
    normalized_monic_coefficients: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """monic 삼차식의 depressed-cubic ``p, q, Delta``를 반환한다."""

    coeffs = np.asarray(normalized_monic_coefficients, dtype=np.float64)
    if coeffs.shape[-1] != 3:
        raise ValueError("monic 계수는 [B, C, D] 세 개여야 합니다.")
    b = coeffs[..., 0]
    c = coeffs[..., 1]
    d = coeffs[..., 2]
    p = c - b**2 / 3.0
    q = 2.0 * b**3 / 27.0 - b * c / 3.0 + d
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3
    return p, q, delta


def _cardano_monic_batch(
    normalized: NDArray[np.float64],
    *,
    discriminant_tol: float,
) -> NDArray[np.complex128]:
    """정규화된 monic 삼차식들을 카르다노 공식으로 푼다."""

    b = normalized[:, 0]
    p, q, delta = cubic_discriminant_terms(normalized)
    roots = np.empty((len(normalized), 3), dtype=np.complex128)

    # Delta를 이루는 두 항의 크기에 비례한 허용오차를 쓴다. 고정된 == 0
    # 비교보다 중근 경계에서 안정적이다.
    delta_scale = np.abs((q / 2.0) ** 2) + np.abs((p / 3.0) ** 3)
    threshold = discriminant_tol * delta_scale
    one_real = delta > threshold
    three_real = delta < -threshold
    repeated = ~(one_real | three_real)

    if np.any(one_real):
        idx = np.flatnonzero(one_real)
        pi = p[idx]
        qi = q[idx]
        sqrt_delta = np.sqrt(np.maximum(delta[idx], 0.0))
        term_u = -qi / 2.0 + sqrt_delta
        term_v = -qi / 2.0 - sqrt_delta

        # 둘 중 절댓값이 큰 항만 직접 세제곱근을 취하고 uv=-p/3으로
        # 다른 항을 복원하면 큰 두 수의 뺄셈에서 오는 상쇄 오차가 줄어든다.
        choose_u = np.abs(term_u) >= np.abs(term_v)
        dominant_term = np.where(choose_u, term_u, term_v)
        dominant = np.cbrt(dominant_term)
        other_term = np.where(choose_u, term_v, term_u)
        safe = np.abs(dominant) > 16.0 * np.finfo(np.float64).eps
        other = np.where(safe, -pi / (3.0 * dominant), np.cbrt(other_term))
        u = np.where(choose_u, dominant, other)
        v = np.where(choose_u, other, dominant)

        y0 = u + v
        pair_real = -0.5 * y0
        pair_imag = (np.sqrt(3.0) / 2.0) * (u - v)
        roots[idx, 0] = y0
        roots[idx, 1] = pair_real + 1j * pair_imag
        roots[idx, 2] = pair_real - 1j * pair_imag

    if np.any(three_real):
        idx = np.flatnonzero(three_real)
        pi = p[idx]
        qi = q[idx]
        radius = 2.0 * np.sqrt(np.maximum(-pi / 3.0, 0.0))
        denominator = 2.0 * np.sqrt(np.maximum(-(pi / 3.0) ** 3, 0.0))
        cosine = np.clip(-qi / denominator, -1.0, 1.0)
        theta = np.arccos(cosine)
        for k in range(3):
            roots[idx, k] = radius * np.cos((theta + 2.0 * np.pi * k) / 3.0)

    if np.any(repeated):
        idx = np.flatnonzero(repeated)
        u = np.cbrt(-q[idx] / 2.0)
        roots[idx, 0] = 2.0 * u
        roots[idx, 1] = -u
        roots[idx, 2] = -u

    roots -= (b / 3.0)[:, None]
    tiny_imag = np.abs(roots.imag) <= 32.0 * np.finfo(np.float64).eps
    roots.imag[tiny_imag] = 0.0
    return roots


def canonicalize_roots(roots: ArrayLike) -> NDArray[np.complex128]:
    """근들을 ``(실수부, 허수부)`` 사전식 순서로 정렬한다."""

    array = np.asarray(roots, dtype=np.complex128)
    was_vector = array.ndim == 1
    if was_vector:
        if array.shape != (3,):
            raise ValueError("근은 세 개여야 합니다.")
        array = array[None, :]
    elif array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("근 배열의 모양은 (3,) 또는 (N, 3)이어야 합니다.")

    order = np.lexsort((array.imag, array.real), axis=1)
    result = np.take_along_axis(array, order, axis=1)
    return result[0] if was_vector else result


def solve_cubic_cardano_batch(
    coefficients: ArrayLike,
    *,
    discriminant_tol: float = DEFAULT_DISCRIMINANT_TOL,
    leading_tol: float = DEFAULT_LEADING_TOL,
) -> NDArray[np.complex128]:
    """여러 실수 계수 삼차방정식의 세 복소근을 카르다노 공식으로 구한다."""

    matrix, _ = _as_coefficient_matrix(coefficients)
    normalized, variable_scale = normalize_cubic_coefficients(
        matrix, leading_tol=leading_tol
    )
    roots = _cardano_monic_batch(
        np.asarray(normalized), discriminant_tol=discriminant_tol
    )
    roots *= np.asarray(variable_scale)[:, None]
    return canonicalize_roots(roots)


def solve_cubic_cardano(
    coefficients: Iterable[float],
    *,
    discriminant_tol: float = DEFAULT_DISCRIMINANT_TOL,
    leading_tol: float = DEFAULT_LEADING_TOL,
) -> NDArray[np.complex128]:
    """``a*x^3+b*x^2+c*x+d=0``의 세 근을 카르다노 공식으로 구한다."""

    roots = solve_cubic_cardano_batch(
        coefficients,
        discriminant_tol=discriminant_tol,
        leading_tol=leading_tol,
    )
    return roots[0] if roots.ndim == 2 and roots.shape[0] == 1 else roots
