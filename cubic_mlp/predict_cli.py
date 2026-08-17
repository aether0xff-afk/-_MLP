"""계수를 입력받아 MLP 근사근을 출력하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cardano import solve_cubic_cardano
from .inference import (
    newton_polish_roots,
    polynomial_residuals,
    predict_roots,
    vieta_set_error,
)


def _format_complex(value: complex) -> str:
    real = 0.0 if abs(value.real) < 5e-10 else value.real
    imag = 0.0 if abs(value.imag) < 5e-10 else value.imag
    if imag == 0.0:
        return f"{real:.8f}"
    return f"{real:.8f} {'+' if imag >= 0 else '-'} {abs(imag):.8f}i"


def _nonnegative_integer(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수여야 합니다.")
    return value


def _read_coefficients(args: argparse.Namespace) -> np.ndarray:
    supplied = [args.a, args.b, args.c, args.d]
    if all(value is None for value in supplied):
        text = input("a b c d를 공백으로 구분해 입력하세요: ")
        try:
            values = [float(token) for token in text.replace(",", " ").split()]
        except ValueError as error:
            raise SystemExit("계수는 실수여야 합니다.") from error
        if len(values) != 4:
            raise SystemExit("계수를 정확히 네 개 입력해야 합니다.")
        return np.asarray(values, dtype=np.float64)
    if any(value is None for value in supplied):
        raise SystemExit("--a, --b, --c, --d를 모두 지정하거나 모두 생략하세요.")
    return np.asarray(supplied, dtype=np.float64)


def _print_roots(
    title: str,
    coefficients: np.ndarray,
    roots: np.ndarray,
) -> None:
    absolute, normalized = polynomial_residuals(coefficients, roots)
    print(f"\n{title}")
    for index, (root, abs_residual, norm_residual) in enumerate(
        zip(roots, absolute, normalized, strict=True), start=1
    ):
        print(
            f"  x{index} = {_format_complex(complex(root))}  "
            f"|P(x)|={abs_residual:.3e}, normalized={norm_residual:.3e}"
        )
    set_error = vieta_set_error(coefficients, roots)
    print(f"  Vieta root-set error(max)={np.max(set_error):.3e}")
    coefficient_scale = float(np.max(np.abs(coefficients)))
    large_backward_error = (
        np.max(normalized) > 5e-2
        and np.max(absolute) / coefficient_scale > 1e-4
    )
    if large_backward_error or np.max(set_error) > 5e-2:
        print("  주의: 잔차 또는 근 집합 오차가 큽니다. 근사값의 신뢰도가 낮습니다.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="학습된 MLP로 삼차방정식의 세 근을 근사합니다."
    )
    parser.add_argument("--a", type=float)
    parser.add_argument("--b", type=float)
    parser.add_argument("--c", type=float)
    parser.add_argument("--d", type=float)
    parser.add_argument("--checkpoint", default="artifacts/cubic_mlp.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--polish-steps",
        type=_nonnegative_integer,
        default=0,
        help="MLP 출력 뒤 적용할 Newton 반복 횟수(기본 0)",
    )
    parser.add_argument(
        "--show-cardano",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="카르다노 공식 기준값도 표시",
    )
    parser.add_argument(
        "--show-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="켤레/실근 구조 정리 전 MLP 원출력도 표시",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    coefficients = _read_coefficients(args)
    a, b, c, d = coefficients
    print(f"방정식: ({a:g})x^3 + ({b:g})x^2 + ({c:g})x + ({d:g}) = 0")
    try:
        result = predict_roots(
            coefficients,
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if np.max(np.abs(result.input_z_scores)) > 5.0:
        print("경고: 입력 분포가 학습 데이터와 멉니다. 잔차를 특히 확인하세요.")
    if args.show_raw:
        _print_roots("MLP 원출력", coefficients, result.raw_roots)
    _print_roots("MLP 근사근(실수 계수 구조 반영)", coefficients, result.structured_roots)

    if args.polish_steps:
        polished = newton_polish_roots(
            result.structured_roots, coefficients, steps=args.polish_steps
        )
        _print_roots(
            f"MLP 초기값 + Newton {args.polish_steps}회",
            coefficients,
            polished,
        )
    if args.show_cardano:
        reference = solve_cubic_cardano(coefficients)
        _print_roots("카르다노 공식 기준값", coefficients, reference)


if __name__ == "__main__":
    main()
