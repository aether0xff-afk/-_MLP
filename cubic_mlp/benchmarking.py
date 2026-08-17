"""카르다노·NumPy·MLP·하이브리드 삼차근 해법의 공정한 벤치마크."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .cardano import (
    canonicalize_roots,
    normalize_cubic_coefficients,
    solve_cubic_cardano,
    solve_cubic_cardano_batch,
)
from .data import DatasetConfig, GeneratedDataset, generate_dataset_with_regimes
from .inference import CubicMLPSolver
from .metrics import evaluate_predictions, optimally_match_roots


def generate_hard_dataset(
    n_samples: int,
    *,
    seed: int = 20260820,
) -> GeneratedDataset:
    """근접/정확 중근과 극단 근 크기를 과대표집한 난제 세트."""

    config = DatasetConfig(
        coefficient_random_fraction=0.10,
        three_real_fraction=0.10,
        complex_pair_fraction=0.10,
        near_multiple_fraction=0.50,
        exact_multiple_fraction=0.20,
        coefficient_limit=5.0,
        root_unit_limit=3.0,
        min_log_scale=-12.0,
        max_log_scale=12.0,
        near_gap_log_min=-10.0,
        near_gap_log_max=-3.0,
    )
    return generate_dataset_with_regimes(n_samples, seed=seed, config=config)


def numpy_roots_batch(coefficients: NDArray[np.float64]) -> NDArray[np.complex128]:
    """NumPy companion-matrix 계열 기준선. NumPy API가 행 배치를 지원하지 않아 반복한다."""

    roots = np.empty((len(coefficients), 3), dtype=np.complex128)
    for index, row in enumerate(coefficients):
        roots[index] = np.roots(row)
    return canonicalize_roots(roots)


def scalar_cardano_loop(coefficients: NDArray[np.float64]) -> NDArray[np.complex128]:
    """한 식씩 고전 카르다노 공식을 호출하는 일반적인 사용 방식."""

    return np.stack([solve_cubic_cardano(row) for row in coefficients])


def _median_runtime(
    function: Callable[[], Any],
    *,
    repeats: int,
    warmups: int = 1,
) -> tuple[float, Any]:
    for _ in range(warmups):
        function()
    durations: list[float] = []
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return float(statistics.median(durations)), result


def _method_speed_record(seconds: float, sample_count: int) -> dict[str, float | int]:
    return {
        "sample_count": sample_count,
        "median_seconds": seconds,
        "microseconds_per_equation": seconds / sample_count * 1e6,
        "equations_per_second": sample_count / seconds,
    }


def _accuracy_record(
    coefficients: NDArray[np.float64],
    predicted_roots: NDArray[np.complex128],
    target_roots: NDArray[np.complex128],
    regimes: NDArray[np.str_] | None = None,
) -> dict[str, Any]:
    normalized, scales = normalize_cubic_coefficients(coefficients)
    predicted_scaled = predicted_roots / np.asarray(scales)[:, None]
    target_scaled = target_roots / np.asarray(scales)[:, None]
    result = evaluate_predictions(
        normalized,
        predicted_scaled,
        target_scaled,
        regimes=regimes,
    )
    _, errors = optimally_match_roots(predicted_scaled, target_scaled)
    sample_max = errors.max(axis=1)
    result["within_tolerance_rate"] = {
        "1e-3": float(np.mean(sample_max <= 1e-3)),
        "1e-2": float(np.mean(sample_max <= 1e-2)),
        "5e-2": float(np.mean(sample_max <= 5e-2)),
        "1e-1": float(np.mean(sample_max <= 1e-1)),
    }
    return result


def run_crossover_scan(
    solver: CubicMLPSolver,
    coefficients: NDArray[np.float64],
    *,
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64, 256, 1_024, 4_096, 16_384),
    repeats: int = 5,
) -> dict[str, Any]:
    """배치 크기별로 MLP가 전통적 행 단위 풀이를 추월하는 지점을 찾는다."""

    usable_sizes = sorted({size for size in batch_sizes if 0 < size <= len(coefficients)})
    if len(coefficients) not in usable_sizes:
        usable_sizes.append(len(coefficients))

    rows: list[dict[str, Any]] = []
    for size in usable_sizes:
        subset = coefficients[:size]
        functions: dict[str, Callable[[], Any]] = {
            "scalar_cardano_loop": lambda subset=subset: scalar_cardano_loop(subset),
            "numpy_roots_loop": lambda subset=subset: numpy_roots_batch(subset),
            "cardano_vectorized": lambda subset=subset: solve_cubic_cardano_batch(subset),
            "mlp_batch": lambda subset=subset, size=size: solver.predict_batch(
                subset, batch_size=size
            ).structured_roots,
            "mlp_newton4": lambda subset=subset, size=size: solver.predict_polished_batch(
                subset, batch_size=size, newton_steps=4
            ),
        }
        timings: dict[str, float] = {}
        for name, function in functions.items():
            seconds, _ = _median_runtime(function, repeats=repeats, warmups=2)
            timings[name] = seconds / size * 1e6
        rows.append({"batch_size": size, "microseconds_per_equation": timings})

    def first_win(candidate: str, baseline: str) -> int | None:
        for row in rows:
            timing = row["microseconds_per_equation"]
            if timing[candidate] < timing[baseline]:
                return int(row["batch_size"])
        return None

    return {
        "rows": rows,
        "first_measured_win": {
            "mlp_vs_scalar_cardano": first_win("mlp_batch", "scalar_cardano_loop"),
            "mlp_vs_numpy_roots": first_win("mlp_batch", "numpy_roots_loop"),
            "mlp_newton4_vs_scalar_cardano": first_win(
                "mlp_newton4", "scalar_cardano_loop"
            ),
            "mlp_newton4_vs_numpy_roots": first_win("mlp_newton4", "numpy_roots_loop"),
            "mlp_vs_vectorized_cardano": first_win("mlp_batch", "cardano_vectorized"),
        },
        "note": "교차점은 이 컴퓨터와 이 구현에서 측정한 배치 크기 격자의 최초 승리점이다.",
    }


def run_benchmark(
    *,
    checkpoint_path: str | Path = "artifacts/cubic_mlp.pt",
    output_path: str | Path = "artifacts/speed_benchmark.json",
    samples: int = 20_000,
    hard_samples: int = 5_000,
    repeats: int = 5,
    single_repeats: int = 30,
    batch_size: int = 65_536,
    device: str = "auto",
    seed: int = 20260819,
    crossover_sizes: tuple[int, ...] = (1, 4, 16, 64, 256, 1_024, 4_096, 16_384),
) -> dict[str, Any]:
    if samples <= 0 or hard_samples <= 0 or repeats <= 0 or single_repeats <= 0:
        raise ValueError("샘플 수와 반복 횟수는 양수여야 합니다.")

    general = generate_dataset_with_regimes(samples, seed=seed)
    hard = generate_hard_dataset(hard_samples, seed=seed + 1)

    load_start = time.perf_counter()
    solver = CubicMLPSolver(checkpoint_path, device=device)
    cold_load_seconds = time.perf_counter() - load_start

    coefficients = general.coefficients
    timing_functions: dict[str, Callable[[], NDArray[np.complex128]]] = {
        "cardano_vectorized": lambda: solve_cubic_cardano_batch(coefficients),
        "numpy_roots_loop": lambda: numpy_roots_batch(coefficients),
        "mlp_batch": lambda: solver.predict_batch(
            coefficients, batch_size=batch_size
        ).structured_roots,
        "mlp_newton4": lambda: solver.predict_polished_batch(
            coefficients, batch_size=batch_size, newton_steps=4
        ),
        "hybrid_mlp_cardano": lambda: solver.predict_hybrid(
            coefficients, batch_size=batch_size
        ).roots,
    }

    speed: dict[str, Any] = {}
    timed_outputs: dict[str, NDArray[np.complex128]] = {}
    for name, function in timing_functions.items():
        seconds, output = _median_runtime(function, repeats=repeats)
        speed[name] = _method_speed_record(seconds, samples)
        timed_outputs[name] = output

    numpy_seconds = float(speed["numpy_roots_loop"]["median_seconds"])
    cardano_seconds = float(speed["cardano_vectorized"]["median_seconds"])
    for record in speed.values():
        method_seconds = float(record["median_seconds"])
        record["speedup_vs_numpy"] = numpy_seconds / method_seconds
        record["speedup_vs_vectorized_cardano"] = cardano_seconds / method_seconds

    single_coefficients = coefficients[:1]
    single_functions: dict[str, Callable[[], Any]] = {
        "cardano_vectorized": lambda: solve_cubic_cardano_batch(single_coefficients),
        "numpy_roots_loop": lambda: numpy_roots_batch(single_coefficients),
        "mlp_warm": lambda: solver.predict_batch(single_coefficients, batch_size=1),
        "mlp_newton4_warm": lambda: solver.predict_polished_batch(
            single_coefficients, batch_size=1, newton_steps=4
        ),
        "hybrid_warm": lambda: solver.predict_hybrid(single_coefficients, batch_size=1),
    }
    single_latency: dict[str, Any] = {}
    for name, function in single_functions.items():
        seconds, _ = _median_runtime(
            function, repeats=single_repeats, warmups=2
        )
        single_latency[name] = {
            "median_seconds": seconds,
            "microseconds": seconds * 1e6,
        }

    general_accuracy = {
        name: _accuracy_record(
            coefficients,
            roots,
            general.roots,
            general.regimes,
        )
        for name, roots in timed_outputs.items()
    }
    general_hybrid = solver.predict_hybrid(coefficients, batch_size=batch_size)

    hard_outputs: dict[str, NDArray[np.complex128]] = {
        "cardano_vectorized": solve_cubic_cardano_batch(hard.coefficients),
        "numpy_roots_loop": numpy_roots_batch(hard.coefficients),
        "mlp_batch": solver.predict_batch(
            hard.coefficients, batch_size=batch_size
        ).structured_roots,
        "mlp_newton4": solver.predict_polished_batch(
            hard.coefficients, batch_size=batch_size, newton_steps=4
        ),
    }
    hard_hybrid = solver.predict_hybrid(hard.coefficients, batch_size=batch_size)
    hard_outputs["hybrid_mlp_cardano"] = hard_hybrid.roots
    hard_accuracy = {
        name: _accuracy_record(
            hard.coefficients,
            roots,
            hard.roots,
            hard.regimes,
        )
        for name, roots in hard_outputs.items()
    }
    crossover = run_crossover_scan(
        solver,
        coefficients,
        batch_sizes=crossover_sizes,
        repeats=repeats,
    )

    document: dict[str, Any] = {
        "benchmark_scope": {
            "device": str(solver.device),
            "checkpoint": str(checkpoint_path),
            "general_samples": samples,
            "hard_samples": hard_samples,
            "timing_repeats": repeats,
            "single_timing_repeats": single_repeats,
            "batch_size": batch_size,
            "checkpoint_cold_load_seconds": cold_load_seconds,
            "warm_timing_excludes_checkpoint_load": True,
            "seed": seed,
        },
        "batch_speed": speed,
        "single_equation_latency": single_latency,
        "general_accuracy": general_accuracy,
        "hard_accuracy": hard_accuracy,
        "batch_crossover": crossover,
        "hybrid_fallback": {
            "general_rate": general_hybrid.fallback_rate,
            "general_count": int(general_hybrid.fallback_mask.sum()),
            "hard_rate": hard_hybrid.fallback_rate,
            "hard_count": int(hard_hybrid.fallback_mask.sum()),
            "residual_threshold": 0.05,
            "vieta_threshold": 0.05,
            "zscore_threshold": 6.0,
            "discriminant_relative_threshold": 1e-6,
            "newton_steps_before_fallback": 3,
        },
        "hard_dataset": {
            "description": "50% near-multiple, 20% exact-multiple, root scale 1e-12..1e12, relative gap 1e-10..1e-3",
            "regime_counts": {
                str(name): int(np.sum(hard.regimes == name))
                for name in np.unique(hard.regimes)
            },
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return document


def _print_summary(document: dict[str, Any]) -> None:
    scope = document["benchmark_scope"]
    print(
        f"장치={scope['device']}, batch={scope['general_samples']:,}, "
        f"cold model load={scope['checkpoint_cold_load_seconds']:.4f}s"
    )
    print("\n배치 속도(체크포인트 로드 제외, 전처리·후처리 포함)")
    print(f"{'method':24s} {'us/eq':>12s} {'eq/s':>14s} {'vs NumPy':>12s} {'vs Cardano':>12s}")
    for name, record in document["batch_speed"].items():
        print(
            f"{name:24s} "
            f"{record['microseconds_per_equation']:12.3f} "
            f"{record['equations_per_second']:14,.0f} "
            f"{record['speedup_vs_numpy']:12.2f}x "
            f"{record['speedup_vs_vectorized_cardano']:12.2f}x"
        )

    print("\n단일 식 warm latency")
    for name, record in document["single_equation_latency"].items():
        print(f"  {name:24s} {record['microseconds']:10.2f} us")

    print("\n무차원 root RMSE")
    for suite_name in ("general_accuracy", "hard_accuracy"):
        print(f"  {suite_name}")
        for name, metrics in document[suite_name].items():
            print(
                f"    {name:22s} RMSE={metrics['root_rmse']:.6g}, "
                f"p95-max={metrics['sample_max_error']['p95']:.6g}"
            )
    fallback = document["hybrid_fallback"]
    print(
        f"\nhybrid fallback: general={fallback['general_rate']:.2%}, "
        f"hard={fallback['hard_rate']:.2%}"
    )
    print("\n배치 교차점(측정한 크기 중 최초)")
    for name, size in document["batch_crossover"]["first_measured_win"].items():
        print(f"  {name:38s} {size if size is not None else '없음'}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cardano, NumPy, MLP, hybrid 삼차근 속도/정확도를 비교합니다."
    )
    parser.add_argument("--checkpoint", default="artifacts/cubic_mlp.pt")
    parser.add_argument("--output", default="artifacts/speed_benchmark.json")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--hard-samples", type=int, default=5_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--single-repeats", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--crossover-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1_024, 4_096, 16_384],
        help="속도 교차점을 측정할 배치 크기 목록",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    document = run_benchmark(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        samples=args.samples,
        hard_samples=args.hard_samples,
        repeats=args.repeats,
        single_repeats=args.single_repeats,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        crossover_sizes=tuple(args.crossover_sizes),
    )
    _print_summary(document)
    print(f"\nJSON 저장: {args.output}")


if __name__ == "__main__":
    main()
