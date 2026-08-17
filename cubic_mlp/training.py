"""MLP 학습 파이프라인과 명령행 인터페이스."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import save_checkpoint
from .data import (
    DatasetConfig,
    GeneratedDataset,
    generate_dataset_with_regimes,
    make_model_arrays,
)
from .losses import CubicRootLoss
from .metrics import evaluate_predictions, pairs_to_complex
from .model import CubicRootMLP, ModelConfig
from .inference import enforce_real_cubic_structure_batch


@dataclass(frozen=True)
class TrainingConfig:
    train_samples: int = 100_000
    validation_samples: int = 10_000
    test_samples: int = 10_000
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    smooth_l1_beta: float = 0.05
    residual_weight: float = 0.05
    vieta_weight: float = 0.02
    auxiliary_warmup_epochs: int = 10
    gradient_clip_norm: float = 1.0
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 18
    seed: int = 20260816


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 사용할 수 없습니다.")
    return device


def _prepare_split(
    dataset: GeneratedDataset,
    input_mean: np.ndarray | None = None,
    input_std: np.ndarray | None = None,
) -> tuple[TensorDataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs, targets, scales = make_model_arrays(dataset.coefficients, dataset.roots)
    if input_mean is None:
        input_mean = inputs.mean(axis=0, dtype=np.float64).astype(np.float32)
    if input_std is None:
        input_std = inputs.std(axis=0, dtype=np.float64).astype(np.float32)
        input_std = np.maximum(input_std, 1e-6)
    standardized = (inputs - input_mean) / input_std
    tensor_dataset = TensorDataset(
        torch.from_numpy(standardized),
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
    )
    return tensor_dataset, inputs, targets, scales, np.asarray(dataset.regimes)


@torch.inference_mode()
def _predict_loader(
    model: CubicRootMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    coefficient_parts: list[np.ndarray] = []
    model.eval()
    for standardized, coefficients, targets in loader:
        prediction = model(standardized.to(device))
        prediction_parts.append(prediction.cpu().numpy())
        target_parts.append(targets.numpy())
        coefficient_parts.append(coefficients.numpy())
    return (
        np.concatenate(prediction_parts),
        np.concatenate(target_parts),
        np.concatenate(coefficient_parts),
    )


def _validation_rmse(
    model: CubicRootMLP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    predicted_pairs, target_pairs, coefficients = _predict_loader(model, loader, device)
    predicted_roots = enforce_real_cubic_structure_batch(
        pairs_to_complex(predicted_pairs), coefficients
    )
    metrics = evaluate_predictions(
        coefficients,
        predicted_roots,
        pairs_to_complex(target_pairs),
    )
    return float(metrics["root_rmse"])


def train_model(
    training_config: TrainingConfig,
    *,
    dataset_config: DatasetConfig | None = None,
    model_config: ModelConfig | None = None,
    checkpoint_path: str | Path = "artifacts/cubic_mlp.pt",
    metrics_path: str | Path = "artifacts/metrics.json",
    requested_device: str = "auto",
) -> dict[str, Any]:
    """데이터 생성부터 최종 독립 테스트까지 전체 학습을 수행한다."""

    dataset_config = dataset_config or DatasetConfig()
    model_config = model_config or ModelConfig()
    set_reproducible_seed(training_config.seed)
    device = choose_device(requested_device)
    print(f"장치: {device}")
    print(
        "데이터 생성: "
        f"train={training_config.train_samples:,}, "
        f"validation={training_config.validation_samples:,}, "
        f"test={training_config.test_samples:,}"
    )
    generation_start = time.perf_counter()
    train_data = generate_dataset_with_regimes(
        training_config.train_samples,
        seed=training_config.seed,
        config=dataset_config,
    )
    validation_data = generate_dataset_with_regimes(
        training_config.validation_samples,
        seed=training_config.seed + 1,
        config=dataset_config,
    )
    test_data = generate_dataset_with_regimes(
        training_config.test_samples,
        seed=training_config.seed + 2,
        config=dataset_config,
    )
    print(f"데이터 생성 완료: {time.perf_counter() - generation_start:.2f}초")

    train_dataset, train_inputs, _, _, _ = _prepare_split(train_data)
    input_mean = train_inputs.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(
        train_inputs.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6
    )
    # 첫 호출은 통계를 얻기 위한 것이므로 같은 배열로 정확히 다시 구성한다.
    train_dataset, _, _, _, train_regimes = _prepare_split(
        train_data, input_mean, input_std
    )
    validation_dataset, _, _, _, validation_regimes = _prepare_split(
        validation_data, input_mean, input_std
    )
    test_dataset, _, _, test_scales, test_regimes = _prepare_split(
        test_data, input_mean, input_std
    )

    generator = torch.Generator().manual_seed(training_config.seed)
    loader_options = {
        "batch_size": training_config.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = CubicRootMLP(model_config).to(device)
    print(f"MLP 파라미터 수: {model.parameter_count:,}")
    objective = CubicRootLoss(
        smooth_l1_beta=training_config.smooth_l1_beta,
        residual_weight=training_config.residual_weight,
        vieta_weight=training_config.vieta_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience,
        min_lr=1e-6,
    )

    history: list[dict[str, float | int]] = []
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        auxiliary_factor = min(
            1.0,
            epoch / max(1, training_config.auxiliary_warmup_epochs),
        )
        running_total = 0.0
        running_match = 0.0
        seen = 0
        for standardized, normalized_coefficients, targets in train_loader:
            standardized = standardized.to(device, non_blocking=True)
            normalized_coefficients = normalized_coefficients.to(
                device, non_blocking=True
            )
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(standardized)
            total_loss, components = objective(
                predictions,
                targets,
                normalized_coefficients,
                auxiliary_factor=auxiliary_factor,
            )
            total_loss.backward()
            clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()

            batch_size = len(standardized)
            seen += batch_size
            running_total += float(total_loss.detach()) * batch_size
            running_match += float(components["matching"]) * batch_size

        validation_rmse = _validation_rmse(model, validation_loader, device)
        scheduler.step(validation_rmse)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": running_total / seen,
            "train_matching_loss": running_match / seen,
            "validation_root_rmse": validation_rmse,
            "learning_rate": learning_rate,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:03d} | loss {epoch_record['train_total_loss']:.6f} "
            f"| val RMSE {validation_rmse:.6f} | lr {learning_rate:.2e}"
        )

        if validation_rmse < best_rmse - 1e-6:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                print(f"조기 종료: {epoch} epoch (최적 epoch={best_epoch})")
                break

    if best_state is None:
        raise RuntimeError("학습 중 유효한 모델 상태를 얻지 못했습니다.")
    model.load_state_dict(best_state)
    model.to(device)

    predicted_pairs, target_pairs, normalized_coefficients = _predict_loader(
        model, test_loader, device
    )
    predicted_scaled = pairs_to_complex(predicted_pairs)
    target_scaled = pairs_to_complex(target_pairs)
    raw_test_metrics = evaluate_predictions(
        normalized_coefficients,
        predicted_scaled,
        target_scaled,
        regimes=test_regimes,
    )
    structured_scaled = enforce_real_cubic_structure_batch(
        predicted_scaled, normalized_coefficients
    )
    test_metrics = evaluate_predictions(
        normalized_coefficients,
        structured_scaled,
        target_scaled,
        regimes=test_regimes,
    )
    test_metrics["raw_mlp_before_structure"] = raw_test_metrics

    # 실제 x 단위의 오차도 기록한다. 규모가 다른 문제를 공정하게 비교하는
    # 주 지표는 위의 무차원 오차이며, 아래 값은 사용자가 체감할 보조 지표다.
    from .metrics import optimally_match_roots

    _, original_errors = optimally_match_roots(
        structured_scaled * test_scales[:, None],
        target_scaled * test_scales[:, None],
    )
    test_metrics["original_unit_root_mae"] = float(original_errors.mean())
    test_metrics["original_unit_root_rmse"] = float(
        np.sqrt(np.mean(original_errors**2))
    )

    total_seconds = time.perf_counter() - training_start
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": best_state,
        "model_config": model_config.to_dict(),
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "dataset_config": dataset_config.to_dict(),
        "training_config": asdict(training_config),
        "best_epoch": best_epoch,
        "best_validation_root_rmse": best_rmse,
        "test_metrics": test_metrics,
        "history": history,
    }
    save_checkpoint(checkpoint_path, payload)

    metrics_document = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "parameter_count": model.parameter_count,
        "training_seconds": total_seconds,
        "best_epoch": best_epoch,
        "best_validation_root_rmse": best_rmse,
        "test_metrics": test_metrics,
        "training_config": asdict(training_config),
        "dataset_config": dataset_config.to_dict(),
        "model_config": model_config.to_dict(),
    }
    metrics_destination = Path(metrics_path)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.write_text(
        json.dumps(metrics_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"체크포인트 저장: {checkpoint_path}")
    print(f"평가 결과 저장: {metrics_path}")
    print(
        f"test scaled root MAE={test_metrics['root_mae']:.6f}, "
        f"RMSE={test_metrics['root_rmse']:.6f}"
    )
    return metrics_document


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="카르다노 데이터로 삼차방정식 근 MLP를 학습합니다."
    )
    defaults = TrainingConfig()
    parser.add_argument("--train-samples", type=int, default=defaults.train_samples)
    parser.add_argument(
        "--validation-samples", type=int, default=defaults.validation_samples
    )
    parser.add_argument("--test-samples", type=int, default=defaults.test_samples)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=defaults.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 등")
    parser.add_argument("--output", default="artifacts/cubic_mlp.pt")
    parser.add_argument("--metrics-output", default="artifacts/metrics.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    defaults = TrainingConfig()
    config = TrainingConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        smooth_l1_beta=defaults.smooth_l1_beta,
        residual_weight=defaults.residual_weight,
        vieta_weight=defaults.vieta_weight,
        auxiliary_warmup_epochs=defaults.auxiliary_warmup_epochs,
        gradient_clip_norm=defaults.gradient_clip_norm,
        scheduler_patience=defaults.scheduler_patience,
        scheduler_factor=defaults.scheduler_factor,
        early_stopping_patience=args.patience,
        seed=args.seed,
    )
    train_model(
        config,
        checkpoint_path=args.output,
        metrics_path=args.metrics_output,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
