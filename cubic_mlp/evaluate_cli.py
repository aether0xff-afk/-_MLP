"""저장한 체크포인트를 새 seed의 합성 데이터로 재평가한다."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .checkpoint import load_checkpoint
from .data import DatasetConfig, generate_dataset_with_regimes, make_model_arrays
from .metrics import evaluate_predictions, pairs_to_complex
from .inference import enforce_real_cubic_structure_batch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="삼차근 MLP를 독립 데이터로 평가합니다.")
    parser.add_argument("--checkpoint", default="artifacts/cubic_mlp.pt")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model, payload = load_checkpoint(args.checkpoint, device=args.device)
    dataset_config = DatasetConfig(**payload["dataset_config"])
    generated = generate_dataset_with_regimes(
        args.samples, seed=args.seed, config=dataset_config
    )
    inputs, targets, _ = make_model_arrays(generated.coefficients, generated.roots)
    standardized = (inputs - payload["input_mean_array"]) / payload["input_std_array"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(standardized)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            parts.append(model(batch.to(args.device)).cpu().numpy())
    predicted = np.concatenate(parts)
    predicted_roots = pairs_to_complex(predicted)
    raw_metrics = evaluate_predictions(
        inputs,
        predicted_roots,
        pairs_to_complex(targets),
        regimes=generated.regimes,
    )
    structured_roots = enforce_real_cubic_structure_batch(predicted_roots, inputs)
    metrics = evaluate_predictions(
        inputs,
        structured_roots,
        pairs_to_complex(targets),
        regimes=generated.regimes,
    )
    metrics["raw_mlp_before_structure"] = raw_metrics
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
