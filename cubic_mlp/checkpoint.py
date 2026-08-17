"""모델 체크포인트 저장과 안전한 복원."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import CubicRootMLP, ModelConfig


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CubicRootMLP, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {source}. 먼저 `python train.py`를 실행하세요."
        )
    # weights_only=True로 임의 객체의 pickle 복원을 막는다. 그래도 출처를
    # 모르는 체크포인트는 사용하지 않는 것이 원칙이다.
    payload = torch.load(source, map_location=device, weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("지원하지 않는 체크포인트 형식입니다.")
    config = ModelConfig.from_dict(payload["model_config"])
    model = CubicRootMLP(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    # 후속 코드에서 바로 연산할 수 있도록 통계만 ndarray로 복원한다.
    payload["input_mean_array"] = np.asarray(payload["input_mean"], dtype=np.float32)
    payload["input_std_array"] = np.asarray(payload["input_std"], dtype=np.float32)
    return model, payload
