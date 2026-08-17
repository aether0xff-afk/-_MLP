"""삼차방정식 근 집합을 출력하는 다층 퍼셉트론."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 3
    hidden_sizes: tuple[int, ...] = (128, 256, 256, 128)
    output_dim: int = 6
    activation: str = "silu"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ModelConfig":
        copied = dict(values)
        copied["hidden_sizes"] = tuple(int(v) for v in copied["hidden_sizes"])
        return cls(**copied)


class CubicRootMLP(nn.Module):
    """3개 무차원 계수에서 세 복소근 ``(Re, Im)``을 회귀한다."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.input_dim != 3 or self.config.output_dim != 6:
            raise ValueError("삼차근 MLP는 input_dim=3, output_dim=6이어야 합니다.")
        if self.config.activation != "silu":
            raise ValueError("현재 지원하는 activation은 'silu'입니다.")

        dimensions = (
            self.config.input_dim,
            *self.config.hidden_sizes,
            self.config.output_dim,
        )
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            linear = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(dimensions) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.network(inputs)
        return output.reshape(*output.shape[:-1], 3, 2)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
