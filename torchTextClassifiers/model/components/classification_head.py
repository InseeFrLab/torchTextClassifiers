from typing import Optional

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        net: Optional[nn.Module] = None,
    ):
        super().__init__()
        if net is not None:
            self.net = net
            self.input_dim = net.in_features
            self.num_classes = net.out_features
        else:
            assert (
                input_dim is not None and num_classes is not None
            ), "Either net or both input_dim and num_classes must be provided."
            self.net = nn.Linear(input_dim, num_classes)
            self.input_dim, self.num_classes = self._get_linear_input_output_dims(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _get_linear_input_output_dims(module: nn.Module):
        """
        Returns (input_dim, output_dim) for any module containing Linear layers.
        Works for Linear, Sequential, or nested models.
        """
        # Collect all Linear layers recursively
        linears = [m for m in module.modules() if isinstance(m, nn.Linear)]

        if not linears:
            raise ValueError("No Linear layers found in the given module.")

        input_dim = linears[0].in_features
        output_dim = linears[-1].out_features
        return input_dim, output_dim
