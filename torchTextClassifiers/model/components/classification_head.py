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
        """
        Classification head for text classification tasks.
        It is a nn.Module that can either be a simple Linear layer or a custom neural network module.

        Args:
            input_dim (int, optional): Dimension of the input features. Required if net is not provided.
            num_classes (int, optional): Number of output classes. Required if net is not provided.
            net (nn.Module, optional): Custom neural network module to be used as the classification head.
                If provided, input_dim and num_classes are inferred from this module.
                Should be either an nn.Sequential with first and last layers being Linears or nn.Linear.
        """
        super().__init__()
        if net is not None:
            # --- Custom net should either be a Sequential or a Linear ---
            if not (isinstance(net, nn.Sequential) or isinstance(net, nn.Linear)):
                raise ValueError("net must be an nn.Sequential when provided.")

            # --- If Sequential, Check first and last layers are Linear ---

            if isinstance(net, nn.Sequential):
                first = net[0]
                last = net[-1]

                if not isinstance(first, nn.Linear):
                    raise TypeError(f"First layer must be nn.Linear, got {type(first).__name__}.")

                if not isinstance(last, nn.Linear):
                    raise TypeError(f"Last layer must be nn.Linear, got {type(last).__name__}.")

                # --- Extract features ---
                self.input_dim = first.in_features
                self.num_classes = last.out_features
                self.net = net
            else:  # if not Sequential, it is a Linear
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
