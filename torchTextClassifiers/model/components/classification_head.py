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
            self.net = net
            
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
            else:  # if not Sequential, it is a Linear
                self.input_dim = net.in_features
                self.num_classes = net.out_features

        else:
            assert (
                input_dim is not None and num_classes is not None
            ), "Either net or both input_dim and num_classes must be provided."
            self.net = nn.Linear(input_dim, num_classes)
            self.input_dim = input_dim
            self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
