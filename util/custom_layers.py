
"""
Contains custom torch layers.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


import torch
import torch.nn as nn


class Reshape(nn.Module):

    """
    Reshapes the input to the given shape.
    """

    def __init__(self, shape: tuple):
        super(Reshape, self).__init__()
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def forward(self, x: torch.Tensor):
        return x.reshape(self._shape)
