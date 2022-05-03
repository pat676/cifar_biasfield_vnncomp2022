"""
The affine transform network.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from .verinet_nn import VeriNetNN, VeriNetNNNode
from .custom_layers import Reshape


# noinspection PyTypeChecker
class BiasFieldNN(VeriNetNN):

    """
    The torch _model used to encode bias-field transforms for verification.

    Given a _model N, the bias-field  Tensor B and an input tensor x, this _model
    creates a new network M that takes as input C variables.

    The matrix B represents a bias field b(i, j) = b_0(i, j) + b_1(i, j) ..., b_c(i, j)
    as B[i, j, c] = b_c(i, j).

    The _model M computes M = N(B'x) where B' = B.reshape(-1, C)
    """

    def __init__(self,
                 model: VeriNetNN,
                 x: torch.Tensor,
                 bias_field: torch.Tensor,
                 add_x: bool = True,
                 mean: torch.Tensor = None,
                 std: torch.Tensor = None,
                 use_gpu: bool = False):

        """
        Args:
            model:
                The original _model.
            x:
                The input vector.
            bias_field:
                A 3-dimensional tensor of size KxLxC representing the bias field,
                where KxL is the size of the input image and C the number of terms
                in the bias field.
                The tensor B represents a bias field b(i, j) = b_0(i, j) + b_1(i, j) +
                ..., b_c(i, j) as B[i, j, c] = b_c(i, j).
            add_x:
                If add x is true, x is added to the bias field, so the input
                to the original network becomes x' = B + x. A multiplicative bias
                field can be implemented by pre-calculating B' = B*x and passing
                B' with add_x = False.
            mean:
                The mean values used for normalising (one per channel).
            std:
                The standard deviation for normalising (one per channel).
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        nodes = [node.copy() for node in model.nodes[1:]]
        nodes = self._adjust_connected_nodes_idx(nodes)

        input_node = VeriNetNNNode(0, nn.Identity(), None, [1])
        bias_field_node = self.get_bias_field_node(x, bias_field, add_x)
        reshape_node = VeriNetNNNode(2, Reshape(x.shape), [1], [idx + 2 for idx in model.nodes[0].connections_to])

        nodes = [input_node, bias_field_node, reshape_node] + nodes

        self._normalise_layer(nodes[1].op, mean, std, x)

        super().__init__(nodes, use_gpu=use_gpu)

    @staticmethod
    def get_bias_field_node(x: torch.Tensor, bias_field: torch.Tensor, add_x: bool) -> torch.Tensor:

        """
        Returns a fully connected node encoded the bias-field additive transformation.

        Args:
            x:
                The input image.
            bias_field:
                A 3-dimensional tensor of size KxLxC representing the bias field,
                where KxL is the size of the input image and C the number of terms
                in the bias field.
                The tensor B represents a bias field b(i, j) = b_0(i, j) + b_1(i, j) +
                ..., b_c(i, j) as B[i, j, c] = b_c(i, j).
            add_x:
                If add x is true, x is added to the bias field, so the input
                to the original network becomes x' = B + x. A multiplicative bias
                field can be implemented by pre-calculating B' = B*x and passing
                B' with add_x = False.
        Returns:
            The FC-node encoding the transformation.
        """

        x = x.view(-1)
        layer = torch.nn.Linear(int(np.prod(bias_field.shape[1:])), x.shape[0])

        layer.weight.data = bias_field.clone()

        if add_x:
            layer.bias.data = x.clone()
        else:
            layer.bias.data = torch.zeros_like(x)

        return VeriNetNNNode(1, layer, [0], [2])

    @staticmethod
    def _adjust_connected_nodes_idx(nodes: list, amount: int = 2):

        """
        Adjusts the index of connection by amount for all nodes.

        Args:
            nodes:
                The nodes of which to adjust the index.
            amount:
                The amount by which to change the index.
        Returns:
            The adjusted nodes
        """

        for node in nodes:
            for i in range(len(node.connections_from)):
                node.connections_from[i] += amount
            for i in range(len(node.connections_to)):
                node.connections_to[i] += amount
            node.idx += amount

        return nodes

    @staticmethod
    def _normalise_layer(layer: torch.nn, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor):

        """
        Encodes normalisation values into the node.

        Args:
            layer:
                The node
            mean:
                The mean
            std:
                The std
            x:
                The input tensor that should be normalised.
        """

        if mean is None or std is None:
            return

        if isinstance(layer, nn.Linear):

            num_channels = x.shape[1]
            pixels_pr_channel = np.prod(list(x.shape[2:]))

            if num_channels != mean.shape[0] or num_channels != std.shape[0]:
                raise ValueError(f"Wrong normalisation dim got mean: {mean.shape[0]}, "
                                 f"std: {std.shape[0]}, expected: {num_channels}")

            for channel in range(num_channels):
                layer.weight.data[channel*pixels_pr_channel: (channel + 1)*pixels_pr_channel, :] /= std[channel]
                layer.bias.data[channel*pixels_pr_channel: (channel + 1)*pixels_pr_channel] -= mean[channel]
                layer.bias.data[channel * pixels_pr_channel: (channel + 1) * pixels_pr_channel] /= std[channel]

        else:
            raise ValueError(f"Normalisation not implemented for node of type: {layer}")


# Polynomial regression
def get_matrix(x, y, order):
    """
    Produces the bias field matrix for coordinates x,y (as created by get_idx).

    Args:
        x:
            The x meshgrid
        y:
            The y meshgrid.
        order:
            The order of the polynomial.
    Returns:
         The bias field matrix.
    """

    # noinspection PyShadowingNames
    def get_list(order):
        if order == 1:
            return [x, x * y, y, np.ones_like(x)]
        elif order == 2:
            return [x ** 2, x ** 2 * y, x ** 2 * y ** 2, x * y ** 2, y ** 2] + get_list(1)
        elif order == 3:
            return [x ** 3, x ** 3 * y, x ** 3 * y ** 2, x ** 3 * y ** 3, x ** 2 * y ** 3, x * y ** 3,
                    y ** 3] + get_list(2)
        elif order == 4:
            return [x ** 4, x ** 4 * y, x ** 4 * y ** 2, x ** 4 * y ** 3, x ** 4 * y ** 4, x ** 3 * y ** 4,
                    x ** 2 * y ** 4, x * y ** 4, y ** 4] + \
                   get_list(3)
        else:
            raise RuntimeError("polynomial for order >= 5 not supported yet!")

    m = get_list(order)

    return np.array(m).T


def get_idx(num_points_x: int, num_points_y: Optional[int] = None):
    """
    Produces the coordinate meshgrid for x and y.

    Args:
        num_points_x:
            The number of points in the x direction.
        num_points_y:
            The number of points in the y direction
    Returns:
         The x,y meshgrids.
    """

    if num_points_y is None:
        num_points_y = num_points_x

    x = np.linspace(0, 1, num_points_x)
    y = np.linspace(0, 1, num_points_y)
    x, y = np.meshgrid(x, y)

    return x, y
