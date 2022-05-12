"""
A class for loading neural networks in onnx format and converting to torch.

OBS:
This code only supports ONNX _model as created in the 'save' method of VeriNetNN.
Other architectures/activations/computational graphs are not considered and might fail.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx.numpy_helper

from .verinet_nn import VeriNetNN, VeriNetNNNode
from .custom_layers import Reshape

class ONNXParser:

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 filepath: str,
                 transpose_fc_weights: bool = False,
                 input_names: tuple = ('0', 'x', 'x:0', 'X_0', 'input', 'input.1', 'Input_1', 'ImageInputLayer'),
                 use_64bit: bool = False):

        """
        Args:
            filepath:
                The path of the onnx file
            transpose_fc_weights:
                If true, weights are transposed for fully-connected layers.
            input_names:
                The name of the network's input in the onnx _model.
            use_64bit:
                If true, values are stored as 64 bit.
        """

        self._model = onnx.load(filepath)

        self._input_names = input_names
        self._transpose_weights = transpose_fc_weights

        self.torch_model = None

        self._pad_nodes = []
        self._constant_nodes = []
        self._bias_nodes = []
        self._onnx_nodes = None

        self._infered_shapes = onnx.shape_inference.infer_shapes(self._model).graph.value_info
        self._tensor_type = torch.DoubleTensor if use_64bit else torch.FloatTensor

        self._node_to_idx = {}

    def to_pytorch(self) -> VeriNetNN:

        """
        Converts the self.onnx _model to a VeriNetNN(torch) _model.

        Returns:
            The VeriNetNN _model.
        """

        self._onnx_nodes = list(self._model.graph.node)
        self._onnx_nodes = self._simplify_complex_flatten(self._onnx_nodes)
        self._onnx_nodes = self._filter_special_nodes(self._onnx_nodes)
        self._onnx_nodes = self._filter_mul_nodes(self._onnx_nodes)
        verinet_nn_nodes = self._process_all_nodes(onnx_nodes=self._onnx_nodes)

        last_idx = verinet_nn_nodes[-1].idx
        verinet_nn_nodes.append(VeriNetNNNode(last_idx+1, nn.Identity(), [last_idx], []))
        verinet_nn_nodes[-2].connections_to = [last_idx+1]

        if len(self._pad_nodes) != 0:
            print(f"Model contained unprocessed padding nodes")

        if len(self._bias_nodes) != 0:
            print(f"Model contained unprocessed bias nodes")

        for node in self._constant_nodes:
            value = float(onnx.numpy_helper.to_array(node.attribute[0].t))
            if value != 1:
                print(f"Model contained unprocessed constant nodes")
                break

        return VeriNetNN(verinet_nn_nodes)

    def _simplify_complex_flatten(self, onnx_nodes: list) -> list:

        """
        Simplifies operational chains of Gather, Unsqueeze, Concat, Reshape to
        flatten when possible.

        Args:
            onnx_nodes:
                A list of the onnx nodes.
        Returns:
            A list of the simplified onnx nodes.
        """

        remove_nodes = []
        insert_nodes = []
        num_flatt = 0

        for i, shape_node in enumerate(self._onnx_nodes):
            if not shape_node.op_type == "Shape" or len(shape_node.output) != 1:
                continue

            gather_nodes = [node for node in onnx_nodes if shape_node.output[0] in node.input]
            # gather_indices = onnx.numpy_helper.to_array(node.attribute[0].t
            if (len(gather_nodes) != 1 or gather_nodes[0].attribute[0].name != "axis" or
                    gather_nodes[0].attribute[0].i != 0 or len(gather_nodes[0].output) != 1 or
                    gather_nodes[0].op_type != "Gather"):
                continue
            gather_node = gather_nodes[0]

            gather_indices_node = [node for node in onnx_nodes if gather_node.input[1] in node.output][0]
            indices = onnx.numpy_helper.to_array(gather_indices_node.attribute[0].t)
            if indices != 0:
                continue

            unsqueeze_nodes = [node for node in onnx_nodes if gather_node.output[0] in node.input]
            if (len(unsqueeze_nodes) != 1 or unsqueeze_nodes[0].attribute[0].name != "axes" or
                    unsqueeze_nodes[0].attribute[0].i != 0 or len(unsqueeze_nodes[0].output) != 1 or
                    unsqueeze_nodes[0].op_type != "Unsqueeze"):
                continue
            unsqueeze_node = unsqueeze_nodes[0]

            concat_nodes = [node for node in onnx_nodes if unsqueeze_node.output[0] in node.input]
            if (len(concat_nodes) != 1 or concat_nodes[0].attribute[0].name != "axis" or
                    concat_nodes[0].attribute[0].i != 0 or len(concat_nodes[0].output) != 1 or
                    concat_nodes[0].op_type != "Concat"):
                continue
            concat_node = concat_nodes[0]

            reshape_nodes = [node for node in onnx_nodes if concat_node.output[0] in node.input]
            if len(reshape_nodes) != 1 or reshape_nodes[0].op_type != "Reshape":
                continue
            reshape_node = reshape_nodes[0]

            input_nodes = [node for node in onnx_nodes if shape_node.input[0] in node.output]
            if len(input_nodes) != 1:
                continue
            input_node = input_nodes[0]

            insert_nodes.append((CustomNode(inputs=input_node.output, output=reshape_node.output,
                                            op_type="Flatten", name=f"CustomFlatt_{num_flatt}"), i))
            num_flatt += 1
            remove_nodes = [gather_node, unsqueeze_node, concat_node, reshape_node,
                            gather_indices_node]  # shape node is replaced below.

        for new_node, i in insert_nodes:
            self._onnx_nodes[i] = new_node

        for node in remove_nodes:
            onnx_nodes.remove(node)

        return onnx_nodes

    def _filter_special_nodes(self, onnx_nodes: list) -> list:

        """
        Filters out special nodes (constant and pad).

        Indices of other nodes input and output are adjusted accordingly.

        Args:
            onnx_nodes:
                The list of all onnx nodes
        """

        new_nodes = []

        for node in onnx_nodes:

            if node.op_type == "Constant":
                self._constant_nodes.append(node)
            elif node.op_type == "Pad":

                self._pad_nodes.append(node)

                if len(node.input) != 1:
                    raise ValueError(f"Expected input of len 1 for: {node}")

            elif node.op_type == "Add":  # Filter all add-nodes that are biases for MatMul nodes

                connected_1 = [other_node for other_node in self._onnx_nodes if node.input[0] in other_node.output]
                connected_2 = [other_node for other_node in self._onnx_nodes if node.input[1] in other_node.output]

                if len(connected_1) == 1 and connected_1[0].op_type == "MatMul" and len(connected_2) == 0:
                    self._bias_nodes.append(node)

                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)

        return new_nodes

    def _filter_mul_nodes(self, onnx_nodes: list) -> list:

        """
        Filters out all mul nodes with multiplier 1.
        """

        new_nodes = []

        for node in onnx_nodes:
            if node.op_type == "Mul":

                const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

                if len(const_nodes) != 1:
                    raise ValueError(f"Expected exactly one constant node, got {const_nodes}")

                const_node = const_nodes[0]

                if len(const_node.output) > 1:
                    raise ValueError(f"Expected constant node to have one output: {const_node}")

                atts = const_node.attribute

                if len(atts) > 1 or atts[0].name != "value":
                    raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

                value = float(onnx.numpy_helper.to_array(atts[0].t))

                if value == 1:
                    in_idx = node.input[0] if node.input[0] not in const_node.output else node.input[1]

                    for out_idx in node.output:
                        self._skip_node(in_idx, out_idx)

                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)

        return new_nodes

    def _process_all_nodes(self, onnx_nodes: list) -> list:

        """
        Loops through all onnx_nodes converting to the corresponding VeriNetNN nodes.

        Args:
            onnx_nodes:
                A list of all onnx nodes.
        Returns:
            A list of the VeriNetNN nodes.
        """

        self._node_to_idx = {}
        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity())]

        idx_num = 0
        for onnx_node in onnx_nodes:

            new_verinet_nn_nodes = self._process_node(onnx_node, idx_num + 1)

            if new_verinet_nn_nodes is not None:
                verinet_nn_nodes += new_verinet_nn_nodes
                idx_num += len(new_verinet_nn_nodes)
                self._node_to_idx[onnx_node.name] = idx_num

        self._add_output_connections(verinet_nn_nodes)
        return verinet_nn_nodes

    @staticmethod
    def _add_output_connections(verinet_nn_nodes: list):

        """
        Adds the output connections for all VeriNetNN nodes.

        Args:
            verinet_nn_nodes:
                The VeriNetNN nodes.
        """

        for i, node in enumerate(verinet_nn_nodes):
            for in_idx in node.connections_from:
                verinet_nn_nodes[in_idx].connections_to.append(i)

    def _process_node(self, node: onnx.NodeProto, idx_num: int) -> Optional[list]:

        """
        Processes an onnx node converting it to a corresponding torch node.

        Args:
            node:
                The onnx node
            idx_num:
                The current node-index number
        Returns:
                A list of corresponding VeriNetNN nodes.
        """

        if node.op_type == "Relu":
            input_connections = self._get_connections_to(node)
            if len(input_connections) > 1:
                raise ValueError(f"Found more than one input connection to {node}")
            return [VeriNetNNNode(idx_num, nn.ReLU(), input_connections)]

        elif node.op_type == "Flatten":
            input_connections = self._get_connections_to(node)
            if len(input_connections) > 1:
                raise ValueError(f"Found more than one input connection to {node}")
            return [VeriNetNNNode(idx_num, nn.Flatten(), input_connections)]

        elif node.op_type == "Gemm":
            if len(node.input) != 3:
                print(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            return self.gemm_to_verinet_nn_node(node, idx_num)

        elif node.op_type == "Conv":
            if len(node.input) != 2 and len(node.input) != 3:
                print(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            return self.conv_to_verinet_nn_node(node, idx_num)

        else:
            print(f"Node not recognised: \n{node}")
            return None

    # noinspection PyArgumentList,PyCallingNonCallable
    def gemm_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'gemm' node to a Linear verinet_nn_node.

        Args:
            node:
                The Gemm node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the torch Linear node.
        """

        [weights] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self._model.graph.initializer if t.name == node.input[2]]

        if self._transpose_weights:
            weights = weights.T

        affine = nn.Linear(weights.shape[1], weights.shape[0])
        affine.weight.data = self.convert(weights.copy())
        affine.bias.data = self.convert(bias.copy())

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, affine, input_connections)]

    # noinspection PyArgumentList,PyCallingNonCallable,PyTypeChecker
    def conv_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Conv' node to a Conv verinet_nn_node.

        Args:
            node:
                The Conv node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node Conv node.
        """

        [weights] = [onnx.numpy_helper.to_array(t).astype(float) for t in self._model.graph.initializer if
                     t.name == node.input[1]]

        if len(node.input) >= 3:
            [bias] = [onnx.numpy_helper.to_array(t).astype(float) for t in self._model.graph.initializer if
                      t.name == node.input[2]]
        else:
            bias = np.zeros(weights.shape[0])

        dilations = 1
        groups = 1
        pads = None
        strides = 1

        for att in node.attribute:
            if att.name == "dilations":
                dilations = [i for i in att.ints]
            elif att.name == "group":
                groups = att.i
            elif att.name == "pads":
                pads = [i for i in att.ints][0:2]
            elif att.name == "strides":
                strides = [i for i in att.ints]

        if pads is None:
            pads = 0

        conv = nn.Conv2d(weights.shape[1]*groups, weights.shape[0], weights.shape[2:4], stride=strides,
                         padding=pads, groups=groups, dilation=dilations)

        conv.weight.data = self.convert(weights.copy())
        conv.bias.data = self.convert(bias.copy())

        input_connections = self._get_connections_to(node)
        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, conv, input_connections)]

    def reshape_to_verinet_nn_node(self, node, idx_num: int) -> list:

        """
        Converts an onnx 'Reshape' node to a Reshape-verinet_nn_node.

        Args:
            node:
                The Reshape-node.
            idx_num:
                The index of the current node.
        Returns:
            A list containing the verinet_nn_node Reshape op.
        """

        value = None
        const_nodes = [const_node for const_node in self._constant_nodes if const_node.output[0] in node.input]

        if len(const_nodes) > 1:
            raise ValueError(f"Expected exactly at most one constant node, got {const_nodes}")

        elif len(const_nodes) == 1:
            const_node = const_nodes[0]

            if len(const_node.output) > 1:
                raise ValueError(f"Expected constant node to have one output: {const_node}")

            atts = const_node.attribute

            if len(atts) > 1 or atts[0].name != "value":
                raise ValueError(f"Expected constant a single 'value' attribute: {const_node}")

            value = tuple(onnx.numpy_helper.to_array(atts[0].t))
            self._constant_nodes.remove(const_node)

        else:
            for init in self._model.graph.initializer:
                if init.name == node.input[1]:
                    value = tuple(onnx.numpy_helper.to_array(init))

        input_connections = self._get_connections_to(node)

        if value is None:
            raise ValueError("Could not find a shape for Reshape operation")

        if len(input_connections) > 1:
            raise ValueError(f"Found more than one input connection to {node}")

        return [VeriNetNNNode(idx_num, Reshape(value), input_connections)]

    def _get_connections_to(self, node) -> list:

        """
        Returns the indices of all nodes that are connected to the given node.

        Note that any node with connections to the given node is assumed to already
        have been processed.

        Args:
            node:
                The onnx node.
        Returns:
            A list of indices of nodes connected to the given node.
        """

        input_idx = []

        for onnx_node in self._onnx_nodes:
            for output in onnx_node.output:
                if output in node.input:
                    input_idx.append(self._node_to_idx[onnx_node.name])

        for in_connection in node.input:
            if in_connection in self._input_names:
                input_idx.append(0)

        return input_idx

    def _skip_node(self, in_idx: int, out_idx: int):

        """
        Changes the indices of all nodes to bypass the given node.

        Args:
            in_idx:
                The input index.
            out_idx:
                The out index. 
        """

        for other_node in self._onnx_nodes + self._pad_nodes:
            for i in range(len(other_node.input)):
                if other_node.input[i] == out_idx:
                    other_node.input[i] = in_idx

    def convert(self, x: torch.Tensor) -> torch.Tensor:

        """
        Helper function to convert tensor to correct format.
        """

        return self._tensor_type(x)


class CustomNode:

    def __init__(self, inputs: list, output: list, op_type: str, name: str):

        self.input = inputs
        self.output = output
        self.op_type = op_type
        self.name = name
