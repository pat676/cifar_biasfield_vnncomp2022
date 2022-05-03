
import os
import sys

import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from util.create_bias_field_net import create_biasfield_models


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):

    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """

    with open(spec_path, "w") as f:

        f.write(f"; Cifar bias field property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python generate_properties.py <SEED>")
        exit()

    torch.random.manual_seed(int(sys.argv[1]))

    num_props = 72
    epsilon = 0.05
    timeout = 300
    num_coeff = 16

    bounds = torch.zeros((16, 2))

    bounds[-1, :] += 1

    bounds[:-1, 0] -= epsilon / (num_coeff - 1)
    bounds[:-1, 1] += epsilon / (num_coeff - 1)
    bounds[-1, 0] -= epsilon
    bounds[-1, 1] += epsilon

    prop_dir = "vnnlib_properties"
    model_dir = "onnx"

    if not os.path.isdir(prop_dir):
        os.mkdir(prop_dir)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    nets_labels = create_biasfield_models(network_path="onnx/cifar_base.onnx", order=3, num_models=num_props)
    dummy_in = torch.zeros((1, 16))

    with open("instances.csv", "w", buffering=1) as instances_file:

        for spec_num, (bias_field_net, label) in enumerate(nets_labels):

            spec_path = os.path.join(prop_dir, f"prop_{spec_num}.vnnlib")
            save_vnnlib(bounds, label, spec_path)

            model_path = os.path.join(model_dir, f"cifar_bias_field_{spec_num}.onnx")
            bias_field_net.save(dummy_in.to(bias_field_net.device), model_path)

            instances_file.write(f"{model_path},{spec_path},{timeout}\n")
