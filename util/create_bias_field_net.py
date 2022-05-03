
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as trans

from .bias_field_nn import BiasFieldNN, get_idx, get_matrix
from .onnx_parser import ONNXParser


def create_biasfield_models(network_path: str, num_models: int = 1, order=3):

    """
    Creates networks augmented with the bias field transformation.

    ArgsL
        network_path:
            The path of the original onnx model.
        num_models:
            The number of biasfield models to create.
        order:
            The order of the bias field polynomial.
    """

    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])

    parser = ONNXParser(network_path)
    model = parser.to_pytorch()
    model.set_device(True)
    model.eval()

    print(f"Network accuracy: {100 * calc_accuracy(model, mean, std):.2f}%")

    images, labels = get_cifar_test_data(10000)
    bias_field_models = []

    for img_num in range(len(images)):

        img, label = images[img_num:img_num+1], labels[img_num]
        img_norm = ((img - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).to(model.device)

        res = model(img_norm)[0]
        pred = int(torch.argmax(res, dim=1))

        if pred != label:
            continue

        num_channels = img.shape[1]
        dim_y, dim_x = img.shape[-2:]
        x_coords, y_coords = get_idx(dim_x, dim_y)
        np_bias_field_terms = get_matrix(x_coords.flatten(), y_coords.flatten(), order=order)
        np_bias_field_terms = np.repeat(np_bias_field_terms, num_channels, axis=0)
        np_bias_field_terms = img.numpy().flatten().reshape(-1, 1) * np_bias_field_terms

        bias_field_terms = torch.Tensor(np_bias_field_terms)
        bias_field_model = BiasFieldNN(model, img, bias_field_terms, add_x=False, use_gpu=True, mean=mean, std=std)
        bias_field_model.eval()

        bias_field_models.append((bias_field_model, label))
        if len(bias_field_models) >= num_models:
            break

        tmp = torch.zeros((1, 16)).to(img_norm.device)
        tmp[0, -1] = 1

    return bias_field_models


def calc_accuracy(model: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor):

    """
    Calculates the accuracy on CIFAR10 for the given model.

    ArgsL
        model:
            The model for which to calculate accuracy.
        mean:
            The mean used for normalising the input.
        std:
            The standard deviation used for normalising the input.
    Returns:
        The accuracy.
    """

    correct, total = 0, 0
    images, labels = get_cifar_test_data(10000)

    for img_num in range(len(images)):

        img, label = images[img_num:img_num + 1], labels[img_num]
        img_norm = ((img - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).to(model.device)

        res = model(img_norm)[0]
        pred = int(torch.argmax(res, dim=1))

        total += 1
        if pred == label:
            correct += 1

    return correct / total


def get_cifar_test_data(num_data: int = 100) -> tuple:

    """
    Returns the first num_data points from the cifar test-set.

    Args:
        num_data:
            The number of datapoints.
    Returns:
        The tuple (images, labels)
    """

    cifar10_test = dset.CIFAR10("./cifar_full/", train=False, download=True, transform=trans.ToTensor())
    loader_test = DataLoader(cifar10_test, sampler=sampler.SubsetRandomSampler(range(num_data)), batch_size=num_data)

    return next(iter(loader_test))
