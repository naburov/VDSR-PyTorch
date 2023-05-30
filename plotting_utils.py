import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_minmax_normalized(tensor: torch.Tensor):
    np_tensor = tensor.numpy()
    if len(np_tensor.shape) == 4:
        np_tensor = np_tensor[0, ...]

    np_tensor = np.abs(np.transpose(np_tensor, axes=[1, 2, 0]))
    np_tensor = (np_tensor - np.amin(np_tensor)) / (np.amax(np_tensor) - np.amin(np_tensor))

    plt.imshow(np_tensor[..., 0])
    plt.show()


def plot_image(tensor: torch.Tensor):
    np_tensor = tensor.numpy()
    if len(np_tensor.shape) == 4:
        np_tensor = np_tensor[0, ...]

    np_tensor = np.abs(np.transpose(np_tensor, axes=[1, 2, 0]))

    plt.imshow(np_tensor[..., 0])
    plt.show()
