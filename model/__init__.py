"""Model

This module contains the model and some basic functions for the model.

Author:  Patr1ick
Date:    04.06.2022
Version: 1.0.0
License: MIT
"""
import string
from PIL import Image
import cv2
import torch
import torchvision
import torchvision.transforms as tt


def saveImage(img: Image, name: string = "drawing.png"):
    """Saves an image with the given name.

    The image will be saved in the same directory as the programm was started.

    Parameters
    ----------
    img: Image
        The Image that should be saved
    name: string
        The name under which the image will be saved (default is "drawing.png")

    Raises
    ------
    ValueError
        If the passed name is none or an empty string is given.

    TestCase
    --------
    Give an image as the input and check if a file is created in the directory.

    """
    if name is None or name == "":
        raise ValueError("The name is none. Please give a correct name")
    img.save(name)


def convertImageToTensor(path: string = "drawing.png") -> torch.Tensor:
    """Convert a image to a PyTorch Tensor.

    The image will be read from the given path and converted to an 28x28 image with only 1 Color channel (Black and White). After that the image is converted to a PyTorch Tensor and the value of that Tensor is normalized.

    Parameters
    ----------
    path: string
        The path to the image (default is "drawing.png")

    Returns
    -------
    torch.Tensor
        The PyTorch Tensor of the image

    TestCase
    --------
    Run this method with some images and check if the Tensors which are created by this method are correct
    """
    # Read the image from the given path
    img = cv2.imread(path)
    # Resize image to 28x28
    img = cv2.resize(img, (28, 28))
    # Convert image to a Black & White color scheme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the image to a tensor and normalize the values
    transform = torchvision.transforms.Compose([
        tt.ToTensor(),
        tt.Normalize((0.5), (0.5))
    ])
    # Return the Tensor
    return transform(img)
