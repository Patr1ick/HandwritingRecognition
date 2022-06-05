import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """
    Class for the neural network to recognize the image. In this project this class is only used as a template to load the parameters of an pre-trained network (See the notebook TrainModel.ipynb).

    Attributes
    ----------
    conv1: Sequetial
        The first layer using a 2D convolution layer, the LeakyReLU activation function and a Pooling Layer.
    conv2: Sequential
        The second layer using a 2D convolution layer, the LeakyReLU activation function and a Pooling Layer.
    conv3: Sequential
        The third layer using a 2D convolution layer, the LeakyReLU activation function and a Pooling Layer.
    out: Sequential
        The output layer which produces the output using first a Linear Layer, the LeakyReLU activation function, a Dropout Layer with 40% Dropout and the last Linear Layer.

    """

    def __init__(self, output_size: int, device="cpu"):
        """
        Parameters
        ----------
        output_size: int
            The output size of the neural network which is the number of classes that should be classified
        device: string
            The device the neural network should be running. If for example CUDA is supported, the network could run on the GPU. (default is cpu)
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 36, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        ).to(device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(36, 72, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        ).to(device)  # New size 7x7
        self.out = nn.Sequential(
            nn.Linear(72*5*5, 200),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(200, output_size)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward method of the neural network.

        This will pass an input through the nerual network and returns the result of the network

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor:
            The ouput of the neural network

        """
        x = self.conv1(x)
        x = self.conv2(x)
        # Reshape the Tensor that it will fit into the Linear Layer
        out = x.view(x.shape[0], -1)
        return self.out(out)
