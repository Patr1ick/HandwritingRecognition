# Handwriting Recognition

This is the project for the elective module "Python" at DHBW Stuttgart.
![Screenshot of the application](/img/Screenshot.png)

## Run the application

### Requirements

It is recommended to run the application with Python 3.9. With pip you can download all the libraries you need for the program.

```bash
pip install -r requirements.txt
```

**You also need the `model.pth` in the same folder where you start the application. This file contains the pre-trained model which is loaded by the programm to generate the results.**  
This pre-trained model has an accurency of 98.39%.

### Run the application

After that you can simply call the following command:

```bash
python main.py
```

### Train the neural network yourself

To train the neural network again or to change for example hyperparameters. You can simply run the [`TrainModel.ipynb`](/TrainModel.ipynb) Notebook. At the end of the notebook it will save the trained model as `model.pth`.

## Information

This programm uses the balanced EMNIST Dataset which means that there are only 47 classes instead of the full 62 classes. This makes it easier to train the neural net, because for each class is the same number of images and the net will not be biased towards a character or digit that is overrepresented. In the diagram below you can see which character are seen as the same (e.g. i=I, j=J).
![Classes of the EMNIST Dataset](/img/EMNIST.png)
[Source](https://arxiv.org/pdf/1702.05373.pdf)
### Dataset

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from [http://arxiv.org/abs/1702.05373](http://arxiv.org/abs/1702.05373)

### Neural Network

The PyTorch library was used for the neural network. The neural network is a simple convolutional neural network with 2 Convolutional layers with the LeakyReLU activation function and each has a Max-Pooling Layer with a kernel_size of 2. After the last Convolutional Layer the size of the input is 3x3 with 64 channels. This tensor will be reshape that it fit into the last second Linear Layers which will produce the ouput. The ouput tensor is a array with the ouput_size (here 47 classes). The probabilities are generated with a softmax function by the app.

## Used tutorials

-   [PyTorch: Save and Load the Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)
-   [PyTorch: Training a Classifier (for CIFAR10 dataset)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## License

This programm is licensed under the [MIT-License](/LICENSE.md).
