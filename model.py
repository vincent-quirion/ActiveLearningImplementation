from torch import nn
from torchvision.models import resnet18


def resnet18MNIST():
    # MNIST has 10 classes for digits from 0 to 9
    model = resnet18(num_classes=10)
    # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) in original architecture
    # changed to work with MNIST (grayscale instead of RGB)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model
