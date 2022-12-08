from torchvision import transforms

"""
Transform for MNIST
mean and std for normalization are from
https://github.com/pytorch/examples/blob/d304b0d4a20d97e3b4529cfd6429102a58e7635a/mnist/main.py#L122
"""
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
