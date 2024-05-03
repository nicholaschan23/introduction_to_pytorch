import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset


def data_stats(trainset):
    # stack all train images together into a tensor of shape #(50000, 3, 32, 32)
    x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

    # get the mean and standard deviation of each rgb channel
    mean = torch.mean(x, dim=(0, 2, 3))
    print(mean)
    # tensor([0.4914, 0.4822, 0.4465])
    std = torch.std(x, dim=(0, 2, 3))
    print(std)
    # tensor([0.2470, 0.2435, 0.2616])


transform = transforms.Compose([transforms.ToTensor()])
print("Training dataset:")
data_stats(torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform))
print("Test dataset:")
data_stats(torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform))
