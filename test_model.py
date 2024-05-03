import torch
import torchvision
import torchvision.transforms as transforms

from LeNet import Net

# load the saved model
PATH = "./model/model.pth"
net = Net()  # re-create the model architecture
net.load_state_dict(torch.load(PATH))
net.eval()  # set the model to evaluation mode

# load test data
testtransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2467, 0.2429, 0.2616))])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=testtransform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))