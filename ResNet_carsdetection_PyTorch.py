!pip freeze | grep "torch"
!pip install --upgrade torch torchvision
!pip freeze | grep "torch"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy
from copy import copy
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000 

#Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Mean and Standard Deviation function
def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


#Data Preparation - full dataset in order to calculate mean and std
full_transform = transforms.Compose([
            transforms.Resize(255),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor() 
        ])

DataPath = "/kaggle/input/cars-types/data"
full_dataset = ImageFolder(root = DataPath, transform=full_transform)

full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=128,
                                         shuffle=True, num_workers=2)

mean, std = online_mean_and_sd(full_loader)

#Train and Validation split
train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size

train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

train_dataset.dataset = copy(full_dataset)

train_dataset.dataset.transform = transforms.Compose(
    [transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
     transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)])

validation_dataset.dataset.transform = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)])


#DataLoaders - training and validation
%config Completer.use_jedi = False


batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2,pin_memory = "true")

val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory = "true")


#ResidualNetwork elements 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7):
        super(ResNet, self).__init__()

        self.in_channels = 96
        self.conv = conv3x3(3, 96)
        self.bn = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 96, layers[0], 2)
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ResNet(block, layers[blocknumbers])
net = ResNet(ResidualBlock, [2, 4, 6, 4]).to(device)


#Moving net to gpu
net.to(device)

#Optimization configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Optimization scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

#Training and Validation
for epoch in range(100):  

    # Training part
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels] #moved to gpu
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics => sum of losess
        running_loss += loss.item()
    scheduler.step()

    # Validation part
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            #moved to gpu
            images, labels = data[0].to(device), data[1].to(device)
            #calculate outputs by running images through the network
            outputs = net(images)
            #the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Loss divided by 500 in this case, because training data contains 50000 imgs and batch_size = 16
    print('Epoch nr. {} \nLoss {}\n'.format(epoch, running_loss/3125))   
    running_loss = 0.0   
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
