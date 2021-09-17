import os
import cv2
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.transforms as T
from torchvision.utils import make_grid
import torchvision.models as models

from sklearn.model_selection import train_test_split

from PIL import Image

import matplotlib.pyplot as plt

from zipfile import ZipFile


### Import train and test data
trainDataPath = "/kaggle/input/dogs-vs-cats/train.zip"
testDataPath = "/kaggle/input/dogs-vs-cats/test1.zip"
ZipFile(trainDataPath, mode="r").extractall()
ZipFile(testDataPath, mode="r").extractall()

TRAIN_PATH = './train/'
TEST_PATH = './test1/'

path = "dog_vs_cat"


### Checking Filenames
imgs = os.listdir(TRAIN_PATH)
test_imgs = os.listdir(TEST_PATH)

print(imgs[:5])
print(test_imgs[:5])


def get_train_transform():
    return T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CatDogDataset(Dataset):

    def __init__(self, imgs, class_to_int, mode="train", transforms=None):

        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, idx):

        image_name = self.imgs[idx]

        ### Reading, converting and normalizing image
        img = cv2.imread(TRAIN_PATH + image_name, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.
        img = Image.open(TRAIN_PATH + image_name)
        img = img.resize((224, 224))

        if self.mode == "train" or self.mode == "val":

            ### Preparing class label
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype=torch.int64)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label

        elif self.mode == "test":

            ### Apply Transforms on image
            img = self.transforms(img)

            return img

    def __len__(self):
        return len(self.imgs)


### Data spliting
train_imgs, val_imgs = train_test_split(imgs, test_size = 0.25)


### Datasets creating
class_to_int = {"dog":0, "cat":1}
train_dataset = CatDogDataset(train_imgs, class_to_int, mode="train", transforms = get_train_transform())
val_dataset = CatDogDataset(val_imgs, class_to_int, mode="val", transforms = get_val_transform())
test_dataset = CatDogDataset(test_imgs, class_to_int, mode="test", transforms = get_val_transform())

train_data_loader = DataLoader(
    dataset=train_dataset,
    num_workers=4,
    batch_size=16,
    shuffle=True
)

val_data_loader = DataLoader(
    dataset=val_dataset,
    num_workers=4,
    batch_size=16,
    shuffle=True
)

test_data_loader = DataLoader(
    dataset=test_dataset,
    num_workers=4,
    batch_size=16,
    shuffle=True
)

dataloaders = {"train": train_data_loader, "val": val_data_loader, "test": test_data_loader}


### Example
for images, labels in train_data_loader:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, 4).permute(1, 2, 0))
    print(labels)
    break

import copy



### Train_model function
def train_model(model, criterion, optimizer, scheduler, num_epochs=2, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        ### Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  ### Set model to training mode
            else:
                model.eval()  ### Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            ### Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                ### zero the parameter gradients
                optimizer.zero_grad()

                ### forward
                ### track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    ### Get model outputs and calculate loss
                    ### Special case for inception because in training it has an auxiliary output. In train
                    ### mode we calculate the loss by summing the final output and the auxiliary output
                    ### but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        ### From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    ### backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                ### statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            ### deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    ### load best model weights
    model.load_state_dict(best_model_wts)
    return model





model_ft = models.inception_v3(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

### Handle the auxilary net
num_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 2)
### Handle the primary net plus adding top layers
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2048),nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.3),nn.Linear(2048,2))

### Model structure
#print(model_ft)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer_conv = optim.SGD(params_to_update, lr=0.05, momentum=0.9)

criterion = nn.CrossEntropyLoss()
### Decay LR by a factor of 0.1 every epoch
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1)

### Training
model_ft = train_model(model_ft, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=30, is_inception=True)