import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image
from imutils.paths import list_images
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import copy, random

import os


class OneRotationTransform:
    """Rotate by one given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class SolarDataset(Dataset):
    def __init__(self, images, labels, transform, transform2=None, inx_aug=None):
        self.images = images
        self.labels = torch.from_numpy(labels)
        self.transform = transform
        self.transform2 = transform2
        self.inx_aug = inx_aug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(cv.cvtColor(self.images[idx], cv.COLOR_BGR2RGB))
        label = self.labels[idx]
        if self.transform2:
            if label in self.inx_aug:
                image = self.transform(image)
            else:
                image = self.transform2(image)
        else:
            image = self.transform(image)

        return image, label


class PredDataset(Dataset):
    # Only used for prediction
    # No labels
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(cv.cvtColor(self.images[idx], cv.COLOR_BGR2RGB))
        image = self.transform(image)

        return image


def load_data(im_dir, store_name=False):
    """load the images and labels

    Parameters
    ----------
    im_dir: str
    Folder name of images. Folder should be like: 'train/category_i/image.png' and use 'train' as im_dir

    store_name: bool
    If True, store the name of solar modules

    Returns
    -------
    images, labels, names: list
    List of images, labels and names if store_name==True
    """
    images = []
    labels = []
    if store_name:
        names = []

        for im_path in tqdm(list(list_images(im_dir))):
            images.append(cv.imread(im_path))
            labels.append(im_path.split('/')[-2])
            names.append(os.path.splitext(os.path.split(im_path)[-1])[0])

        return images, labels, names
    else:
        for im_path in tqdm(list(list_images(im_dir))):
            images.append(cv.imread(im_path))
            labels.append(im_path.split('/')[-2])

        return images, labels


def train_model(model, trainloader, valloader, solar_train, solar_val, criterion, optimizer, scheduler, device, num_epochs=25):
    """Train a neural network using pytorch. Different loss functions are compared

    Parameters
    ----------
    model:
    Pytorch model

    trainloader, valloader: 
    Pytorch dataloader

    solar_train, solar_val:
    Pytorch dataset

    criterion:
    Loss function

    optimizer:
    optimizer

    scheduler:
    Learning rate scheduler

    num_epochs: int
    Number of epochs

    device: torch.device()

    Returns
    -------
    weights_dict: dict
    The model weights on optimal epochs. Selection can be based on optimal accuracy, loss or f1 score of validation set

    loss_acc: dict
    Record of loss, accuracy and f1 score during training and validation process
    """
    loss_acc = {
        'train': {"loss": [], "acc": [], "f1_macro": []},
        'val': {"loss": [], "acc": [], "f1_macro": []}
    }

    best_acc_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1e6
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
                dataset_size = len(solar_train)
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valloader
                dataset_size = len(solar_val)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'val':  # collect all the labels in one epoch
                preds_val = []
                labels_val = []
            elif phase == 'train':
                preds_train = []
                labels_train = []
                # prob_val = []
                # m = nn.Softmax(dim=1)
            for inputs, labels in tqdm(dataloader):
                # collect val label to compute f1
                if phase == 'val':
                    labels_val = np.append(labels_val, labels.numpy())
                elif phase == 'train':
                    labels_train = np.append(labels_train, labels.numpy())
                inputs = inputs.to(device)
                labels = labels.long()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        preds_train = np.append(preds_train, preds.cpu().numpy())
                        loss.backward()
                        optimizer.step()

                    # record val prediction to compute f1
                    if phase == "val":
                        # prob_val= np.append(prob_val, m(outputs.cpu()).numpy())
                        preds_val = np.append(preds_val, preds.cpu().numpy())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            loss_acc[phase]['loss'].append(epoch_loss)
            loss_acc[phase]['acc'].append(float(epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # compute f1
            if phase == 'val':
                # labels_val_binary = label_binarize(labels_val, classes = le.transform(le.classes_))
                # prob_val = np.array(prob_val)
                # epoch_prc_micro = average_precision_score(labels_val_binary.ravel(), prob_val.ravel(), average='micro')
                epoch_f1 = f1_score(labels_val, preds_val, average='macro')
                loss_acc[phase]['f1_macro'].append(epoch_f1)
                print('F1: {:4f}'.format(epoch_f1))
            elif phase == 'train':
                epoch_f1 = f1_score(labels_train, preds_train, average='macro')
                loss_acc[phase]['f1_macro'].append(epoch_f1)
                print('F1: {:4f}'.format(epoch_f1))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_f1_model_wts = copy.deepcopy(model.state_dict())

        print()

    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val F1: {:4f}'.format(best_f1))

    return {'best_acc': best_acc_model_wts, 'best_loss': best_loss_model_wts, 'best_f1': best_f1_model_wts}, loss_acc


def training_plot(epochs, loss_accuracy, metric='loss'):
    """Plot the metrics change during training/val process

    Parameters
    ----------
    epochs: int
    Number of epochs

    loss_accuracy: dict
    Record of loss, accuracy and f1 score during training and validation process

    metric: str
    Metrics to show
    """
    plt.plot(range(epochs), loss_accuracy['train'][metric], label=f'train_{metric}')
    plt.plot(range(epochs), loss_accuracy['val'][metric], label=f'val_{metric}')
    plt.legend()


def predict_test(testloader, model_fit, device):
    """Make prediction on the testing set

    Parameters
    ----------
    testloader: torch.utils.data.dataloader.DataLoader
    Dataloader of testing set

    model_fit: Pytorch model
    Machine learning model used to make prediction

    device: torch.device
    Device to work on

    Returns
    -------
    pred_test: list
    Predicted labels

    prob_test: list
    Probability of each category
    """
    pred_test = []
    prob_test = []

    m = nn.Softmax(dim=1)
    with torch.no_grad():
        for input in tqdm(testloader):
            input = input.to(device)

            output = model_fit(input)
            prob_test += m(output.cpu()).tolist()
            _, pred = torch.max(output, 1)
            pred_test += pred.cpu().tolist()
    return pred_test, prob_test


