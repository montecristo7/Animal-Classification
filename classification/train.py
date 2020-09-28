import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

from classification_dataload import ClassificationDataset
from util import train_model


def resnet_classification(loading_model=False):
    image_datasets = {x: ClassificationDataset(set_name=x, root_dir='images')
                      for x in ['train', 'validation']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dataloaders['train'].dataset.classes))
    model_ft = model_ft.to(device)

    if not loading_model:
        criterion_ft = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft, hist = train_model(model_ft, criterion_ft, optimizer_ft, exp_lr_scheduler,
                                     dataloaders=dataloaders, device=device, dataset_sizes=dataset_sizes,
                                     num_epochs=100)

        torch.save(model_ft.state_dict(), 'resnet101.pth')
        pickle.dump(hist, open('resnet101.list', "wb"))

    else:
        model_ft.load(torch.load('resnet101.pth'))
        hist = pickle.load(open('resnet101.list', "rb"))
    return model_ft, hist


if __name__ == '__main__':
    resnet_classification(loading_model=False)
