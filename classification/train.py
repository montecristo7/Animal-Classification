import multiprocessing
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification_dataload import ClassificationDataset
from util import train_model, initialize_model


def resnet_classification(loading_model=False, image_root='image', model_name='resnet101',
                          target_category='species_binary', num_epochs=10):
    image_datasets = {x: ClassificationDataset(set_name=x, root_dir=image_root, target_category=target_category)
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=6,
                                                  shuffle=True, num_workers=multiprocessing.cpu_count() // 2)
                   for x in ['train', 'val']}
    # class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model_ft = initialize_model(model_name)

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
                                     num_epochs=num_epochs)

        torch.save(model_ft.state_dict(), '{}.pth'.format(model_name))
        pickle.dump(hist, open('{}.list'.format(model_name), "wb"))

    else:
        model_ft.load_state_dict(torch.load('{}.pth'.format(model_name)))
        hist = pickle.load(open('{}.list'.format(model_name), "rb"))
    return model_ft, hist


if __name__ == '__main__':
    model_ft, hist = resnet_classification(
        loading_model=False,
        model_name='resnet101_binary_300',
        num_epochs=20,
        target_category='species_binary',
    )
