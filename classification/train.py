import multiprocessing
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification_dataload import ClassificationDataset
from util import train_model, initialize_model


def resnet_classification(loading_model=False,  model_name='resnet101',
                          target_category='binary', num_epochs=10, exclude_category=('Human', 'Unknown')):
    image_datasets = {
        x: ClassificationDataset(set_name=x, target_category=target_category,
                                 exclude_category=exclude_category,
                                 flip_image=True)
        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=multiprocessing.cpu_count() // 2)
                   for x in ['train', 'val']}

    using_gpu = torch.cuda.is_available()
    if using_gpu:
        print('Using GPU!')
    else:
        print('Using CPU!')
    device = torch.device("cuda:0" if using_gpu else "cpu")
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model_ft, hist_pre = initialize_model(model_name, in_features=len(dataloaders['train'].dataset.classes),
                                          device=device)

    if not loading_model:
        criterion_ft = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft, hist = train_model(model_ft, criterion_ft, optimizer_ft, exp_lr_scheduler,
                                     dataloaders=dataloaders, device=device, dataset_sizes=dataset_sizes,
                                     num_epochs=num_epochs)

        if hist_pre:
            hist['train']['epoch_loss'] = hist_pre['train']['epoch_loss'] + hist['train']['epoch_loss']
            hist['train']['epoch_acc'] = hist_pre['train']['epoch_acc'] + hist['train']['epoch_acc']
            hist['val']['epoch_loss'] = hist_pre['val']['epoch_loss'] + hist['val']['epoch_loss']
            hist['val']['epoch_acc'] = hist_pre['val']['epoch_acc'] + hist['val']['epoch_acc']

        torch.save(model_ft.state_dict(), '{}.pth'.format(model_name))
        pickle.dump(hist, open('{}.list'.format(model_name), "wb"))

    else:
        model_ft.load_state_dict(torch.load('{}.pth'.format(model_name)))
        hist = pickle.load(open('{}.list'.format(model_name), "rb"))
    return model_ft, hist


if __name__ == '__main__':
    target_category = 'species'
    exclude_category = ('Human', )
    num_epochs = 15

    model_name = f'resnet101_{target_category}_{datetime.now().strftime("%Y%m%d")}'
    print(f'Run Model: {model_name} with epochs {num_epochs}')

    model_ft, hist = resnet_classification(
        loading_model=False,
        model_name=model_name,
        num_epochs=num_epochs,
        target_category=target_category,
        exclude_category=exclude_category,
    )
