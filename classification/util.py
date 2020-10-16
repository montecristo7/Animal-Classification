import copy
import pathlib
import pickle
import time

import numpy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from torchvision import models


def imshow(inps, title=None, savefig=None):
    """Imshow for Tensor."""
    if len(inps.shape) > 3:
        inps = numpy.concatenate([inps[i, :, :, :].numpy() for i in range(inps.shape[0])], axis=2).transpose((1, 2, 0))
    else:
        inps = inps.numpy().transpose((1, 2, 0))
    fig = plt.figure(figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')

    plt.imshow(inps)
    if title is not None:
        plt.title(', '.join(title))

    if savefig:
        fig1 = plt.gcf()
        fig1.savefig(savefig, dpi=100, facecolor='w', edgecolor='k')
    plt.pause(0.001)  # pause a bit so that plots are updated


def acc_curve(hist, savefig=None):
    train_acc = [float(i) for i in hist['train']['epoch_acc']]
    val_acc = [float(i) for i in hist['val']['epoch_acc']]

    plt.figure(figsize=(15, 12))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Acc')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Valididation Acc')
    plt.ylim((0.5, 1))

    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title('Accuracy Curve', fontsize=20)
    # show the legend
    plt.legend(fontsize=15)
    if savefig:
        fig1 = plt.gcf()
        plt.savefig(pathlib.Path('savefig') / '{}.png'.format(savefig), dpi=100)
    plt.show()


def loss_curve(hist, savefig=None):
    train_loss = [float(i) for i in hist['train']['epoch_loss']]
    val_loss = [float(i) for i in hist['val']['epoch_loss']]

    plt.figure(figsize=(15, 12))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Valididation Loss')
    plt.ylim(0)

    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title('Loss Curve', fontsize=20)
    # show the legend
    plt.legend(fontsize=15)
    if savefig:
        fig1 = plt.gcf()
        plt.savefig(pathlib.Path('savefig') / '{}.png'.format(savefig), dpi=100)
    plt.show()


def roc(lr_probs, testy, savefig=None):
    ns_probs = [0 for _ in range(len(testy))]
    # calculate scores
    ns_auc = metrics.roc_auc_score(testy, ns_probs)
    lr_auc = metrics.roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = metrics.roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(testy, lr_probs)

    fig = plt.figure(figsize=(15, 12), dpi=50, facecolor='w', edgecolor='k')
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # show the legend
    plt.legend()
    # show the plot
    if savefig:
        fig1 = plt.gcf()
        plt.savefig(savefig, dpi=100, facecolor='w', edgecolor='k')
    plt.show()


def prc(pred_list, lr_probs, testy, savefig=None):
    yhat = pred_list
    lr_precision, lr_recall, _ = metrics.precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = metrics.f1_score(testy, yhat), metrics.auc(lr_recall, lr_precision)
    # summarize scores
    print('Model: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    fig = plt.figure(figsize=(15, 12), dpi=50, facecolor='w', edgecolor='k')
    # plot the precision-recall curves
    no_skill = len(testy[testy == 1]) / len(testy)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Model')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # show the legend
    plt.legend()
    # show the plot
    if savefig:
        fig1 = plt.gcf()
        plt.savefig(savefig, dpi=100, facecolor='w', edgecolor='k')
    plt.show()


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    hist = {
        'train': {
            'epoch_loss': [],
            'epoch_acc': [],
        },
        'val': {
            'epoch_loss': [],
            'epoch_acc': [],
        }
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and val phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
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
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            hist[phase]['epoch_loss'].append(epoch_loss)
            hist[phase]['epoch_acc'].append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, hist


def initialize_model(model_name, in_features, device):
    if model_name.startswith('resnet101'):
        model_ft = models.resnet101(pretrained=True)
    elif model_name.startswith('resnet50'):
        model_ft = models.resnet50(pretrained=True)
    elif model_name.startswith('resnet34'):
        model_ft = models.resnet34(pretrained=True)
    elif model_name.startswith('resnet18'):
        model_ft = models.resnet18(pretrained=True)
    elif model_name.startswith('resnet152'):
        model_ft = models.resnet152(pretrained=True)
    else:
        raise ValueError('unknown resnet model {}'.format(model_name))

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, in_features)
    model_ft = model_ft.to(device)

    try:
        model_ft.load_state_dict(torch.load('{}.pth'.format(model_name)))
        hist = pickle.load(open('{}.list'.format(model_name), "rb"))
        print('Loading customized pre-trained model')
    except Exception:
        print('Loading empty pre-trained model')
        hist = None

    return model_ft, hist
