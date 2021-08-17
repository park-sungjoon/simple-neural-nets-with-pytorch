import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm


loss_func = nn.CrossEntropyLoss(reduction='mean')


def compute_loss(model, imgs, labels):
    """compute loss and true positives

    Args:
        model: neural network
        imgs(torch.tensor): batch of input image
        labels(torch.tensor) labels corresponding to imgs

    Returns:
        tuple of the form (loss computed for batch (torch.tensor), true positives in the batch)
    """
    logits, probabilities = model(imgs)
    predictions = torch.argmax(probabilities, dim=1)
    TP = torch.sum(predictions == labels)
    return loss_func(logits, labels), TP


def train_model(model, train_loader, test_loader, device, epochs=10):
    """ Train a given model
    Args:
        model: model to be trained
        train_loader (torch.utils.data.DataLoader): DataLoader for training
        test_loader (torch.utils.data.DataLoader): DataLoader for validation
        device (str): device on which we train 'cpu' or 'cuda'
        epochs (int): number of epochs in training loop

    Returns:
        list: list of loss and accuracy obtained for training and validation. 
            Note that we assume that the total number of training data is 60,000
            and that the number of test data is 10,000 (true for full MNIST dataset)
    """
    # optimize with Adam and use cross entropy loss
    optimizer = Adam(model.parameters(), lr=0.001)

    # define lists that will contain loss and accuracy values obtained during training and validation
    loss_trn_list = []
    loss_val_list = []
    acc_trn_list = []
    acc_val_list = []
    for epoch in tqdm(range(epochs), desc=f'percent of epochs completed'):
        loss_trn = 0.
        loss_val = 0.
        trn_correct = 0
        val_correct = 0

        # training loop
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss, TP = compute_loss(model, imgs.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_trn += loss.item()
            trn_correct += TP.item()

        loss_trn_list.append(loss_trn * 128 / 60000)  # rougly the average training loss
        acc_trn_list.append(trn_correct / 60000)  # average training accuracy

        # validation loop
        model.eval()
        with torch.no_grad():
            for imgs, labels in test_loader:
                loss, TP = compute_loss(model, imgs.to(device), labels.to(device))
                loss_val += loss.item()
                val_correct += TP.item()

        loss_val_list.append(loss_val * 128 / 10000)  # roughly the average validation loss
        acc_val_list.append(val_correct / 10000)  # average validation accuracy

    return loss_trn_list, loss_val_list, acc_trn_list, acc_val_list
