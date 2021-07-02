import torch.optim as optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
# from torchsummary import summary
from data import dataloader
from model import Net
from test import test
from utils import plot_single
import numpy as np


def train_model(name, device, input_size = (3, 32, 32), MAX_LR = 0.2, EPOCHS = 20, train_batch_size=128, val_batch_size=128, ):
    model = Net().to(device)
    train_losses = []
    test_losses = []
    train_acc = []
    test_accs = []

    train_loader, test_loader = dataloader("CIFAR10", train_batch_size=train_batch_size,
                                                val_batch_size=val_batch_size,)


    wd = 0

    optimizer = get_optimizer(model.parameters(), lr=0.01, momentum=0.9, weight_decay=wd)
    # lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, int((EPOCHS+1)*0.7), EPOCHS], [MAX_LR/5.0, MAX_LR, MAX_LR/5.0, 0.004])[0]
    # lr_func = lambda: lr_schedule(global_step/())/BATCH_SIZE

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=MAX_LR,
                                                steps_per_epoch=len(train_loader),
                                                epochs=EPOCHS) 
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.05, patience=1)
     
    print("*****Training Start*****")
    for epoch in range(EPOCHS):
        print('EPOCH {} and Learning Rate {}: '.format(epoch+1, scheduler.get_last_lr()))
        train_epoch_losses, train_epoch_acc = train(model, device, train_loader, optimizer, scheduler)
        test_loss, test_acc = test(model, device, test_loader)
        train_losses.extend(train_epoch_losses)
        train_acc.extend(train_epoch_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if (epoch > 0) and (test_accs[epoch] > test_accs[epoch - 1]):
            torch.save(model.state_dict(), f"{name}.pth")

  
    plot_single(name, train_losses, train_acc, test_losses, test_accs)
    return model, train_losses, test_losses, train_acc, test_accs

def get_optimizer(params, lr=0.01, momentum=0.9, weight_decay=0):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def train(model, device, train_loader, optimizer, scheduler,  lambda_l1=None):
    train_losses = []
    train_acc = []

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)
        if lambda_l1:
            l1 = 0
            for p in model.parameters():
                l1 += p.abs().sum()
            loss += lambda_l1 * l1

        train_losses.append(loss)

        loss.backward()
        optimizer.step()

        # lr changes
        scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_losses, train_acc