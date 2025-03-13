import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import argparse
import os
import core

from dataset import Dataset, TestDataset


parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--data', default='./data/data.npz', type=str, help='data npz path')
parser.add_argument('--data_test', default='./data/data_test.npz', type=str, help='test data')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batchsize', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--checkpoint', default=None, help='pretrained model')
parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
args = parser.parse_args()

dataset = Dataset(args.data)
dataLoader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)
test_dataset = TestDataset(args.data_test)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = core.CLF_QP_Net(n_input=4, n_hidden=64, n_controls=1)

#net = torch.nn.DataParallel(net)
net = net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
drop_after_epoch = [
    int(args.epochs * 1/3), int(args.epochs * 2/3), int(args.epochs * 5/6)]
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=drop_after_epoch, gamma=0.3)

nominal_controller = core.Segway_LQR()
x_goal = torch.from_numpy(np.array([2, 0, 0, 0], dtype=np.float32).reshape(1, 4, 1))

min_loss_test = np.inf

for i in range(args.epochs):
    loss_train = 0.0
    for j, (xs, labels) in enumerate(dataLoader):
        xs = xs.to(device)
        labels = labels.to(device)

        safe_mask = labels == 1
        unsafe_mask = labels == -1

        optimizer.zero_grad()
        loss = core.lyapunov_loss(xs, x_goal, safe_mask, unsafe_mask, net, nominal_controller)
        loss.backward()
        optimizer.step()

        loss_train += loss.detach().cpu().numpy()

    scheduler.step()
    loss_train = loss_train / len(dataLoader)

    with torch.no_grad():
        xs, labels = test_dataset.get_test_data()
        xs = torch.from_numpy(xs).to(device)
        labels = torch.from_numpy(labels).to(device)

        safe_mask = labels == 1
        unsafe_mask = labels == -1

        loss_test = core.lyapunov_loss(xs, x_goal, safe_mask, unsafe_mask, net, 
                                       nominal_controller, print_loss=True)
        loss_test = loss_test.detach().cpu().numpy()
        print('Epoch {}, Loss train: {:.3f}, test: {:.3f}'.format(i, loss_train, loss_test))

    if loss_test < min_loss_test:
        filename = 'save/model.pth'
        torch.save(net.state_dict(), filename)
        min_loss_test = loss_test
        print('model saved')