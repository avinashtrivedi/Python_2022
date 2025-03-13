import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
import os

from core import (
    CLF_QP_Net,
    lyapunov_loss
)

def compute_safe_mask(s):
    x, y = s[:, 0], s[:, 1]
    rho = torch.sqrt(x**2 + y**2)
    in_safe_region = torch.logical_or(rho <= 2, torch.logical_and(rho >=6, rho <= 8))
    in_triangle = torch.logical_and(y <= 0, torch.abs(x) <= torch.abs(y))
    safe_mask = torch.logical_or(in_safe_region, in_triangle)

    in_dangerous_region = torch.logical_and(rho >= 3, rho <= 5) 
    dang_mask = torch.logical_and(in_dangerous_region, torch.logical_not(in_triangle))
    far_region = rho >= 10
    dang_mask = torch.logical_or(dang_mask, far_region)
    return safe_mask, dang_mask


N_train = 1000000
s_train = torch.Tensor(N_train, 4).uniform_(-12, 12)

batch_size = 64
epochs = 100

N_test = 10000
s_test = torch.Tensor(N_test, 4).uniform_(-12, 12)
safe_mask_test, unsafe_mask_test = compute_safe_mask(s_test)

s0 = torch.zeros(1, 4)

clf_net = CLF_QP_Net(4, 256, 2)

filename = 'logs/model.pth'
if os.path.exists(filename):
    print('loading saved model')
    clf_net.load_state_dict(torch.load(filename))

# Initialize the optimizer
# print('before optimizer')
optimizer = optim.Adam(clf_net.parameters(), lr=1e-4, weight_decay=1e-5)

# print('After optimizer')


# Train!
test_losses = []
u_learn_data = []
for epoch in range(epochs):
    xx = torch.tensor([])
    # Randomize presentation order
    permutation = torch.randperm(N_train)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        s = s_train[indices]

        # Segment into safe/unsafe
        safe_mask, unsafe_mask = compute_safe_mask(s)

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Compute loss
        loss = 0.0
        
        myloss = lyapunov_loss(s,
                              s0,
                              safe_mask,
                              unsafe_mask,
                              clf_net,
                              print_loss=False)
        
        
        loss += myloss[0]
        xx = myloss[1]

        # Accumulate loss from this epoch and do backprop
        loss.backward()
        loss_acumulated += loss.detach()

        # Update the parameters
        optimizer.step()
    
    # store
    
    u_learn_data.append(xx)

    # Print progress on each epoch, then re-zero accumulated loss for the next epoch
    print(f'Epoch {epoch + 1} training loss: {loss_acumulated / (N_train / batch_size)}')
    loss_acumulated = 0.0

    # Get loss on test set
    with torch.no_grad():
        # Compute loss
        loss = 0.0
        loss += lyapunov_loss(s_test,
                              s0,
                              safe_mask_test,
                              unsafe_mask_test,
                              clf_net,
                              print_loss=True)[0]
        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")
        if not os.path.exists('./vis'):
            os.mkdir('./vis')
        
        V, _ = clf_net.compute_lyapunov(s_test)
        s_test_np = s_test.detach().numpy()
        V_np = V.detach().numpy()

        plt.scatter(s_test_np[:, 0], s_test_np[:, 1], c=V_np, s=1)
        plt.savefig('./vis/{}.png'.format(epoch))

        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            filename = 'logs/model.pth'
            torch.save(clf_net.state_dict(), filename)
        test_losses.append(loss.item())
# save
# print(u_learn_data)
u_learn_data = [i.detach().numpy() for i in u_learn_data]
np.save('u_learn_data.npy', np.array(u_learn_data))  