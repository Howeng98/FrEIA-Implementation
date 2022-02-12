import torch 
import torch.nn as nn
from sklearn.datasets import make_moons

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# Parameters
BATCH_SIZE = 100
N_DIM = 2
EPOCHS = 50

# Subnet in the affine coupling block
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 512),
        nn.ReLU(),
        nn.Linear(512, dims_out)
        )

# ReversibleSequential Network
inn = Ff.SequenceINN(N_DIM)
for k in range(8):
    inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

# Optimizer
optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)

# Train
for i in range(EPOCHS):
    optimizer.zero_grad()
    data, label = make_moons(n_samples=BATCH_SIZE, noise=0.05)
    x = torch.Tensor(data)

    # calculate transformed variable z and log Jacobian determinant
    z, log_jac_det = inn(x)

    # L2 norm
    loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
    loss = loss.mean() / N_DIM

    # backpropagate and update weights
    loss.backward()
    optimizer.step()
    if i % 5 == 0:
        print('EPOCH:{:2d} , Loss:{:.4f}'.format(i, loss))

# Generate sample from a standard normal distribution by INN
z = torch.randn(BATCH_SIZE, N_DIM)
samples, _ = inn(z, rev=True)
