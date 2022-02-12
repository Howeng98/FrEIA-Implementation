import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import config as c

in1 = Ff.InputNode(100, name='Input 1') # 1D vector
in2 = Ff.InputNode(20, name='Input 2') # 1D vector
cond = Ff.ConditionNode(42, name='Condition')

def subnet(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 256), nn.ReLU(),
                         nn.Linear(256, dims_out))

perm = Ff.Node(in1, Fm.PermuteRandom, {}, name='Permutation')
split1 =  Ff.Node(perm, Fm.Split, {}, name='Split 1')
split2 =  Ff.Node(split1.out1, Fm.Split, {}, name='Split 2')
actnorm = Ff.Node(split2.out1, Fm.ActNorm, {}, name='ActNorm')
concat1 =  Ff.Node([actnorm.out0, in2.out0], Fm.Concat, {}, name='Concat 1')
affine = Ff.Node(concat1, Fm.AffineCouplingOneSided, {'subnet_constructor': subnet},
                 conditions=cond, name='Affine Coupling')
concat2 =  Ff.Node([split2.out0, affine.out0], Fm.Concat, {}, name='Concat 2')

output1 = Ff.OutputNode(split1.out0, name='Output 1')
output2 = Ff.OutputNode(concat2, name='Output 2')

example_INN = Ff.GraphINN([in1, in2, cond,
                           perm, split1, split2,
                           actnorm, concat1, affine, concat2,
                           output1, output2])

# dummy inputs:
x1, x2, c = torch.randn(16, 100), torch.randn(16, 20), torch.randn(16, 42)

# compute the outputs
(z1, z2), log_jac_det = example_INN([x1, x2], c=c)

# invert the network and check if we get the original inputs back:
(x1_inv, x2_inv), log_jac_det_inv = example_INN([z1, z2], c=c, rev=True)
assert (torch.max(torch.abs(x1_inv - x1)) < 1e-5 and torch.max(torch.abs(x2_inv - x2)) < 1e-5)