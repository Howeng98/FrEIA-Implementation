import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import config as c
from modules import ConditionalSwap, FixedRandomElementwiseMultiply

# define a graph-INN

# nodes
input_1 = Ff.InputNode(c.DIMS_IN, name='input_1')
input_2 = Ff.InputNode(c.DIMS_IN, name='input_2')

condition = Ff.ConditionNode(1, name='condition')

mult_1 = Ff.Node(input_1.out0, FixedRandomElementwiseMultiply, {}, name='mult_1')
cond_swap = Ff.Node([mult_1.out0, input_2.out0], ConditionalSwap, {}, conditions=condition, name='condition_swap')
mult_2 = Ff.Node(cond_swap.out1, FixedRandomElementwiseMultiply, {}, name='mult_2')

output_1 = Ff.OutputNode(cond_swap.out0, name='output_1')
output_2 = Ff.OutputNode(mult_2.out0, name='output_2')

net = Ff.GraphINN([input_1, input_2, mult_1, mult_2, condition, cond_swap, output_1, output_2])

# define inputs
x1 = torch.randn(c.BATCH_SIZE, c.DIMS_IN)
x2 = torch.randn(c.BATCH_SIZE, c.DIMS_IN)
c  = torch.randn(c.BATCH_SIZE)

# run forward
(z1, z2), log_jac_det = net([x1, x2], c=c)

# run in reverse without necessarily calculating Jacobian term (i.e. jac=False)
(x1_rev, x2_rev), _ = net([z1, z2], c=c, rev=True, jac=False)
