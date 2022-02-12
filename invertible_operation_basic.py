import FrEIA.framework as Ff
import FrEIA.modules as Fm
import config as c
from modules import FixedRandomElementwiseMultiply

# build up basic net using SequenceINN
net = Ff.SequenceINN(c.N_DIM)
for i in range(2):
    net.append(FixedRandomElementwiseMultiply)

# define inputs
x = torch.randn(c.BATCH_SIZE, c.N_DIM)

# run forward
z, log_jac_det = net(x)

# run in reverse
x_rev, log_jac_det_rev = net(z, rev=True)