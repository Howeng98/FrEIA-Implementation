import torch 
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 512),
        nn.ReLU(),
        nn.Linear(512, dims_out)
        )

class FixedRandomElementwiseMultiply(Fm.InvertibleModule):

    def __init__(self, dims_in):
        super().__init__(dims_in)
        self.random_factor = torch.randint(1, 3, size=(1, dims_in[0][0]))

    def forward(self, x, rev=False, jac=True):
        # the Jacobian term is trivial to calculate so we return it
        # even if jac=False

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]
        if not rev:
            # forward operation
            x = x * self.random_factor
            log_jac_det = self.random_factor.float().log().sum()
        else:
            # backward operation
            x = x / self.random_factor
            log_jac_det = -self.random_factor.float().log().sum()

        return (x,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims

class ConditionalSwap(Fm.InvertibleModule):

    def __init__(self, dims_in, dims_c):
        super().__init__(dims_in, dims_c=dims_c)

    def forward(self, x, c, rev=False, jac=True):
        # in this case, the forward and reverse operations are identical
        # so we don't use the rev argument
        x1, x2 = x
        log_jac_det = 0.

        # make copies of the inputs
        x1_new = x1 + 0.
        x2_new = x2 + 0.

        for i in range(x1.size(0)):
            x1_new[i] = x1[i] if c[0][i] > 0 else x2[i]
            x2_new[i] = x2[i] if c[0][i] > 0 else x1[i]

        return (x1_new, x2_new), log_jac_det

    def output_dims(self, input_dims):
        return input_dims