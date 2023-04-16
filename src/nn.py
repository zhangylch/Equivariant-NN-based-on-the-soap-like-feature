import flax
from flax import linen as nn
from flax.linen.activation import silu
from typing import Sequence


class MLP(nn.Module):
    nl: Sequence[int]    # The nl is the structure of the nn but do not include the input layer
    nout: int
    def setup(self):
        self.nn=[nn.Dense(neuron) for neuron in self.nl]
        self.output=nn.Dense(self.nout)

    def __call__(self,x):
        for layer in self.nn:
            x=silu(layer(x))
        return self.output(x)
