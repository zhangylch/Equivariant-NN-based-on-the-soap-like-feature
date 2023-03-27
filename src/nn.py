import flax
from flax import linen as nn
from flax.linen.activation import silu
from typing import Sequence


class MLP(nn.Module):
    nl: Sequence[int]=None    # The nl is the structure of the nn but do not include the input layer
    def setup(self):
        self.nn=[nn.Dense(neuron) for neuron in self.nl]

    def __call__(self,x):
        for layer in self.nl[0:-1]:
            x=silu(layer(x))
        return self.nl[-1](x)
