import flax
from flax import linen as nn
from flax.linen.activation import silu

class NN(nn.Module):
    def setup(self,nl,outputneuron):
        MLP=[]
        for i in range(nl):
            MLP.append(nn.Dense(i))
            MLP.append(silu)
        MLP.append(nn.Dense(outputneuron))
        self.MLP=nn.Sequential(MLP)

    def __call__(self,x):
        return self.MLP(x)
