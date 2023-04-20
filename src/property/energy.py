import jax
import jax.numpy as np
import flax.linen as nn
import model.MPNN as MPNN

class Property(nn.module):
    
    def __call__(self,params,cart,atomindex,shifts,species):
        return MPNN(params,cart,atomindex,shifts,species)
