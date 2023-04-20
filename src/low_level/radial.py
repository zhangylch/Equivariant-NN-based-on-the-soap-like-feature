import jax
import jax.numpy as jnp
from jax.nn.initializers import ones,uniform
from flax import linen as nn
from jax.numpy import dtype

class radial_func(nn.Module):
    nwave: int
    cutoff: float
    Dtype: dtype=dtype(jnp.float32)
    
    def setup(self):
        uniform_init=uniform(self.cutoff)
        self.alpha=self.param("aplha",ones,(self.nwave,1),self.Dtype)
        self.center=self.param("center",uniform_init,(self.nwave,1),self.Dtype)

    def __call__(self,distances):
        return self.gaussian_func(distances)*self.cutoff_func(distances)
    
    def gaussian_func(self,distances):
        '''
        Here, we employ the gaussian function with optimizable parameters as the radial function
        distances is the array to store the distances between the center atoms with its neighbors with the dimension of (numatom*neigh,batchsize) float/double
        alpha is an optimizable parameters with its dimension of (nwave)
        center is an optimizable parameters with its dimension of (nwave)
        '''
        shift_distances=distances-self.center
        gaussian=jnp.exp(-self.alpha*(shift_distances*shift_distances))
        return gaussian
    
    def cutoff_func(self,distances):
        tmp=(jnp.cos(distances/self.cutoff*jnp.pi)+1.0)/2.0
        return tmp*tmp*tmp  # here to use the a^3 to keep the smooth of hessian functtion
