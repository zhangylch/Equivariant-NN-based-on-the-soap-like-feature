import jax
import jax.numpy as jnp

@jit
def radial_func(distances,alpha,center):
    '''
    Here, we employ the gaussian function with optimizable parameters as the radial function
    distances is the array to store the distances between the center atoms with its neighbors with the dimension of (n,batchsize) float/double
    '''
    gaussian=jnp.exp(alpha*(distances[None,:,:]-center[:,None,None]))
    return gaussian

def cutoff_func(distances,cutoff)
    a=(jnp.cos(distances/cutoff*jnp.pi)+1)/2.0
    return a*a*a  # here to use the a^3 to keep the smooth of hessian functtion
