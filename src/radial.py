import jax
import jax.numpy as jnp

@jit
def radial(distances,alpha,center):
    '''
    Here, we employ the gaussian function with optimizable parameters as the radial function
    distances is the array to store the distances between the center atoms with its neighbors with the dimension of (n,batchsize) float/double
    '''
    gaussian=jnp.exp(alpha*(distances[None,:,:]-center[:,None,None]))
    return gaussian
