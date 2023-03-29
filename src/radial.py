import jax
import jax.numpy as jnp

@jit
def radial_func(distances,alpha,center):
    '''
    Here, we employ the gaussian function with optimizable parameters as the radial function
    distances is the array to store the distances between the center atoms with its neighbors with the dimension of (numatom*neigh,batchsize) float/double
    alpha is an optimizable parameters with its dimension of (nwave)
    center is an optimizable parameters with its dimension of (nwave)
    '''
    shift_distances=distances[None,:,:]-center[:,None,None]
    gaussian=jnp.exp(alpha*(shift_distances*shift_distances))
    return gaussian

@jit
def cutoff_func(distances,cutoff)
    '''

    '''
    tmp=(jnp.cos(distances/cutoff*jnp.pi)+1.0)/2.0
    return tmp*tmp*tmp  # here to use the a^3 to keep the smooth of hessian functtion
