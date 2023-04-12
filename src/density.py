import jax 
import jax.numpy as jnp


@jit
def density(sph,radial,index_l,index_neigh,index_center,coefficients,MP_sph=jnp.zeros((0)),density=jnp.zeros((0))):
    '''
    sph is the spherical harmonic expansion with the dimension of (L*L,n). float/double
    radial is the array to store the radial function with the dimension of (nwave,n,batchsize) float/double
          n is the max number of neighbour atoms.
          batchsize is the number of structures in each mini-batch.
          L is the max angular moment.

    '''
    # to obtain the index for the coefficients for neighbour 
    neigh_coeff=coefficients[index_neigh]
    radial=jnp.einsum("ij,ij->ij",radial,neigh_coeff)
    r_sph=jnp.einsum("ji,ik -> ijk",sph,opt_radial)
    r_sph=MP_sph[index_neigh]+r_sph
    sum_sph=jnp.zeros((coefficients.shape[0],sph.shape[0],coefficients.shape[1]))
    contract_sph=jnp.square(sum_sph.at[index_center].add(r_sph))
    density=density.at[...,index_l].add(contract_sph).reshape(coefficients,shape[0],-1)
    return density,sum_sph

