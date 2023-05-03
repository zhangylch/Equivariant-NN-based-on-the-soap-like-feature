import jax 
import jax.numpy as jnp


def density(sph,radial,radial_cutoff,index_l,index_neigh,index_center,coefficients,MP_sph,density=jnp.zeros((0)),Dtype=jnp.dtype("float32")):
    '''
    sph is the spherical harmonic expansion with the dimension of (L*L,n). float/double
    radial is the array to store the radial function with the dimension of (nwave,n,batchsize) float/double
          n is the max number of neighbour atoms.
          batchsize is the number of structures in each mini-batch.
          L is the max angular moment.

    '''
    # to obtain the index for the coefficients for neighbour 
    neigh_coeff=coefficients[index_neigh]
    opt_radial=jnp.einsum("i,ji,ij->ij",radial_cutoff,radial,neigh_coeff)
    r_sph=jnp.einsum("ji,ik -> ijk",sph,opt_radial)
    r_sph=jnp.einsum("ijk,i",MP_sph[index_neigh],radial_cutoff)+r_sph # out of bound will be ingored
    sum_sph=jnp.zeros((coefficients.shape[0],sph.shape[0],coefficients.shape[1]),dtype=Dtype)
    sum_sph=sum_sph.at[index_center].add(r_sph)
    contract_sph=jnp.square(sum_sph)
    density=density.at[:,index_l].add(contract_sph)
    return density,sum_sph

