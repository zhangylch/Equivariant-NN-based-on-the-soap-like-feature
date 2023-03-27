import jax 
import jax.numpy as jnp


@jit
def invariant(sph,MP_sph,radial,index_l,index_neigh,index_center,coefficients,params,model):
    '''
    sph is the spherical harmonic expansion with the dimension of (L*L,n,batchsize). float/double
    radial is the array to store the radial function with the dimension of (nwave,n,batchsize) float/double
          n is the max number of neighbour atoms.
          batchsize is the number of structures in each mini-batch.
          L is the max angular moment.

    '''
    # to obtain the index for the coefficients for neighbour 
    expand_index=jnp.repeat(jnp.extend_dims(index_neigh,axis=2),repeats=coefficients.shape[1],axis=2)
    expand_coeff=jnp.repeat(jnp.extend_dims(coefficients,axis=0),repeats=index_neigh.shape[0],axis=0)
    neigh_coeff=jnp.take_along_axis(expand_coeff,expand_index,axis=1)
    sph=jn.take_along_axis(MP_sph,expand_index,axis=1)+sph
    radial=jnp.einsum("ijk,kji->ijk",radial,neigh_coeff)
    r_sph=jnp.einsum("kij,lij -> jilk",sph,opt_radial)
    sum_sph=jnp.sum(r_sph,axis=2)
    density=density.at[:,index_center,:,index_l].add(sum_sph).reshape(*sum_orb[:-2],-1)
    invariant_quantity=jnp.sum(model.apply(params,density),axis=1)
    return invariant_quantity,sum_sph

