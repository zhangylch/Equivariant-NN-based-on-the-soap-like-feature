import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from flax import linen as nn
class MPNN():
    def __init__(self,oc_loop,params_list,radial_func,cutoff_func,SPH_CAL,emb_nn,invariant):
        self.oc_loop=oc_loop
        self.params_list=params_list
        self.radial_func=radial_func
        self.cutoff_func=cutoff_func
        self.sph_cal=SPH_CAL
        self.emb_nn=emb_nn
        self.invariant=invariant

    def __call__(self,coor,atomindex,shifts,species):
        return self.forward(coor,atomindex,shifts,species)

    @partial(jit,static_augnums=0)
    def MPNN(self,coor,atomindex,shifts,species)
        # define the class for calculating the spherical harmonic expansion
        cart=coor[atomindex[0]]-cart[atomindex[1]]+shifts
        distances=jnp.linalg.norm(cart,axis=0)
        radial=jnp.einsum("ijk,jk ->ijk",self.radial_func(distances,params_list[-2],params_list[-1]),self.cutoff_func(distances,cutoff))
        sph=sph_cal(cart)
        ext_sph=jnp.zeros(sph.shape[2],maxnumatom,sph.shape[0])
        invariant_quantity=self.emb_nn(params[-3],species)
        for ioc_loop in range(oc_loop+1):
            params=self,params_list[ioc_loop]
            invariant_quantity,ext_sph=self.invariant(sph,ext_sph,radial,index_l,atomindex[1],atomidex[0],invariant_quantity,params,model)
        return invariant_quantity
