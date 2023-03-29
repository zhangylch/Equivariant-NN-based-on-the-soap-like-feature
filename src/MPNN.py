import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from flax import linen as nn

class MPNN():
    def __init__(self,oc_loop,numatom,nn_list,params_list,radial_func,cutoff_func,SPH_CAL,emb_nn,invariant):
        self.numatom=numatom
        self.oc_loop=oc_loop
        self.nn_list=nn_list
        self.params_list=params_list
        self.radial_func=radial_func
        self.cutoff_func=cutoff_func
        self.sph_cal=SPH_CAL
        self.emb_nn=emb_nn
        self.invariant=invariant

    def __call__(self,coor,atomindex,shifts,species):
        '''
        atomindex with the dimension of (2,batchsize,numatom*neigh)
        '''
        return self.forward(coor,atomindex,shifts,species)

    @partial(jit,static_augnums=0)
    def forward(self,coor,atomindex,shifts,species)
        # define the class for calculating the spherical harmonic expansion
        cart=jnp.take_along_axis(coor,atomindex[0],axis=1)-jnp.take_along_axis(coor,atomindex[1],axis=1)+shifts
        distances=jnp.linalg.norm(cart,axis=2).T  # to convert to the dimension (n,batchsize)
        radial=jnp.einsum("ijk,jk ->ijk",self.radial_func(distances,params_list[-2],params_list[-1]),self.cutoff_func(distances,cutoff))
        incart=cart.transpose(2,1,0)
        sph=sph_cal(incart,distances)
        MP_sph=jnp.zeros(sph.shape[0],self.numatom,sph.shape[2])
        invariant_quantity=self.emb_nn(params[-3],species.reshape(-1,1))
        for ioc_loop in range(oc_loop+1):
            params=self,params_list[ioc_loop]
            model=self.nn_list[ioc_loop]
            invariant_quantity,MP_sph=self.invariant(sph,MP_sph,radial,index_l,atomindex[1],atomidex[0],invariant_quantity,params,model)
        return invariant_quantity
