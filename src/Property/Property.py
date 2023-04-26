import jax
import jax.numpy as np
import flax.linen as nn
import model.MPNN as MPNN
import jax.random as jrm
from jax.numpy import dtype
from functools import partial

class Property():
    def __init__(emb_nl=[16,16],MP_nl=[64,64],output_nl=[64,64],key=jrm.PRNGKey(0),nwave=8,max_l=2,MP_lopp=2,cutoff=5.0,Dtype=dtype(jnp.float32),force=None):
        MPNN=MPNN.MPNN(emb_nl,MP_nl,output_nl,key=key[0],nwave=nwave,max_l=max_l,MP_loop=MP_loop,cutoff=cutoff,Dtype=dtype)
        if force is not None:
            self.model=jax.value_and_grad(MPNN,argnums=1)
        else:
            self.model=MPNN

    @partial(jit,static_argnums=0)    
    def __call__(self,params,cart,atomindex,shifts,species):
        return self.model.apply(params,cart,atomindex,shifts,species)
