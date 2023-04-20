import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype,array
import flax.linen as nn
from typing import Sequence
from src.low_level import density, MLP, sph_cal, radial


class MPNN(nn.Module):
    emb_nl: Sequence[int]
    MP_nl: Sequence[int]
    output_nl: Sequence[int]
    key: array=jrm.PRNGKey(0)
    nwave: int=8
    max_l: int=2
    MP_loop: int=2
    cutoff: float=5.0
    Dtype: dtype=dtype(jnp.float32)
    

    def setup(self):
        self.r_max_l=self.max_l+1
        self.norbit=self.nwave*self.r_max_l

        key=jrm.split(self.key,num=3)
        # define the class for the calculation of radial function
        self.radial_func = radial.radial_func(self.nwave,self.cutoff,Dtype=self.Dtype)
        self.radial_params = self.radial_func.init(key[0],jrm.uniform(key[1],(10,)))
        
        # define the class for the calculation of spherical harmonic expansion
        self.sph_cal=sph_cal.SPH_CAL(max_l=self.max_l,Dtype=self.Dtype)
        # the first time is slow for the compile of the jit
        self.sph_cal(jnp.ones(3))
        
        # define the embedded layer used to convert the atomin number of a coefficients
        key=jrm.split(key[-1])
        self.emb_nn=MLP.MLP(self.emb_nl,self.nwave)
        self.emb_params=self.emb_nn.init(key[0],jnp.ones(1))

        # used for the convenient summation over the same l
        self.index_l=jnp.array([0],dtype=jnp.int32)
        for l in range(1,self.r_max_l):
            self.index_l=jnp.hstack((self.index_l,jnp.ones((2*l+1,),dtype=jnp.int32)*l))

        # define the density calculation
        self.density=density.density

        # Instantiate the NN class for the MPNN
        # create tge model for each iterations in MPNN
        self.MPNN_list=[MLP.MLP(self.MP_nl,self.nwave) for iMP_loop in range(self.MP_loop)]
        self.outnn=MLP.MLP(self.output_nl,1)
        key=jrm.split(key[-1],num=self.MP_loop+2)  # The 3 more key is for the final nn, embedded nn and the seed to generate next key.
        random_x=jnp.ones(self.norbit)
        # initialize the model 
        self.MP_params_list=[self.MPNN_list[iMP_loop].init(key[iMP_loop],random_x) for iMP_loop in range(self.MP_loop)]
        self.out_params=self.outnn.init(key[self.MP_loop],random_x)
        #embeded nn

       

    def __call__(self,cart,atomindex,shifts,species):
        coor=cart[:,atomindex[1]]-cart[:,atomindex[0]]+shifts
        distances=jnp.linalg.norm(coor,axis=0)
        radial=self.radial_func.apply(self.radial_params,distances)
        sph=self.sph_cal(coor/self.cutoff)
        MP_sph=jnp.zeros((cart.shape[0],sph.shape[0],self.nwave),dtype=cart.dtype)
        density=jnp.zeros((cart.shape[1],self.r_max_l,self.nwave),dtype=cart.dtype)
        coefficients=self.emb_nn.apply(self.emb_params,species)
        for inn, nn in enumerate(self.MPNN_list):
            density,MP_sph=self.density(sph,radial,self.index_l,atomindex[1],atomindex[0],coefficients,MP_sph,density)
            coefficients=nn.apply(self.MP_params_list[inn],density.reshape(-1,self.norbit))
        density,MP_sph=self.density(sph,radial,self.index_l,atomindex[1],atomindex[0],coefficients,MP_sph,density)
        output=jnp.sum(self.outnn.apply(self.out_params,density.reshape(-1,self.norbit)))
        return output


