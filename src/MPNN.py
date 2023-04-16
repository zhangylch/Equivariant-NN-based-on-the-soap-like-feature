import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype


class MPNN():
    def __init__(self,key,radial_func,sph_cal,density,MLP,nwave=8,max_l=2,cutoff=5.0,MP_loop=2,emb_nl=[8,8],MP_nl=[64,64],output_nl=[64,64],Dtype=jnp.dtype("float32")):
        self.nwave=nwave
        self.MP_loop=MP_loop
        self.max_l=max_l+1
        self.norbit=nwave*self.max_l
        self.cutoff=cutoff
        # define the class for the calculation of spherical harmonic expansion
        self.sph_cal=sph_cal(max_l=max_l,Dtype=Dtype)
        # the first time is slow for the compile of the jit
        self.sph_cal(jnp.ones(3))
        
        key=jrm.split(key,num=3)
        # define the class for the calculation of radial function
        self.radial_func = radial_func(nwave,cutoff,Dtype=Dtype)
        self.radial_params = self.radial_func.init(key[0],jrm.uniform(key[1],(10,)))
        
        # define the embedded layer used to convert the atomin number of a coefficients
        key=jrm.split(key[-1])
        self.emb_nn=MLP(emb_nl,nwave)
        self.emb_params=self.emb_nn.init(key[0],jnp.ones(1))


        # used for the convenient summation over the same l
        self.index_l=jnp.array([0],dtype=jnp.int32)
        for l in range(1,self.max_l):
            self.index_l=jnp.hstack((self.index_l,jnp.ones((2*l+1,),dtype=jnp.int32)*l))

        # define the density calculation
        self.density=density

        # Instantiate the NN class for the MPNN
        # create tge model for each iterations in MPNN
        self.nn_list=[]
        for iMP_loop in range(MP_loop):
            self.nn_list.append(MLP(MP_nl,nwave))
        self.nn_list.append(MLP(output_nl,1))
        key=jrm.split(key[-1],num=MP_loop+2)  # The 3 more key is for the final nn, embedded nn and the seed to generate next key.
        random_x=jnp.ones(self.norbit)
        # initialize the model 
        self.params_list=[]
        for iMP_loop in range(MP_loop+1):
            self.params_list.append(self.nn_list[iMP_loop].init(key[iMP_loop],random_x))
        #embeded nn

       

    def __call__(self,cart,atomindex,shifts,species):
        coor=cart[:,atomindex[1]]-cart[:,atomindex[0]]+shifts
        distances=jnp.linalg.norm(coor,axis=0)
        radial=self.radial_func.apply(self.radial_params,distances)
        sph=self.sph_cal(coor/self.cutoff)
        MP_sph=jnp.zeros((cart.shape[0],sph.shape[0],self.nwave),dtype=cart.dtype)
        density=jnp.zeros((cart.shape[1],self.max_l,self.nwave),dtype=cart.dtype)
        coefficients=self.emb_nn.apply(self.emb_params,species)
        for inn, nn in enumerate(self.nn_list):
            density,MP_sph=self.density(sph,radial,self.index_l,atomindex[1],atomindex[0],coefficients,MP_sph,density)
            coefficients=nn.apply(self.params_list[inn],density.reshape(-1,self.norbit))
            print(density.shape)
            print(density)
            #print(coefficients)
        output=jnp.sum(coefficients)
        return output


