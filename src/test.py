import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
# here is an example to use the sph calculation
import timeit 

import sph_cal
import density
import radial
sys.path.append("..")
from fortran import getneigh as fortran_neigh
import nn

class test():

    def __init(self,key,nwave,cutoff,index_l,radial_func,sph_cal,emb_nl,density):
       
        initkey=jrm.split(key,num=3)
        self.radial_func = radial.radial_func(nwave,cutoff)
        self.radial_params=radial_func.init(initkey[0],jrm.random.uniform(initkey[1],(10,)))
        self.sph_cal=sph_cal
        self.density=density
        self.index_l=index_l

    def __call__(cart,atomindex,shifts,coefficients):
        coor=cart[:,atomindex[1]]-cart[:,atomindex[0]]+shifts
        distances=jnp.linalg.norm(coor,axis=1)
        coor_t=jnp.einsum("ij ->ji",coor)
        radial=self.radial_func(self.radial_params,distances)
        sph=self.sph_cal(coor_t)
        density,sum_sph=self.density(sph,radial,self.index_l,atomindex[:,1],atomindex[:,0],coefficients)
        return density

max_l=3
index_l=jnp.array([0],dtype=jnp.int32)
for l in range(0,max_l+1):
    index_l=jnp.hstack((index_l,jnp.ones((2*l+1,),dtype=jnp.int32)*l))

cutoff=5.0
nwave=4
numatom=4
key=jrm.PRNGKey(0)
init_key=jrm.split(key,num=3)
emb_nl=[16,16,nwave]
cart=np.random.rand(3,numatom)*10
print(cart)
species=jnp.arange(numatom)
cell=jnp.zeros((3,3))
cell=cell.at[0,0].set(25.0)
cell=cell.at[1,1].set(25.0)
cell=cell.at[2,2].set(25.0)
atomindex=np.ones((2,20))
shifts=np.ones((3,20))
in_dier=cutoff/2.0
fortran_neigh.init_neigh(cutoff,in_dier,cell)

fortran_neigh.get_neigh(cart,atomindex,shifts)

fortran_neigh.deallocate_all()

jax.lax.stop_gradient(cart)
sph_cal=SPH_CAL(max_l=max_l)

radial_func=radial.radial_func(nwave,cutoff)

emb_nn=nn.MLP(emb_nl)
emb_params=emb_nn.init(init_key[2],jnp.ones(1))
coefficients=emb_nn(emb_params,species)

test_den=test(init_key[1],nwave,cutoff,index_l,radial_func,sph_cal.compute_sph,emb_nl,density.density)

density=test_den(cart,atomindex,shifts,coefficients)
print(density)

'''
spp=sph_cal.compute_sph(cart)
forward=jax.jit(jax.vmap(sph_cal,in_axes=(1),out_axes=(1)))
'''
