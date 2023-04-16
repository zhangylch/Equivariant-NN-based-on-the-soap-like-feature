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
import MPNN
sys.path.append("../fortran")
import getneigh
import nn

cutoff=5.0
nwave=2
max_l=4
numatom=8
maxneigh=40
key=jrm.PRNGKey(0)
emb_nl=[4,4]
MP_nl=[16,16]
output_nl=[8,8]
MP_loop=0
dtype=jnp.dtype("float32")
cart=(np.random.rand(3,numatom)*10).astype(dtype)
species=jnp.array([12,1,1,1,3,5,3,1]).reshape(-1,1)
cell=jnp.zeros((3,3),dtype=dtype)
cell=cell.at[0,0].set(25.0)
cell=cell.at[1,1].set(25.0)
cell=cell.at[2,2].set(25.0)
atomindex=np.ones((2,maxneigh),dtype=np.int32)
shifts=np.ones((3,maxneigh),dtype=dtype)
in_dier=cutoff
getneigh.init_neigh(numatom,cutoff,in_dier,cell)

atomindex,shifts=getneigh.get_neigh(cart,maxneigh)
print(atomindex)
getneigh.deallocate_all()


model=MPNN.MPNN(key,radial.radial_func,sph_cal.SPH_CAL,density.density,nn.MLP,nwave=nwave,max_l=max_l,cutoff=cutoff,MP_loop=MP_loop,emb_nl=emb_nl,MP_nl=MP_nl,output_nl=output_nl,Dtype=dtype)

energy=model(cart,atomindex,shifts,species)
rotate=jnp.zeros((3,3),dtype=dtype)
ceta=np.pi/4
rotate=rotate.at[2,2].set(1.0)
rotate=rotate.at[1,1].set(jnp.cos(ceta))
rotate=rotate.at[0,0].set(jnp.cos(ceta))
rotate=rotate.at[0,1].set(jnp.sin(ceta))
rotate=rotate.at[1,0].set(-jnp.sin(ceta))
cart=jnp.einsum("ij,jk->ik",rotate,cart)
energy1=model(cart,atomindex,shifts,species)
print(energy1,energy,energy1-energy)
