import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
import jax.random as jrm
import scipy

class SPH_CAL():
    def __init__(self,max_l=3):
        '''
         max_l: maximum L for angular momentum
         device: cpu/gpu
         dtype:  torch.float32/torch.float64

        '''
        #  form [0,max_L]
        if max_l<1: raise ValueError("The angular momentum must be greater than or equal to 1. Or the angular momentum is lack of angular information, the calculation of the sph is meanless.")
        self.max_l=max_l+1
        self.pt=np.empty((self.max_l,self.max_l),dtype=np.int64)
        self.yr=np.empty((self.max_l,self.max_l),dtype=np.int64)
        self.yr_rev=np.empty((self.max_l,self.max_l),dtype=np.int64)
        num_lm=int((self.max_l+1)*self.max_l/2)
        self.coeff_a=np.empty(num_lm)
        self.coeff_b=np.empty(num_lm)
        tmp=jnp.arange(self.max_l)
        self.prefactor1=-np.sqrt(1.0+0.5/tmp)
        self.prefactor2=np.sqrt(2.0*tmp+3)
        ls=tmp*tmp
        for l in range(self.max_l):
            self.pt[l,0:l+1]=tmp[0:l+1]+int(l*(l+1)/2)
            # here the self.yr and self.yr_rev have overlap in m=0.
            self.yr[l,0:l+1]=ls[l]+l+tmp[0:l+1]
            self.yr_rev[l,0:l+1]=ls[l]+l-tmp[0:l+1]
            if l>0.5:
                self.coeff_a[self.pt[l,0:l]]=np.sqrt((4.0*ls[l]-1)/(ls[l]-ls[0:l]))
                self.coeff_b[self.pt[l,0:l]]=-np.sqrt((ls[l-1]-ls[0:l])/(4.0*ls[l-1]-1.0))

        self.sqrt2_rev=np.sqrt(1/2.0)
        self.sqrt2pi_rev=np.sqrt(0.5/np.pi)


    #@partial(jit,static_argnums=0)
    def compute_sph(self,cart):
        '''
        cart: Cartesian coordinates with the dimension (3,n,batch) n is the max number of neigbbors and the rest complemented with 0. Here, we do not do the lod of tensor to keep the dimension of batch for the convenient calculation of sample to sample gradients.
        '''
        distances=jnp.linalg.norm(cart,axis=0)  # to convert to the dimension (n,batchsize)
        d_sq=distances*distances
        sph=jnp.empty((self.max_l*self.max_l,cart.shape[1],cart.shape[2]))
        sph=sph.at[0].set(self.sqrt2pi_rev*self.sqrt2_rev)
        sph=sph.at[1].set(self.prefactor1[1]*self.sqrt2pi_rev*cart[1])
        sph=sph.at[2].set(self.prefactor2[0]*self.sqrt2_rev*self.sqrt2pi_rev*cart[2])
        sph=sph.at[3].set(self.prefactor1[1]*self.sqrt2pi_rev*cart[0])
        for l in range(2,self.max_l):
            sph=sph.at[self.yr[l,0]].set(self.coeff_a[self.pt[l,0]]*(cart[2]*sph[self.yr[l-1,0]]+self.coeff_b[self.pt[l,0]]*d_sq*sph[self.yr[l-2,0]]))
            for m in range(1,l-1):
                sph=sph.at[self.yr[l,m]].set(self.coeff_a[self.pt[l,m]]*(cart[2]*sph[self.yr[l-1,m]]+self.coeff_b[self.pt[l,m]]*d_sq*sph[self.yr[l-2,m]]))
                sph=sph.at[self.yr_rev[l,m]].set(self.coeff_a[self.pt[l,m]]*(cart[2]*sph[self.yr_rev[l-1,m]]+self.coeff_b[self.pt[l,m]]*d_sq*sph[self.yr_rev[l-2,m]]))
            sph=sph.at[self.yr[l,l-1]].set(self.prefactor2[l-1]*cart[2]*sph[self.yr[l-1,l-1]])
            sph=sph.at[self.yr_rev[l,l-1]].set(self.prefactor2[l-1]*cart[2]*sph[self.yr_rev[l-1,l-1]])
            sph=sph.at[self.yr[l,l]].set(self.prefactor1[l]*(cart[0]*sph[self.yr[l-1,l-1]]-cart[1]*sph[self.yr_rev[l-1,l-1]]))
            sph=sph.at[self.yr_rev[l,l]].set(self.prefactor1[l]*(cart[0]*sph[self.yr_rev[l-1,l-1]]+cart[1]*sph[self.yr[l-1,l-1]]))
        return sph

'''
# here is an example to use the sph calculation

import timeit 
from jax.scipy.special import sph_harm
max_l=4
key=jrm.PRNGKey(0)
init_key=jrm.split(key)
cart=jrm.uniform(key,(3,1,1))
sph=SPH_CAL(max_l=max_l)
starttime = timeit.default_timer()
print("The start time is :",starttime)
sph.compute_sph(cart)
print("The time difference is :", timeit.default_timer() - starttime)
#cart=jrm.uniform(key,(3,1000,10000))
print("hello")
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp1=sph.compute_sph(cart)
print("The time difference is :", timeit.default_timer() - starttime)
print(tmp1)
print(cart[:,0,0])
r=jnp.linalg.norm(cart,axis=0)
r1=jnp.linalg.norm(cart[0:2],axis=0)
ceta=jnp.arccos(cart[2]/r)
phi=jnp.arccos(cart[0]/r1)
print("The start time is :",starttime)
tmp=sph_harm(m=jnp.array([1]), n=jnp.array([1]), theta=ceta[0],phi=phi[0])
print("The time difference is :", timeit.default_timer() - starttime)
'''
