import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial

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
        self.pt=np.empty((self.max_l,self.max_l))
        self.yr=np.empty((self.max_l,self.max_l))
        self.yr_rev=np.empty((self.max_l,self.max_l))
        num_lm=(self.max_l+1)*self.max_l/2
        self.coeff_a=np.empty(num_lm)
        self.coeff_b=np.empty(num_lm)
        tmp=jnp.arange(self.max_l)
        ls=tmp*tmp
        for l in range(self.max_l):
            self.pt[l,0:l+1]=tmp[0:l+1]+l*(l+1)/2
            # here the self.yr and self.yr_rev have overlap in m=0.
            self.yr[l,0:l+1]=ls[l]+l+tmp[0:l+1]
            self.yr_rev[l,0:l+1]=ls[l]+l-tmp[0:l+1]
            if l>0.5:
                self.coeff_a[self.pt[l,0:l]]=np.sqrt((4.0*ls[l]-1)/(ls[l]-ls[0:l]))
                self.coeff_b[self.pt[l,0:l]]=-np.sqrt((ls[l-1]-ls[0:l])/(4*(ls[l-1]-1)))
        self.sqrt2_rev=np.sqrt(1/2.0)

    def __call__(self,cart):
        return self.compute_sph(cart)

    @partial(jit,static_augnums=0)
    def compute_sph(self,incart,distances)
        '''
        cart: Cartesian coordinates with the dimension (batch,n,3) n is the max number of neigbbors and the rest complemented with 0. Here, we do not do the lod of tensor to keep the dimension of batch for the convenient calculation of sample to sample gradients.
        '''
        cart=incart.transpose(2,1,0)
        d_sq=distance*distance
        temp=np.sqrt(0.5/np.pi)
        sph=jnp.empty((self.max_l*self.max_l,incart.shape[1],incart.shape[0]))
        sph.at[0].set(temp)
        sph.at[1].set(np.sqrt(3)*temp*cart[2])
        sph.at[2].set(-np.sqrt(3/2)*temp*cart[0])
        sph.at[3].set(-np.sqrt(3/2)*temp*cart[1])
        for l in range(2,self.max_l):
            sph.at[self.yr[l,0]].set(self.sqrt2_rev*self.coeff_a[self.pt[l,0]]*(cart[2]*sph[self.yr[l-1,0]]+self.coeff_b[self.pt[l,0]]*d_sq*sph[self.yr[l-2,0]]))
            for m in range(1,l-1):
                sph.at[self.yr[l,m]].set(self.coeff_a[self.pt[l,m]]*(cart[2]*sph[self.yr[l-1,m]]+self.coeff_b[self.pt[l,m]]*d_sq*sph[self.yr[l-2,m]]))
                sph.at[self.yr_rev[l,m]].set(self.coeff_a[self.pt[l,m]]*(cart[2]*sph[self.yr_rev[l-1,m]]+self.coeff_b[self.pt[l,m]]*d_sq*sph[self.yr_rev[l-2,m]]))
            sph.at[self.yr[l,l-1]].set(np.sqrt(2*l+1)*cart[2]*sph[self.yr[l-1,l-1]])
            sph.at[self.yr_rev[l,l-1]].set(np.sqrt(2*l+1)*cart[2]*sph[self.yr_rev[l-1,l-1]])
            sph.at[self.yr[l,l]].set(-np.sqrt(1.0+1.0/2.0/l)*(cart[0]*sph[self.yr[l-1,l-1]]-cart[1]*sph[self.yr_rev[l-1,l-1]]))
            sph.at[self.yr_rev[l,l]].set(-np.sqrt(1.0+1.0/2.0/l)*(cart[0]*sph[self.yr_rev[l-1,l-1]]+cart[1]*sph[self.yr[l-1,l-1]]))
        return sph

