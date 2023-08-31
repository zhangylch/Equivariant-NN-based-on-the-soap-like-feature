import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype,array
import flax.linen as nn
from typing import Sequence
from src.low_level import MLP, sph_cal
from flax.core import freeze



class MPNN(nn.Module):

    '''
    This MPNN module is designed for the calculation of equivariant message passing neural network for both periodic and molecular systems.
    Not only the invairant feature but also the eqivariant feature (sperical harmonic) are passed and progressively refined. 
    parameters:

    emb_nl: list
        defines the nn structure (only the hidden layers) of embedding neural network. Examples: [16,16]

    MP_nl: list
        defines the nn structure (only the hidden layers) used in each step except the last step of MPNN. Examples: [32,32]

    output_nl: list
        defines the nn structures of the last step of the MPNN (only the hidden layer), which will output the desired quantities, such as atomic energies or electron wave functions. Example: [64,64]

    nwave: int32/int64
         represents the number of gaussian radial functions. Example: 8

    max_l: int32/int64
         represents the maximal angular quantum numebr for the evaluation of spherical harmonic. Example: 2

    MP_loop: int32/int64
         represents the number of passing message in MPNN. Example: 3

    cutoff: float32/float64
         represents the cutoff radius for evaluating the local descriptors. Example: 4.0

    Dtype: jnp.float32/jnp.float64
         represents the datatype in this module. Example: jnp.float32
    '''  

    emb_nl: Sequence[int]
    MP_nl: Sequence[int]
    output_nl: Sequence[int]
    nwave: int=8
    max_l: int=2
    MP_loop: int=2
    cutoff: float=5.0
    Dtype: dtype=dtype(jnp.float32)
    

    def setup(self):
        self.r_max_l=self.max_l+1
        self.norbit=self.nwave*self.r_max_l

        # define the class for the calculation of spherical harmonic expansion
        self.sph_cal=sph_cal.SPH_CAL(max_l=self.max_l,Dtype=self.Dtype)
        # the first time is slow for the compile of the jit
        self.sph_cal(jnp.ones(3))
        
        # define the embedded layer used to convert the atomin number of a coefficients
        self.emb_nn=MLP.MLP(self.emb_nl,self.nwave*3)
        self.emb_params=self.param("emb_params",self.emb_nn.init,(jnp.ones(1)))

        # used for the convenient summation over the same l
        self.index_l=jnp.array([0],dtype=jnp.int32)
        for l in range(1,self.r_max_l):
            self.index_l=jnp.hstack((self.index_l,jnp.ones((2*l+1,),dtype=jnp.int32)*l))

        # Instantiate the NN class for the MPNN
        # create tge model for each iterations in MPNN
        self.MPNN_list=[MLP.MLP(self.MP_nl,self.nwave) for iMP_loop in range(self.MP_loop)]
        self.outnn=MLP.MLP(self.output_nl,1)
        random_x=jnp.ones(self.norbit)
        # initialize the model 
        self.MP_params_list=[self.param("MPNN_"+str(iMP_loop)+"_params",self.MPNN_list[iMP_loop].init,(random_x)) for iMP_loop in range(self.MP_loop)]
        self.out_params=self.param("out_params",self.outnn.init,(random_x))
        #embeded nn

       

    def __call__(self,cart,atomindex,shifts,species):
        '''
        cart: jnp.float32/jnp.float64.
            represents the cartesian coordinates of systems with dimension 3*Natom. Natom is the number of atoms in the system.

        atomindx: jnp.int64/inp.int32
            stores the index of centeral atoms and theirs corresponding neighbor atoms with the dimension 2*Neigh. Neigh is the number of total neighbor atoms in the system.

        shifts: jnp.float32/jnp.float64
            stores the offset corresponding to each neighbor atoms with the dimension Neigh*3.

        species: jnp.float32/jnp.float64
            represents the atomic number of each center atom with the dimension Natom.
        '''
        coor=cart[:,atomindex[1]]-cart[:,atomindex[0]]+shifts
        distances=jnp.linalg.norm(coor,axis=0)
        emb_coeff=self.emb_nn.apply(self.emb_params,species)
        expand_coeff=emb_coeff[atomindex[1]]
        coefficients=expand_coeff[:,:self.nwave]
        alpha=expand_coeff[:,self.nwave:2*self.nwave]
        center=expand_coeff[:,2*self.nwave:]
        radial=self.gaussian(distances,alpha,center)
        radial_cutoff=self.cutoff_func(distances)
        sph=self.sph_cal(coor/self.cutoff)
        MP_sph=jnp.zeros((cart.shape[0],sph.shape[0],self.nwave),dtype=cart.dtype)
        density=jnp.zeros((cart.shape[1],self.r_max_l,self.nwave),dtype=cart.dtype)
        for inn, nn in enumerate(self.MPNN_list):
            equi_feature = jnp.einsum("i,ij,ij,ki -> ikj",radial_cutoff,radial,coefficients,sph)
            density,MP_sph = self.density(equi_feature,radial_cutoff,atomindex[1],atomindex[0],MP_sph,density=density)
            coefficients = nn.apply(self.MP_params_list[inn],density.reshape(-1,self.norbit))[atomindex[1]]
        density,MP_sph=self.density(equi_feature,radial_cutoff,atomindex[1],atomindex[0],MP_sph,density=density)
        output=self.outnn.apply(self.out_params,density.reshape(-1,self.norbit))
        return output.reshape(-1)

    def density(self,equi_feature,radial_cutoff,index_neigh,index_center,MP_sph,density=jnp.zeros((0))):
        '''
        The method is used to calculate the density from the neigh euqivariant feature.
        '''
        r_sph=jnp.einsum("ijk,i -> ijk",MP_sph[index_neigh],radial_cutoff)+equi_feature # out of bound will be ingored
        sum_sph=jnp.zeros((density.shape[0],MP_sph.shape[1],MP_sph.shape[2]),dtype=self.Dtype)
        sum_sph=sum_sph.at[index_center].add(r_sph)
        contract_sph=jnp.square(sum_sph)
        density=density.at[:,self.index_l].add(contract_sph)
        return density,sum_sph

    def gaussian(self,distances,alpha,center):
        '''
        gaussian radial functions
        '''
        shift_distances=alpha*(distances[:,None]-center)
        gaussian=jnp.exp(-shift_distances*shift_distances)
        return gaussian
    
    def cutoff_func(self,distances):
        '''
        cutoff function:
        '''
        tmp=(jnp.cos(distances/self.cutoff*jnp.pi)+1.0)/2.0
        return tmp*tmp*tmp  # here to use the a^3 to keep the smooth of hessian functtion

