import jax
import jax.numpy as jnp
from flax import linen as nn

class optimizer():

    def __init__(self,MPNN,get_loss,optim,lr_scheduler,dataloader):
        self.value_and_grad=jax.value_and_grad(self.get_loss)
        self.optim=optim
        self.lr_scheduler=lr_scheduler
        self.dataloader_train=dataloader
        self.nprop

    def __call__(self,):
        lossprop=jnp.zeros(self.nprop
        while True:
             for data in self.dataloader_train:
                 if lr < self.lr_scheduler.end_lr: break
                 nnprop=self.MPNN(cart,atomindex,shifts,species)
                 loss=self.value_and_grad(self.get_loss(nnprop,abprop))
                 lossprop+=lossprop+loss

    def get_loss(cart,atomindex,shifts,species,abprop):
        nnprop=self.MPNN(cart,atomindex,shifts,species)
