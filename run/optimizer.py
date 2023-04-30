import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrm
from jax import device_put
from flax import linen as nn
import src.model.MPNN as MPNN
import src.dataloader.dataloader as dataloader
import src.dataloader.cpu_gpu as torch_load
import fortran.getneigh as getneigh
import optax
import time
from flax import serialization
import torch

#define the parameters
emb_nl=[16,16]
MP_nl=[64,64]
output_nl=[64,64]
nwave=8
max_l=2
MP_loop=2
cutoff=5.0
maxneigh=16000
batchsize_train=256
batchsize_val=2*batchsize_train
dtype=jnp.float32
patience_epoch=100
decay_factor=0.5
force_table=True
queue_size=5
if force_table==True:
    nprop=2
else:
    nprop=1
cpu_gpu="gpu"
floder="../data/H2O/"
start_lr=2e-3
end_lr=1e-5
init_weight=[0.1,5.0]
final_weight=[0.1,0.1]

#generate the random number 
key=jrm.PRNGKey(0)
key=jrm.split(key)

device=jax.devices(cpu_gpu)
if cpu_gpu=="gpu":
    torch_device= torch.device("cuda")
else:
    torch_device= torch.device("cpu")

#Instantiate the dataloader
train_floder=floder+"train/"
val_floder=floder+"validation/"
load_train=dataloader.DataLoader(maxneigh,batchsize_train,cutoff=cutoff,dier=cutoff/2.0,datafloder=train_floder,force_table=force_table,min_data_len=None,shuffle=True,Dtype=dtype,device=device[0])
load_val=dataloader.DataLoader(maxneigh,batchsize_val,cutoff=cutoff,dier=cutoff/2.0,datafloder=val_floder,force_table=force_table,min_data_len=None,shuffle=False,Dtype=dtype,device=device[0])
ntrain=[load_train.numpoint]
nval=[load_val.numpoint]
if force_table:
    ntrain.append(jnp.sum(load_train.numatoms)*3)
    nval.append(jnp.sum(load_val.numatoms)*3)
load_train=torch_load.CudaDataLoader(load_train, torch_device, queue_size=queue_size)
load_val=torch_load.CudaDataLoader(load_val, torch_device, queue_size=queue_size)


ntrain=jnp.array(ntrain,dtype=dtype)
nval=jnp.array(nval,dtype=dtype)

# generate the random cart to initialize the model.
cart=jnp.array((np.random.rand(3,4))).astype(dtype)
atomindex=jnp.array([[0,0,1,1,2,3],[1,2,0,3,0,1]],dtype=jnp.int32)
shifts=jnp.zeros((3,6),dtype=dtype)
species=jnp.array([12,1,1,1]).reshape(-1,1)
model=MPNN.MPNN(emb_nl,MP_nl,output_nl,key=key[0],nwave=nwave,max_l=max_l,MP_loop=MP_loop,cutoff=cutoff,Dtype=dtype)
params=model.init(key[0],cart,atomindex,shifts,species)
if force_table:
    model=jax.value_and_grad(model.apply,argnums=1)
else:
    model=model.apply
#model=jax.jit(nn.vmap(model,variable_axes={'params': None},split_rngs={'params': False},in_axes=0,out_axes=0),static_argnums=0,device=device[0])

# define the dataloader

# define the loss function
#@jax.jit
def get_loss(params,cart,atomindex,shifts,species,label,weight):
    def predict(in_cart,in_atomindex,in_shifts,in_species):
        prediction=model(params,in_cart,in_atomindex,in_shifts,in_species)
        return prediction
    vmapmodel=jax.vmap(predict,in_axes=0,out_axes=0)     
    prediction=vmapmodel(cart,atomindex,shifts,species)
    lossprop=jnp.array([jnp.sum(jnp.square(iprediction-ilabel)) for iprediction, ilabel in zip(prediction, label)])
    loss=jnp.inner(lossprop,weight)
    return loss,lossprop

loss_grad_fn=jax.value_and_grad(get_loss,has_aux=True)

#jax.config.update("jax_debug_nans", True)

init_weight=jnp.array(init_weight,dtype=dtype)
final_weight=jnp.array(final_weight,dtype=dtype)
bestloss=1e20
epoch=0
weight=init_weight
lr=start_lr
ferr=open("nn.err","w")
ferr.write("Equivariant MPNN package based on three-body descriptors \n")
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
#manually do learning rate decay(similar as the reducelronplateau in torch)
# define the optimizer
optim=optax.adam(learning_rate=lr)
opt_state=optim.init(params)
while True:
    if start_lr<end_lr: break
    decay_epoch=0
    while True:
        if epoch>patience_epoch: break 
        loss_train=jnp.zeros(1,dtype=dtype)
        lossprop_train=jnp.zeros(nprop,dtype=dtype)
        for data in load_train:
            cart,atomindex,shifts,species,label=data           
            (loss,lossprop),grads=loss_grad_fn(params,cart,atomindex,shifts,species,label,weight)
            updates,opt_state=optim.update(grads,opt_state)
            params=optax.apply_updates(params,updates)
            loss_train+=loss
            lossprop_train+=lossprop
         
        loss_val=jnp.zeros(1,dtype=dtype)
        lossprop_val=jnp.zeros(nprop,dtype=dtype)
        for data in load_val:
            cart,atomindex,shifts,species,label=data           
            loss,loss_prop=get_loss(params,cart,atomindex,shifts,species,label,weight)
            loss_val+=loss
            lossprop_val+=loss_prop
        if loss_val<bestloss:
            decay_epoch=0
            bestloss=loss_val
        else:
            decay_epoch+=1
        epoch+=1
        lossprop_train=jnp.sqrt(lossprop_train/ntrain)
        lossprop_val=jnp.sqrt(lossprop_val/nval)
        #output the error 
        ferr.write("Epoch= {:6},  lr= {:5e}  ".format(epoch,lr))
        ferr.write("train: ")
        for error in lossprop_train:
            ferr.write("{:10e} ".format(error))
        ferr.write(" validation: ")
        for error in lossprop_val:
            ferr.write("{:10e} ".format(error))
        ferr.write(" \n")
        ferr.flush()
    lr=lr*decay_factor
    # update the learning rate but keep track of the optim state, so we do not reinitialize the state of optim
    optim=optax.adam(learning_rate=lr)
    weight=(init_weight-final_weight)*(lr-end_lr)/(start_lr-end_lr)+final_weight
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
