import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
import src.model.MPNN as MPNN
import src.dataloader.dataloader as dataloader
import fortran.getneigh as getneigh
import optax
import time


#define the parameters
emb_nl=[16,16]
MP_nl=[64,64]
output_nl=[64,64]
nwave=8
max_l=2
MP_loop=2
cutoff=5.0
batchsize_train=32
batchsize_test=2*batchsize_train
dtype=dtype(jnp.float32)
patience_epoch=100
decay_factor=0.5
force_table=True
if force_table==True:
    nprop=2
else:
    nprop=1
device="cpu"
floder="/data/home/scv2201/run/zyl/data/ch4/2.5e3/"
init_weight=[1.0,5.0]
final_weight=[1.0,0.5]

#generate the random number 
key=jrm.PRNGKey(0)
key=jrm.split(key)

device=jax.devices(device)
# generate the random cart to initialize the model.
cart=jax.device_put((np.random.rand(3,2)).astype(dtype),device)
atomindex=jnp.array([[0,1],[1,0]],dtype=jnp.int32)
shifts=jax.device_put(jnp.zeros((3,2),dtype=dtype),device)
species=jax.device(jnp.array([12,1,1,1,1]).reshape(-1,1),device)
model=MPNN.MPNN(emb_nl,MP_nl,output_nl,key=key[0],nwave=nwave,max_l=max_l,MP_loop=MP_loop,cutoff=cutoff,Dtype=dtype)
params=model.init(key[0],cart,atomindex,shifts,species)
if force_table is not None:
    model=jax.value_and_grad(model,argnums=1)
model=jax.jit(jax.vmap(Prop_cal,in_axes=0,output_axes=0),argnums=0,device=device)

# define the dataloader

# define the loss function
@jax.jit
def get_loss(params,cart,atomindex,shifts,species,label,weight):
    prediction=model(params,cart,atomindex,shifts,species)
    lossprop=jnp.concatenate([jnp.sum(jnp.square(iprediction-ilabel)) for iprediction, ilabel in zip(prediction, label)])
    loss=jnp.inner(lossprop,weight)
    return loss,lossprop

loss_grad_fn=jax_value_grad(get_loss,has_aux=True)


#Instantiate the dataloader
train_floder=floder+"train"
test_floder=floder+"test"
load_train=dataloader.DataLoader(maxneigh,batchsize_train,cutoff=cutoff,in_dier=cutoff/2.0,floder_list=train_floder,force_table=force_table,min_data_len=None,shuffle=True,Dtype=dtype,device=device)
load_test=dataloader.DataLoader(maxneigh,batchsize_test,cutoff=cutoff,in_dier=cutoff/2.0,floder_list=test_floder,force_table=force_table,min_data_len=None,shuffle=True,Dtype=dtype,device=device)

init_weight=jnp.array(init_weight,dtype=dtype)
final_weight=jnp.array(final_weight,dtype=dtype)
bestloss=1e20
epoch=0
weight=init_weight
lr=start_lr
ferr=open("nn.err","w")
ferr.write("Equivariant MPNN package based on three-body descriptors")
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
            (loss,lossprop),grads=loss_grad_fn(params,*data,weight)
            updates,opt_state=optim.update(grads,opt_state)
            params=optax.apply_updates(params,updates)
            loss_train+=loss
            lossprop_train+=lossprop

        loss_test=jnp.zeros(1,dtype=dtype)
        lossprop_test=jnp.zeros(nprop,dtype=dtype)
        for data in load_test:
            loss,loss_prop=get_loss(params,*data)
            loss_test+=loss
            lossprop_test+=loss
        if loss_test<bestloss:
            decay_epoch=0
            bestloss=loss_test
        else:
            decay_epoch+=1
        epoch+=1
        #output the error 
        ferr.write("Epoch= {:6},  lr= {:5e}  ".format(epoch,lr))
        ferr.write("train: ")
        for error in lossprop_train:
            ferr.write("{:10e} ".format(error))
        ferr.write(" test: ")
        for error in lossprop_test:
            ferr.write("{:10e} ".format(error))
        ferr.write(" \n")
    lr=lr*decay_factor
    # update the learning rate but keep track of the optim state, so we do not reinitialize the state of optim
    optim=optax.adam(learning_rate=lr)
    weight=(init_weight-final_weight)*(lr-end_lr)/(start_lr-end_lr)+final_weight
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
