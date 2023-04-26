import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrm
from jax import device_put
from flax import linen as nn
import src.model.MPNN as MPNN
import src.dataloader.dataloader as dataloader
import fortran.getneigh as getneigh
import optax
import time
from flax import serialization

#define the parameters
emb_nl=[16,16]
MP_nl=[64,64]
output_nl=[64,64]
nwave=8
max_l=2
MP_loop=2
cutoff=5.0
maxneigh=20
batchsize_train=32
batchsize_test=2*batchsize_train
dtype=jnp.float32
patience_epoch=100
decay_factor=0.5
force_table=True
if force_table==True:
    nprop=2
else:
    nprop=1
device="cpu"
floder="2.5e3/"
start_lr=2e-3
end_lr=1e-5
init_weight=[0.1,5.0]
final_weight=[0.1,0.1]

#generate the random number 
key=jrm.PRNGKey(0)
key=jrm.split(key)

device=jax.devices(device)
# generate the random cart to initialize the model.
cart=jax.device_put(jnp.array((np.random.rand(3,4))).astype(dtype),device[0])
atomindex=jnp.array([[0,0,1,1,2,3],[1,2,0,3,0,1]],dtype=jnp.int32)
shifts=jax.device_put(jnp.zeros((3,6),dtype=dtype),device[0])
species=jax.device_put(jnp.array([12,1,1,1]).reshape(-1,1),device[0])
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
    def squared_error(in_cart,in_atomindex,in_shifts,in_species):
        prediction=model(params,in_cart,in_atomindex,in_shifts,in_species)
        return prediction
    prediction=jax.vmap(squared_error)(cart,atomindex,shifts,species)
    lossprop=jnp.array([jnp.sum(jnp.square(iprediction-ilabel)) for iprediction, ilabel in zip(prediction, label)])
    loss=jnp.inner(lossprop,weight)
    return loss,lossprop

loss_grad_fn=jax.value_and_grad(get_loss,has_aux=True)


#Instantiate the dataloader
train_floder=floder+"train/"
test_floder=floder+"test/"
load_train=dataloader.DataLoader(maxneigh,batchsize_train,cutoff=cutoff,dier=cutoff/2.0,datafloder=train_floder,force_table=force_table,min_data_len=None,shuffle=True,Dtype=dtype,device=device[0])
load_test=dataloader.DataLoader(maxneigh,batchsize_test,cutoff=cutoff,dier=cutoff/2.0,datafloder=test_floder,force_table=force_table,min_data_len=None,shuffle=True,Dtype=dtype,device=device[0])
ntrain=jnp.array([load_train.numpoint,load_train.numpoint*3*5])
ntest=jnp.array([load_test.numpoint,load_test.numpoint*3*5])


init_weight=device_put(jnp.array(init_weight,dtype=dtype),device=device[0])
final_weight=device_put(jnp.array(final_weight,dtype=dtype),device=device[0])
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
         
        loss_test=jnp.zeros(1,dtype=dtype)
        lossprop_test=jnp.zeros(nprop,dtype=dtype)
        for data in load_test:
            cart,atomindex,shifts,species,label=data           
            loss,loss_prop=get_loss(params,cart,atomindex,shifts,species,label,weight)
            loss_test+=loss
            lossprop_test+=loss_prop
        if loss_test<bestloss:
            decay_epoch=0
            bestloss=loss_test
        else:
            decay_epoch+=1
        epoch+=1
        lossprop_train=jnp.sqrt(lossprop_train/ntrain)
        lossprop_test=jnp.sqrt(lossprop_test/ntest)
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
