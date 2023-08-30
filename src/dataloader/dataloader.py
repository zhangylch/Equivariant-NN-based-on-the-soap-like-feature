import jax
import jax.numpy as jnp
from jax import device_put
import numpy as np
import src.dataloader.read_data as read_data
import fortran.getneigh as getneigh

class DataLoader():
    '''
    This module are responsible for reading data from file, devide the total data into each batch and perform the calculation of neighbour list.
    maxneigh: int32/int64
        is used to define an array to hold the list of neighboring atoms. Therefore, it must be greater than the sum of the number of neighboring atoms of all central atoms.

    batchsize: int32/int64

    cutoff: float32/float64
        represents the cutoff radius for building the neighlist. Example: 4.0

    dier: float32/float64
        represents the length of cut box used in the cell-linked list algorithm. Typical value is equal to cutoff or half of cutoff.

    datafolder: string
        stores the path where save the dataset.

    force_table: True/False
        indicates if the forces are included in the training.

    shuffle: True/False
        indicates if shuffle the data in each epoch.

    Dtype: jnp.float32/jnp.float64

    device: jax.devices("cpu")/jax.devices("cuda")
    '''
    def __init__(self,maxneigh,batchsize,cutoff=5.0,dier=2.5,datafolder="train/",force_table=True,shuffle=True,Dtype=jnp.float32,device=jax.devices("cpu")):
        numpoint,coor,cell,species,numatoms,pot,force =  \
        read_data.Read_data(datafloder=datafloder,force_table=force_table,Dtype=Dtype)
        self.numpoint=numpoint
        self.numatoms=np.array(numatoms,dtype=jnp.int32)
        self.Dtype=Dtype
        self.device=device
        neighlist=[]
        shiftimage=[]
        coordinates=[]
        for i,icart in enumerate(coor):
            icell=cell[i]
            getneigh.init_neigh(cutoff,dier,icell)
            cart,atomindex,shifts,scutnum=getneigh.get_neigh(icart,maxneigh)
            getneigh.deallocate_all()
            neighlist.append(atomindex)
            shiftimage.append(shifts)
            coordinates.append(cart)
        self.image=np.array(coordinates,dtype=Dtype)
        self.neighlist=np.array(neighlist,dtype=jnp.int32)
        self.shiftimage=np.array(shiftimage,dtype=Dtype)
        if force_table:
            self.label=[np.array(pot,dtype=Dtype),np.array(force,dtype=Dtype)]
        else:   
            self.label=[np.array(pot,dtype=Dtype)]
        self.species=np.array(species,dtype=jnp.int32)
        self.batchsize=batchsize
        self.end=numpoint
        self.shuffle=shuffle               # to control shuffle the data
        if self.shuffle:
            self.shuffle_list=np.random.permutation(self.end)
        else:
            self.shuffle_list=np.arange(self.end)
        self.length=int(np.ceil(self.end/self.batchsize))
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.min_data:
            upboundary=min(self.end,self.ipoint+self.batchsize)
            index_batch=self.shuffle_list[self.ipoint:upboundary]
            coor=self.image[index_batch]
            shiftimage=self.shiftimage[index_batch]
            neighlist=self.neighlist[index_batch]
            abprop=(label[index_batch] for label in self.label)
            species=self.species[index_batch]
            self.ipoint+=self.batchsize
            return coor,neighlist,shiftimage,species,abprop
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=np.random.permutation(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
