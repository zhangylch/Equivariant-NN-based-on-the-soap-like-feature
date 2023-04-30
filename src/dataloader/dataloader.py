import jax
import jax.numpy as jnp
from jax import device_put
import numpy as np
import src.dataloader.read_data as read_data
import fortran.getneigh as getneigh

class DataLoader():
    def __init__(self,maxneigh,batchsize,cutoff=5.0,dier=2.5,datafloder="train/",force_table=True,min_data_len=None,shuffle=True,Dtype=jnp.float32,device=jax.devices("cpu")):
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
        if not min_data_len:
            self.min_data=self.end
        else:
            self.min_data=min_data_len
        self.length=int(np.ceil(self.min_data/self.batchsize))
      
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
