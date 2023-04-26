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
        self.maxneigh=maxneigh
        self.cutoff=cutoff
        self.dier=dier
        self.Dtype=Dtype
        self.device=device
        self.image=jnp.array(coor,dtype=Dtype)
        if force_table:
            self.label=[jnp.array(pot,dtype=Dtype),jnp.array(force,dtype=Dtype)]
        else:   
            self.label=[jnp.array(pot,dtype=Dtype)]
        self.cell=jnp.array(cell,dtype=Dtype)
        self.species=jnp.array(species,dtype=jnp.int32)
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
            index_batch=self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)]
            coordinates=self.image[index_batch]
            abprop=(device_put(label[index_batch],device=self.device) for label in self.label)
            cell=self.cell[index_batch]
            species=self.species[index_batch]
            neighlist=[]
            shiftimage=[]
            coor=[]
            for i,cart in enumerate(coordinates):
                icell=cell[i]
                getneigh.init_neigh(self.cutoff,self.dier,icell)
                cart,atomindex,shifts,scutnum=getneigh.get_neigh(cart,self.maxneigh)
                getneigh.deallocate_all()
                neighlist.append(atomindex)
                shiftimage.append(shifts)
                coor.append(cart)
            neighlist=jnp.array(neighlist,dtype=jnp.int32)
            shiftimage=jnp.array(shiftimage,dtype=self.Dtype)
            self.ipoint+=self.batchsize
            coor=device_put(jnp.array(coor,dtype=self.Dtype),device=self.device)
            neighlist=device_put(neighlist,device=self.device)
            shiftimage=device_put(shiftimage,device=self.device)
            species=device_put(species,device=self.device)
            return coor,neighlist,shiftimage,species,abprop
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=np.random.permutation(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
