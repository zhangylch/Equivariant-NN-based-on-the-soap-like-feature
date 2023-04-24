import jax
import jax.numpy as jnp
import numpy as np
import dataloader.read_data as read_data
import fortran.getneigh as getneigh

class DataLoader():
    def __init__(self,maxneigh,batchsize,cutoff=5.0,in_dier=2.5,floder_list="train",force_table=True,min_data_len=None,shuffle=True,Dtype=jnp.flaot32,device=jax.devices("gpu")):
        numpoint,atom,species,numatoms,scalmatrix,period_table,coor,pot,force=  \
        read_data.Read_data(floder_list="test",force_table=force_table)
        self.maxneigh=maxneigh
        self.cutoff=cutoff
        self.in_dier=in_dier
        self.Dtype=Dtype
        self.image=jnp.array(coor,dtype=Dtype)
        self.label=[jnp.array(pot,dtype=Dtype),jnp.array(force,dtype=Dtype)]
        self.cell=jnp.array(scalmatrix,dtype=Dtype)
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
        #print(dist.get_rank(),self.length,self.end)
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.min_data:
            index_batch=self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)]
            coordinates=self.image.index_select(0,index_batch)
            abprop=(device_put(label.index_select(0,index_batch),device=self.device) for label in self.label)
            cell=self.cell.index_select(0,index_batch)
            species=self.species.index_select(0,index_batch)
            neighlist=[]
            shiftimage=[]
            for i,cart in enumerate(coordinates):
                icell=cell[i].transpose()
                cart=cart.transpose()
                getneigh.init_neigh(self.cutoff,self.in_dier,icell)
                atomindex,shifts,scutnum=getneigh.get_neigh(cart,self.maxneigh)
                getneigh.deallocate_all()
                neighlist.append(atomindex)
                shiftimage.append(shifts)
            neighlist=jnp.array(neighlist,dtype=jnp.int32)
            shiftimage=jnp.array(shiftimage,dtype=self.Dtype)
            self.ipoint+=self.batchsize
            #print(dist.get_rank(),self.ipoint,self.batchsize)
            return device_put(coordinates,device=self.device),device_put(neighlist,device=self.device),device_put(shiftimage,device=self.device),device_put(species,device=self.device),abprop
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=np.random.permutation(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
