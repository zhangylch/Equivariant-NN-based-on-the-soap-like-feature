import numpy as np
import jax
import math

# read system configuration and energy/force
def Read_data(datafloder="train/",force_table=None,Dtype=np.float32):
    coor=[]
    cell=[]
    pot=[] 
    force=None
    species=[]
    numatoms=[]
    #===================variable for force====================
    if force_table==1:
       force=[]
    numpoint=0
    num=0
    fname2=datafloder+'configuration'
    icell=np.zeros((3,3),dtype=Dtype)
    with open(fname2,'r') as f1:
        while True:
            string=f1.readline()
            if not string: break
            numatom=int(string)
            numatoms.append(numatom)
            string=f1.readline()
            # here to save the coordinate with row first to match the neighluist in fortran
            m=np.array(list(map(float,string.split())))
            icell[:,0]=m
            string=f1.readline()
            m=np.array(list(map(float,string.split())))
            icell[:,1]=m
            string=f1.readline()
            m=np.array(list(map(float,string.split())))
            icell[:,2]=m
            icoor=np.zeros((3,numatom),dtype=Dtype)
            ispecies=np.zeros((numatom,1),dtype=np.int32)
            if force_table==1: iforce=np.zeros((3,numatom),dtype=Dtype)
            for i in range(numatom):
                string=f1.readline()
                m=string.split()
                tmp=np.array(list(map(float,m[1:])))
                ispecies[num,0]=tmp[0]
                icoor[:,num]=tmp[1:4]
                if force_table: iforce[:,num]=-tmp[4:7]
            string=f1.readline()
            pot.append(float(string.split()[1]))
            numpoint+=1
            coor.append(icoor)
            cell.append(icell)
            species.append(ispecies)
            if force_table: force.append(iforce)
    return numpoint,coor,cell,species,numatoms,pot,force
