import numpy as np
import math

# read system configuration and energy/force
def Read_data(floderlist=["train","validation"],force_table=None):
    coor=[]
    scalmatrix=[]
    abprop=[] 
    force=None
    atom=[]
    species=[]
    numatoms=[]
    period_table=[]
    # tmp variable
    #===================variable for force====================
    if force_table==1:
       force=[]
    numpoint=[0 for _ in range(len(floderlist))]
    num=0
    for ifloder,floder in enumerate(floderlist):
        fname2=floder+'configuration'
        with open(fname2,'r') as f1:
            while True:
                string=f1.readline()
                if not string: break
                string=f1.readline()
                scalmatrix.append([])
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()))
                scalmatrix[num].append(m)
                string=f1.readline()
                m=list(map(float,string.split()[1:4]))
                period_table.append(m)
                coor.append([])
                species.append([])
                atom.append([])
                if force_table==1: force.append([])
                while True:
                    string=f1.readline()
                    m=string.split()
                    if m[0]=="abprop:":
                        abprop.append(list(map(float,m[1:2])))
                        break
                    if not force_table:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        species[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                    else:
                        atom[num].append(m[0]) 
                        tmp=list(map(float,m[1:]))
                        species[num].append(tmp[0])
                        coor[num].append(tmp[1:4])
                        force[num].append(tmp[4:7])
                numpoint[ifloder]+=1
                numatoms.append(len(atom[num]))
                num+=1
    return numpoint,atom,species,numatoms,scalmatrix,period_table,coor,abprop,force
