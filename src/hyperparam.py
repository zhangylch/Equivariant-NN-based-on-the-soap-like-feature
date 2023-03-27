# This modeule is created to define the hyperparameters for the MPNN

# The number of the radial function (n) and the maximal angular moment(L).
nradial=8
L=3
norbit=nradial*(L+1)
# the final nn structure
nl=[32,32,1]
# the nn structure for iterations
oc_nl=[32,32,norbit]
# the nn structure for embedded layer
emb_nl[16,16,nradial]
# The number of iterations of MPNN
oc_loop=3
# cutoff
cutoff=4.0
# Maximum atomic number of all structures in the data set
nn_list=[]
key=jrm.PRNGKey(0)
# create tge model for each iterations in MPNN
for ioc_loop in range(oc_loop):
    nn_list.append(MLP(oc_nl))
nn_list.append(MLP(nl))
init_key=jrm(key,num=oc_loop+3)  # The 3 more key is for the final nn, embedded nn and the seed to generate next key.
random_x=jnp.ones(norbit)
# initialize the model 
params_list=[]
for ioc_loop in range(oc_loop+1):
    params_list.append(nn_list[ioc_loop].init(init_key[ioc_loop],random_x))
#embeded nn
emb_nn=MLP(emb_nl)
params.append(emb_nn.init(init_key[oc_loop+1],jnp.ones(1)))

# define optimizable params for radial function
radial_key=jrm.split(init_key[oc_loop+2],num=3)
alpha=jrm.random.uniform(radial_key[0],(nradial,))
center=jrm.random.uniform(radial_key[1],(nradial,))*cutoff
params.append(alpha)
params.append(center)
