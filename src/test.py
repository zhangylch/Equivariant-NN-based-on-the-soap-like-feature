import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
# here is an example to use the sph calculation
import timeit 

from sph_cal import *

max_l=4
key=jrm.PRNGKey(0)
init_key=jrm.split(key)
cart=jrm.uniform(key,(3,1000))
sph=SPH_CAL(max_l=max_l)

jax.lax.stop_gradient(cart)
forward=jax.jit(jax.vmap(sph_cal,in_axes=(1),out_axes=(1)))

