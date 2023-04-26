import jax
import jax.numpy as jnp
import flax.linen as nn
import src.model.MPNN as MPNN

class VmapMPNN(nn.Module):
    emb_nl: Sequence[int]
    MP_nl: Sequence[int]
    output_nl: Sequence[int]
    key: array=jrm.PRNGKey(0)
    force_table: bool=True
    nwave: int=8
    max_l: int=2
    MP_loop: int=2
    cutoff: float=5.0
    Dtype: dtype=dtype(jnp.float32)

    def setup():
         self.model=MPNN.MPNN(self.emb_nl,self.MP_nl,self.output_nl,key=self.key[0],nwave=self.nwave,max_l=self.max_l,MP_loop=self.MP_loop,self.cutoff=self.cutoff,Dtype=self.Dtype)
