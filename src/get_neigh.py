import fortran import fortran_neigh

def get_neigh(cart,cutoff):
    fortran_neigh.get_neigh(cart,atomindex,shifts)
    return atomindex,shifts
