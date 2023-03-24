module initmod
     implicit none
     integer(kind=4),parameter :: intype=4,typenum=8
     integer(kind=intype) :: interaction,max_neigh,numatom,length
     integer(kind=intype) :: nimage(3),rangebox(3)
     integer(kind=intype),allocatable :: index_rs(:,:,:),index_numrs(:,:,:,:,:)
     real(kind=typenum) :: rc,rcsq,dier
     real(kind=typenum) :: matrix(3,3),inv_matrix(3,3),rangecoor(3)
     real(kind=typenum),allocatable :: imageatom(:,:,:),shiftvalue(:,:)
end module

