subroutine get_neigh(coor,atomindex,shifts)
     use initmod
     implicit none
     integer(kind=intype) :: numatom,num,iatom,i,j,l,i1,i2,i3
     integer(kind=intype) :: sca(3),boundary(2,3)
     real(kind=typenum) :: oriminv(3),tmp(3),tmp1(3),boundary(3,3)
     real(kind=typenum) :: coor(3,numatom),fcoor(3,numatom),imageatom(3,numatom,length)
     real(kind=typenum) :: shifts(3,max_neigh*numatom),atomindex(2,max_neigh*numatom)
!f2py integer(kind=intype),intent(in) :: numatom
!f2py real(kind=typenum),intent(in),check(0==0) :: coor
!f2py integer(kind=intype),intent(out),check(0==0) :: atomindex
!f2py real(kind=typenum),intent(out),check(0==0) :: shifts
       fcoor=matmul(inv_matrix,coor)
! move all atoms to an cell which is convenient for the expansion of the image
       oriminv=coor(:,1)
       do iatom=2,numatom
         sca=nint(fcoor(:,iatom)-fcoor(:,1))
         coor(:,iatom)=coor(:,iatom)-sca(1)*scalmatrix(:,1)-sca(2)*scalmatrix(:,2)-sca(3)*scalmatrix(:,3)
         do j=1,atomdim
           if(coor(j,iatom)<oriminv(j)) oriminv(j)=coor(j,iatom)
         end do
       end do
       oriminv=oriminv-rc
       do iatom=1,numatom
         coor(:,iatom)=coor(:,iatom)-oriminv
       end do
! obatin image 
       do l=1,length
         do iatom=1,numatom
           imageatom(:,iatom,l)=coor(:,iatom)+shiftvalue(:,l)
         end do
       end do

       do l=1,length
         do iatom=1,numatom
           if(imageatom(1,iatom,l)>0d0 .and. imageatom(1,iatom,l)<rangecoor(1).and. imageatom(2,iatom,l)>0d0 .and.
           imageatom(2,iatom,l)<rangecoor(2)  &
           .and. imageatom(3,iatom,l)>0d0 .and. imageatom(3,iatom,l)<rangecoor(3)) then
             sca=ceiling(imageatom(:,iatom,l)/dier)
             index_numrs(:,sca(1),sca(2),sca(3))=[iatom,l] 
           end if
         end do
       end do
       scutnum=1
       ninit=(length+1)/2
       do iatom = 1, numatom
         sca=ceiling(coor(:,iatom)/dier)
         imageatom(:,iatom,ninit)=100.0
         ninit=(length+1)/2
         dire(:,natom,ninit)=100d0
         do i=1,3
           boundary(1,i)=max(1,sca(i)-interaction)
           boundary(2,i)=min(rangebox(i),sca(i)+interaction)
         end do
         do i3=boundary(1,3),boundary(2,3)
           do i2=boundary(1,2),boundary(2,2)
             do i1=boundary(1,1),boundary(2,1)
               j=index_numrs(1,i,i1,i2,i3)
               l=index_numrs(2,i,i1,i2,i3)
               tmp1=imageatom(:,j,l)-coor(:,iatom)
               tmp=dot_product(tmp1,tmp1)
               if(tmp<=rcsq) then
                 atomindex(:,scutnum)=[iatom-1,j-1]
                 shifts(:,scutnum)=shiftvalue(:,l)
                 scutnum=scutnum+1
               end if
             end do
           end do
         end do
         imageatom(:,natom,ninit)=coor(:,j)
       end do
     return
end subroutine
   
