subroutine EANN_out(coor,atomindex,shifts)
     use initmod
     implicit none
     integer(kind=intype) :: numatom,num,maxneigh
     integer(kind=intype) :: i,i1,i2,i3,j,k,l,iatom
     integer(kind=intype) :: sca(3),rangebox(3),boundary(2,3)
     real(kind=typenum) :: oriminv(3),tmp(3),tmp1(3)
     real(kind=typenum) :: coor(3,numatom),fcoor(3,numatom),imageatom(3,numatom,length)
     real(kind=typenum) :: shifts(3,max_neigh*numatom),atomindex(2,max_neigh*numatom)
!f2py integer(kind=intype),intent(in) :: numatom
!f2py real(kind=typenum),intent(in),check(0==0) :: coor
!f2py integer(kind=intype),intent(out),check(0==0) :: atomindex
!f2py real(kind=typenum),intent(out),check(0==0) :: shifts
       fcoor=matmul(inv_matrix,coor)
! move all atoms to an cell which is convenient for the expansion of the image
       oriminv=coor(:,1)
       do i=2,numatom
         sca=nint(fcoor(:,i)-fcoor(:,1))
         coor(:,i)=coor(:,i)-sca(1)*scalmatrix(:,1)-sca(2)*scalmatrix(:,2)-sca(3)*scalmatrix(:,3)
         do j=1,atomdim
           if(coor(j,i)<oriminv(j)) oriminv(j)=coor(j,i)
         end do
       end do
       oriminv=oriminv-maxrc
       do i=1,numatom
         coor(:,i)=coor(:,i)-oriminv
       end do
! obatin image 
       l=0
       do i=-nimage(3),nimage(3)
         do j=-nimage(2),nimage(2)
           do k=-nimage(1),nimage(1)
             l=l+1
             do num=1,numatom
               imageatom(:,num,l)=cart(:,num)+shiftvalue(:,l)
             end do
           end do
         end do
       end do
       do num=1,length
         do i=1,numatom
           imageatom(:,i,num)=imageatom(:,i,num)-oriminv
           if(imageatom(1,i,num)>0d0 .and. imageatom(1,i,num)<rangecoor(1).and. imageatom(2,i,num)>0d0 .and. imageatom(2,i,num)<rangecoor(2)  &
           .and. imageatom(3,i,num)>0d0 .and. imageatom(3,i,num)<rangecoor(3)) then
             sca=ceiling(imageatom(:,i,num)/dier)
             index_numrs(:,sca(1),sca(2),sca(3))=[i-1,num-1] 
           end if
         end do
       end do
       scutnum=0
       ninit=(length+1)/2
       do iatom = 1, numatom
         sca=ceiling(coor(:,iatom)/dier)
         imageatom(:,iatom,ninit)=100.0
         ninit=(length+1)/2
         dire(:,natom,ninit)=100d0
         do i=1,3
           boundary(1,i)=max(1,sca(i)-interaction)
           boundary(2,i)=min(numrs(i),sca(i)+interaction)
         end do
         do i3=boundary(1,3),boundary(2,3)
           do i2=boundary(1,2),boundary(2,2)
             do i1=boundary(1,1),boundary(2,1)
               j=index_numrs(1,i,i1,i2,i3)+1
               num=index_numrs(2,i,i1,i2,i3)+1
               tmp1=dire(:,j,num)-coor(:,iatom)
               tmp=dot_product(tmp1,tmp1)
               if(tmp<=rcsq(ntype)) then
                 atomindex(:,scutnum)=[iatom-1,j-1]
                 shift(:,scutnum)=shiftvalue(:,l)
                 scutnum=scutnum+1
               end if
             end do
           end do
         end do
         imageatom(:,natom,ninit)=coor(:,j)
       end do
     return
end subroutine
   
