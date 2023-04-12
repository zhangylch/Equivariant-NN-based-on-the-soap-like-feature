subroutine push( positions, velocities, a,  dt, m,n)
    integer, intent(in) :: n,m
    real(8), intent(in) :: dt
    real(8), dimension(3,n), intent(in) :: velocities
    real(8), dimension(3,n), intent(in) :: positions
    real(8), dimension(3,n) :: position1
    real(8), dimension(3,m), intent(out) :: a
    real(8), dimension(3,n) :: b
      position1=positions
      do i = 1, n
        position1(:,i) = position1(:,i) + dt*velocities(:,i)
      end do
      do i=1,m
         a(:,i)=position1(:,i)
      end do
      b=position1
end subroutine push
