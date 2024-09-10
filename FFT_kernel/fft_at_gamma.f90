!
! Copyright (C) 2015-2024 M. Govoni
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! This file is part of WEST.
!
! Contributors to this file:
! Marco Govoni
!
!-----------------------------------------------------------------------
MODULE fft_at_gamma
  !-----------------------------------------------------------------------
  !
  ! Everything is done following dffts
  !
  USE kinds,                ONLY : DP
  USE fft_interfaces,       ONLY : fwfft,invfft
  USE fft_types,            ONLY : fft_type_descriptor
#if defined(__CUDA)
  USE west_gpu,             ONLY : dfft_nl_d,dfft_nlm_d
#endif
  !
  IMPLICIT NONE
  !
  COMPLEX(DP),PARAMETER :: z_0 = (0._DP,0._DP)
  COMPLEX(DP),PARAMETER :: z_i = (0._DP,1._DP)
  !
  CONTAINS
  !
  SUBROUTINE double_invfft_gamma_streamed(id1, dfft,n,nx,a1,a2,b,cdriver)
   use nvtx
   IMPLICIT NONE
   !
   ! I/O
   !
   TYPE(fft_type_descriptor),INTENT(IN) :: dfft
   INTEGER,INTENT(IN)      :: n,nx
   INTEGER, INTENT(IN)     :: id1
   COMPLEX(DP),INTENT(IN)  :: a1(nx)
   COMPLEX(DP),INTENT(IN)  :: a2(nx)
   COMPLEX(DP),INTENT(OUT) :: b(dfft%nnr)
   #if defined(__CUDA)
   ATTRIBUTES(DEVICE) :: a1,a2,b
   #endif
   CHARACTER(LEN=*),INTENT(IN) :: cdriver
   ! Workspace
   INTEGER :: ig
   !$acc kernels
   b(:) = z_0
   !$acc end kernels
   !$acc parallel loop
   DO ig = 1,n
      b(dfft_nl_d(ig)) = a1(ig)+z_i*a2(ig)
      b(dfft_nlm_d(ig)) = CONJG(a1(ig)-z_i*a2(ig))
   ENDDO
   !$acc end parallel
   CALL invfft_y_gpu(cdriver,b,dfft)
 END SUBROUTINE

  !SUBROUTINE double_invfft_gamma_streamed(id1, dfft,n,nx,a1,a2,b,cdriver)
  !  USE openacc
  !  USE cudafor
  !  IMPLICIT NONE
  !  
  !  TYPE(fft_type_descriptor),INTENT(IN) :: dfft
  !  INTEGER,INTENT(IN) :: n,nx, id1
  !  INTEGER:: id2
  !  COMPLEX(DP),INTENT(IN) :: a1(nx), a2(nx)
  !  COMPLEX(DP),INTENT(OUT) :: b(dfft%nnr)
  !  CHARACTER(LEN=*),INTENT(IN) :: cdriver
  !  INTEGER :: ig
  !  ATTRIBUTES(DEVICE) :: a1,a2,b
!
  !  INTEGER(KIND=cuda_stream_kind) :: stream1, stream2
  !  INTEGER :: istat
  !  
  !  istat = cudaStreamCreate(stream1)
  !  istat = cudaStreamCreate(stream2)
  !  id2 = id1 + 1
  !  
  !  ! Set OpenACC CUDA streams
  !  CALL acc_set_cuda_stream(id1, stream1)
  !  CALL acc_set_cuda_stream(id2, stream2)
  !  
  !  ! Initialize b to zero
  !  !$acc kernels async(1)
  !  b(:) = z_0
  !  !$acc end kernels
  !  
  !  !$acc parallel loop async(id1)
  !  DO ig = 1, n/2
  !    b(dfft_nl_d(ig)) = a1(ig) + z_i*a2(ig)
  !  END DO
  !  !$acc end parallel loop
 !
  !  !$acc parallel loop async(id2)
  !  DO ig = n/2 + 1, n
  !    b(dfft_nl_d(ig)) = a1(ig) + z_i*a2(ig)
  !    b(dfft_nlm_d(ig)) = CONJG(a1(ig) - z_i*a2(ig))
  !  END DO
  !  !$acc end parallel loop
 !
  !  !$acc wait(id1)
  !  !$acc wait(id2)
 !
  !  CALL invfft_y_gpu(cdriver, b, dfft, 1, stream1)
 !
  !  istat = cudaStreamDestroy(stream1)
  !  istat = cudaStreamDestroy(stream2)
!
  !END SUBROUTINE
  !
 SUBROUTINE batched_double_invfft_gamma(dfft, n, nx, batch_size, a1, a2, b, cdriver)
   !
   ! Batched INVFFT : G ---> R
   !
   ! INPUT  : n         = actual number of PW
   !          nx        = maximum number of PW
   !          batch_size= number of functions to process in parallel
   !          a1        = COMPLEX array containing batch_size COMPLEX functions in G space
   !          a2        = COMPLEX array containing batch_size COMPLEX functions in G space
   ! OUTPUT : b         = COMPLEX array containing batch_size*2 REAL functions in R space
   !
   use nvtx
   IMPLICIT NONE
   !
   ! I/O
   !
   TYPE(fft_type_descriptor), INTENT(IN) :: dfft
   INTEGER, INTENT(IN) :: n, nx, batch_size
   COMPLEX(DP), INTENT(IN) :: a1(nx, batch_size)
   COMPLEX(DP), INTENT(IN) :: a2(nx, batch_size)
   COMPLEX(DP), INTENT(OUT) :: b(dfft%nnr, batch_size)
   #if defined(__CUDA)
   ATTRIBUTES(DEVICE) :: a1, a2, b
   #endif
   CHARACTER(LEN=*), INTENT(IN) :: cdriver
   !
   ! Workspace
   !
   INTEGER :: ig, ibatch
   !
   !!$acc kernels
   !b(:,:) = z_0
   !!$acc end kernels
   !

   !DO ibatch = 1, batch_size
   !  !$acc parallel loop async(ibatch)
   !   DO ig = 1, n
   !    b(dfft_nl_d(ig), ibatch) = a1(ig, ibatch) + z_i * a2(ig, ibatch)
   !    b(dfft_nlm_d(ig), ibatch) = CONJG(a1(ig, ibatch) - z_i * a2(ig, ibatch))
   !  END DO
   !  !$acc end parallel
   !END DO
   !!$acc wait
   !
   call nvtxStartRange("batched_invfft")
   DO ibatch = 1, batch_size
     CALL invfft(cdriver, b(:, ibatch), dfft)
   END DO
   call nvtxEndRange
 END SUBROUTINE

  SUBROUTINE double_invfft_gamma(dfft,n,nx,a1,a2,b,cdriver)
    !
    ! INVFFT : G ---> R
    !
    ! INPUT  : n     = actual number of PW
    !          nx    = maximum number of PW
    !          a1    = COMPLEX array containing ONE COMPLEX function in G space
    !          a2    = COMPLEX array containing ONE COMPLEX function in G space
    ! OUTPUT : b     = ONE COMPLEX array containing TWO REAL functions in R space
    !
    use nvtx
    IMPLICIT NONE
    !
    ! I/O
    !
    TYPE(fft_type_descriptor),INTENT(IN) :: dfft
    INTEGER,INTENT(IN) :: n,nx
    COMPLEX(DP),INTENT(IN) :: a1(nx)
    COMPLEX(DP),INTENT(IN) :: a2(nx)
    COMPLEX(DP),INTENT(OUT) :: b(dfft%nnr)
    #if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: a1,a2,b
    #endif
    CHARACTER(LEN=*),INTENT(IN) :: cdriver
    !
    ! Workspace
    !
    INTEGER :: ig
    !
    #if defined(__CUDA)
    !$acc kernels
    b(:) = z_0
    !$acc end kernels
    !
    !$acc parallel loop
    DO ig = 1,n
       b(dfft_nl_d(ig)) = a1(ig)+z_i*a2(ig)
       b(dfft_nlm_d(ig)) = CONJG(a1(ig)-z_i*a2(ig))
    ENDDO
    !$acc end parallel
    #else
    !$OMP PARALLEL PRIVATE(ig)
    !$OMP DO
    DO ig = 1,dfft%nnr
       b(ig) = z_0
    ENDDO
    !$OMP ENDDO
    !$OMP DO
    DO ig = 1,n
       b(dfft%nl (ig)) =       a1(ig) + z_i * a2(ig)
       b(dfft%nlm(ig)) = CONJG(a1(ig) - z_i * a2(ig))
    ENDDO
    !$OMP ENDDO
    !$OMP END PARALLEL
    #endif
    !
    call nvtxStartRange("invfft")
    CALL invfft(cdriver,b,dfft)
    call nvtxEndRange
    !
  END SUBROUTINE
  !
  SUBROUTINE double_fwfft_gamma(dfft,n,nx,a,b1,b2,cdriver)
    !
    ! FWFFT : R ---> G
    !
    ! INPUT  : n     = actual number of PW
    !          nx    = maximum number of PW
    !          a     = ONE COMPLEX array containing TWO REAL functions in R space
    ! OUTPUT : b1    = ONE COMPLEX array containing ONE COMPLEX function in G space
    !          b2    = ONE COMPLEX array containing ONE COMPLEX function in G space
    !
    IMPLICIT NONE
    !
    ! I/O
    !
    TYPE(fft_type_descriptor),INTENT(IN) :: dfft
    INTEGER,INTENT(IN) :: n,nx
    COMPLEX(DP),INTENT(INOUT) :: a(dfft%nnr)
    COMPLEX(DP),INTENT(OUT) :: b1(nx)
    COMPLEX(DP),INTENT(OUT) :: b2(nx)
    #if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: a,b1,b2
    #endif
    CHARACTER(LEN=*),INTENT(IN) :: cdriver
    !
    ! Workspace
    !
    INTEGER :: ig
    COMPLEX(DP) :: fm,fp
    !
    CALL fwfft(cdriver,a,dfft)
    !
    ! Keep only G>=0
    !
    #if defined(__CUDA)
    !$acc parallel loop
    DO ig = 1,n
       fp = (a(dfft_nl_d(ig))+a(dfft_nlm_d(ig)))*0.5_DP
       fm = (a(dfft_nl_d(ig))-a(dfft_nlm_d(ig)))*0.5_DP
       b1(ig) = CMPLX(REAL(fp,KIND=DP),AIMAG(fm),KIND=DP)
       b2(ig) = CMPLX(AIMAG(fp),-REAL(fm,KIND=DP),KIND=DP)
    ENDDO
    !$acc end parallel
    !
    IF(nx > n) THEN
       !$acc kernels
       b1(n+1:nx) = z_0
       b2(n+1:nx) = z_0
       !$acc end kernels
    ENDIF
    #else
    !$OMP PARALLEL PRIVATE(ig,fp,fm)
    !$OMP DO
    DO ig = 1,n
       fp = ( a(dfft%nl (ig)) + a(dfft%nlm(ig)) )*0.5_DP
       fm = ( a(dfft%nl (ig)) - a(dfft%nlm(ig)) )*0.5_DP
       b1(ig) = CMPLX(REAL(fp,KIND=DP), AIMAG(fm), KIND=DP)
       b2(ig) = CMPLX(AIMAG(fp), -REAL(fm,KIND=DP), KIND=DP)
    ENDDO
    !$OMP ENDDO
    !$OMP END PARALLEL
    !
    DO ig = (n+1),nx
       b1(ig) = z_0
       b2(ig) = z_0
    ENDDO
    #endif
    !
  END SUBROUTINE
  !
  SUBROUTINE single_invfft_gamma(dfft,n,nx,a1,b,cdriver)
    !
    ! INVFFT : G ---> R
    !
    ! INPUT  : n     = actual number of PW
    !          nx    = maximum number of PW
    !          a1    = ONE COMPLEX arrays containing ONE COMPLEX functions in G space
    ! OUTPUT : b     = ONE COMPLEX array containing ONE REAL functions in R space + 0
    !
    IMPLICIT NONE
    !
    ! I/O
    !
    TYPE(fft_type_descriptor),INTENT(IN) :: dfft
    INTEGER,INTENT(IN) :: n,nx
    COMPLEX(DP),INTENT(IN) :: a1(nx)
    COMPLEX(DP),INTENT(OUT) :: b(dfft%nnr)
    #if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: a1,b
    #endif
    CHARACTER(LEN=*),INTENT(IN) :: cdriver
    !
    ! Workspace
    !
    INTEGER :: ig
    !
    #if defined(__CUDA)
    !$acc kernels
    b(:) = z_0
    !$acc end kernels
    !
    !$acc parallel loop
    DO ig = 1,n
       b(dfft_nl_d(ig)) = a1(ig)
       b(dfft_nlm_d(ig)) = CONJG(a1(ig))
    ENDDO
    !$acc end parallel
    #else
    !$OMP PARALLEL PRIVATE(ig)
    !$OMP DO
    DO ig = 1,dfft%nnr
       b(ig) = z_0
    ENDDO
    !$OMP ENDDO
    !$OMP DO
    DO ig = 1,n
       b(dfft%nl (ig)) =       a1(ig)
       b(dfft%nlm(ig)) = CONJG(a1(ig))
    ENDDO
    !$OMP ENDDO
    !$OMP END PARALLEL
    #endif
    !
    CALL invfft(cdriver,b,dfft)
    !
  END SUBROUTINE
  !
  SUBROUTINE single_fwfft_gamma(dfft,n,nx,a,b1,cdriver)
    !
    ! FWFFT : R ---> G
    !
    ! INPUT  : n     = actual number of PW
    !          nx    = maximum number of PW
    !          a     = ONE COMPLEX array containing ONE REAL functions in R space + 0
    ! OUTPUT : b1    = ONE COMPLEX array containing ONE COMPLEX functions in G space
    !
    IMPLICIT NONE
    !
    ! I/O
    !
    TYPE(fft_type_descriptor),INTENT(IN) :: dfft
    INTEGER,INTENT(IN) :: n,nx
    COMPLEX(DP),INTENT(INOUT) :: a(dfft%nnr)
    COMPLEX(DP),INTENT(OUT) :: b1(nx)
    #if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: a,b1
    #endif
    CHARACTER(LEN=*),INTENT(IN) :: cdriver
    !
    ! Workspace
    !
    INTEGER :: ig
    !
    CALL fwfft(cdriver,a,dfft)
    !
    ! Keep only G>=0
    !
    #if defined(__CUDA)
    !$acc parallel loop
    DO ig = 1,n
       b1(ig) = a(dfft_nl_d(ig))
    ENDDO
    !$acc end parallel
    !
    IF(nx > n) THEN
       !$acc kernels
       b1(n+1:nx) = z_0
       !$acc end kernels
    ENDIF
    #else
    !$OMP PARALLEL PRIVATE(ig)
    !$OMP DO
    DO ig=1,n
       b1(ig) = a(dfft%nl(ig))
    ENDDO
    !$OMP ENDDO
    !$OMP END PARALLEL
    !
    DO ig = (n+1),nx
       b1(ig) = z_0
    ENDDO
    #endif
    !
  END SUBROUTINE
  !
END MODULE
