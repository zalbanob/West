MODULE broadcast_module
  USE iso_c_binding, ONLY: c_devptr
  USE cudafor
  USE cublas
  USE kinds,         ONLY : DP
  IMPLICIT NONE

  INTERFACE broadcast_array
    MODULE PROCEDURE broadcast_array_2D
    MODULE PROCEDURE broadcast_array_3D
  END INTERFACE
CONTAINS

  SUBROUTINE broadcast_array_2D(devptr, array, batch_size)
    TYPE(c_devptr), DIMENSION(:), INTENT(INOUT) :: devptr
    REAL(DP), DIMENSION(:,:), INTENT(IN) :: array
    INTEGER, INTENT(IN) :: batch_size
    INTEGER :: i

    TYPE(c_devptr) :: array_devptr
    integer(c_intptr_t) :: i8

    !$acc declare deviceptr(array)
    #if defined(__CUDA)
    attributes(DEVICE) :: array
    #endif

    !$acc host_data
    DO i = 1, batch_size
      devptr(i) = transfer(loc(array(1, 1)), devptr(i))
    END DO
    !$acc end host_data
  END SUBROUTINE broadcast_array_2D

  SUBROUTINE broadcast_array_3D(devptr, array, batch_size)
    TYPE(c_devptr), DIMENSION(:), INTENT(INOUT) :: devptr
    REAL(DP), DIMENSION(:,:,:), INTENT(IN) :: array
    INTEGER, INTENT(IN) :: batch_size
    INTEGER :: i
    !$acc declare deviceptr(array)
    #if defined(__CUDA)
    attributes(DEVICE) :: array
    #endif

    !$acc host_data
    DO i = 1, batch_size
      devptr(i) = transfer(loc(array(1, 1, i)), devptr(i))
    END DO
    !$acc end host_data
  END SUBROUTINE broadcast_array_3D

END MODULE broadcast_module


module bcast_mul
  use cudafor
  use cublas
  use iso_c_binding
  use broadcast_module
  USE kinds,                 ONLY : DP
  IMPLICIT NONE

  TYPE DGEMM_BATCHED_STATE
    type(cublasHandle):: handle
  END TYPE DGEMM_BATCHED_STATE

  INTERFACE DGEMM_BATCHED_BROADCAST
    module PROCEDURE bcast_dgemm_2D_3D
    module PROCEDURE bcast_dgemm_3D_2D
    module PROCEDURE bcast_dgemm_2D_2D
    module PROCEDURE bcast_dgemm_3D_3D
  END INTERFACE

  contains

  SUBROUTINE init_dgemm_batched_broadcast(state)
    TYPE(DGEMM_BATCHED_STATE), INTENT(OUT) :: state
    integer :: stat
    stat = cublasCreate(state%handle)
    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS initialization error: ", stat
      stop
    end if
  END SUBROUTINE init_dgemm_batched_broadcast

  subroutine destroy_dgemm_batched_state(state)
    TYPE(DGEMM_BATCHED_STATE), INTENT(IN) :: state
    integer:: stat
    stat = cublasDestroy(state%handle)
    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS shutdown error: ", stat
    end if
  end subroutine

  recursive subroutine bcast_dgemm_2D_3D(state, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, batch_count)
    use cudafor
    use cublas
    use iso_c_binding
    implicit none
    
    integer, intent(in) :: ldA,ldB, ldC, m,n,k, batch_count
    REAL(DP), dimension(:,:)   :: A
    REAL(DP), dimension(:,:,:) :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)

    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    type(c_devptr), dimension(batch_count) :: devptr_A, devptr_B, devptr_C
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE), INTENT(in):: state
  
    if (transA == 'N') then
      cublasTransA = CUBLAS_OP_N
    else if (transA == 'T') then
      cublasTransA = CUBLAS_OP_T
    else
      print *, "Invalid transA value"
      stop
    end if
  
    if (transB == 'N') then
      cublasTransB = CUBLAS_OP_N
    else if (transB == 'T') then
      cublasTransB = CUBLAS_OP_T
    else
      print *, "Invalid transB value"
      stop
    end if

    !$acc data create(devptr_A, devptr_B, devptr_C)
    CALL broadcast_array(devptr_A, A, batch_count)
    CALL broadcast_array(devptr_B, B, batch_count)
    CALL broadcast_array(devptr_C, C, batch_count)
    !$acc update device(devptr_A, devptr_B, devptr_C)
    
    !$acc host_data use_device(devptr_A, devptr_B, devptr_C)
    stat = cublasDgemmBatched(        &
          state%handle,               &
          cublasTransA, cublasTransB, &
          m, n, k,                    &
          alpha,                      &
          devptr_A, ldA,              &
          devptr_B, ldB,              &
          beta,                       &
          devptr_C, ldC,              &
          batch_count)
    !$acc end host_data
    !$acc end data


    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS error: ", stat
    end if
  end subroutine bcast_dgemm_2D_3D

  subroutine bcast_dgemm_3D_2D(state, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, batch_count)
    use cudafor
    use cublas
    use iso_c_binding
    implicit none
    interface
      subroutine checkGpuMem() bind(C, name="check_gpu_mem")
      end subroutine checkGpuMem
    end interface
    
    integer, intent(in) :: ldA,ldB, ldC, m,n,k, batch_count
    REAL(DP), dimension(:,:,:) :: A
    REAL(DP), dimension(:,:) :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)

    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    type(c_devptr), dimension(batch_count) :: devptr_A, devptr_B, devptr_C
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE), INTENT(in):: state

    if (transA == 'N') then
      cublasTransA = CUBLAS_OP_N
    else if (transA == 'T') then
      cublasTransA = CUBLAS_OP_T
    else
      print *, "Invalid transA value"
      stop
    end if
  
    if (transB == 'N') then
      cublasTransB = CUBLAS_OP_N
    else if (transB == 'T') then
      cublasTransB = CUBLAS_OP_T
    else
      print *, "Invalid transB value"
      stop
    end if
    !call checkGpuMem()

    !$acc data create(devptr_A, devptr_B, devptr_C)
    CALL broadcast_array(devptr_A, A, batch_count)
    CALL broadcast_array(devptr_B, B, batch_count)
    CALL broadcast_array(devptr_C, C, batch_count)
    !$acc update device(devptr_A, devptr_B, devptr_C)
    
    !$acc host_data use_device(devptr_A, devptr_B, devptr_C)
    stat = cublasDgemmBatched(        &
          state%handle,               &
          cublasTransA, cublasTransB, &
          m, n, k,                    &
          alpha,                      &
          devptr_A, ldA,              &
          devptr_B, ldB,              &
          beta,                       &
          devptr_C, ldC,              &
          batch_count)
    !$acc end host_data
    !$acc end data

    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS error: ", stat
    end if
  end subroutine bcast_dgemm_3D_2D

  subroutine bcast_dgemm_2D_2D(state,transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, batch_count)
    use cudafor
    use cublas
    use iso_c_binding
    implicit none
    
    integer, intent(in) :: ldA,ldB, ldC, m,n,k, batch_count
    REAL(DP), dimension(:,:) :: A
    REAL(DP), dimension(:,:) :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)

    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    type(c_devptr), dimension(batch_count) :: devptr_A, devptr_B, devptr_C
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE), INTENT(in):: state
    
  
    if (transA == 'N') then
      cublasTransA = CUBLAS_OP_N
    else if (transA == 'T') then
      cublasTransA = CUBLAS_OP_T
    else
      print *, "Invalid transA value"
      stop
    end if
  
    if (transB == 'N') then
      cublasTransB = CUBLAS_OP_N
    else if (transB == 'T') then
      cublasTransB = CUBLAS_OP_T
    else
      print *, "Invalid transB value"
      stop
    end if
  
    !$acc data create(devptr_A, devptr_B, devptr_C)
    CALL broadcast_array(devptr_A, A, batch_count)
    CALL broadcast_array(devptr_B, B, batch_count)
    CALL broadcast_array(devptr_C, C, batch_count)
    !$acc update device(devptr_A, devptr_B, devptr_C)

    !$acc host_data use_device(devptr_A, devptr_B, devptr_C)
    stat = cublasDgemmBatched(        &
          state%handle,               &
          cublasTransA, cublasTransB, &
          m, n, k,                    &
          alpha,                      &
          devptr_A, ldA,              &
          devptr_B, ldB,              &
          beta,                       &
          devptr_C, ldC,              &
          batch_count)
    !$acc end host_data
    !$acc end data


    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS error: ", stat
    end if
  end subroutine bcast_dgemm_2D_2D

  subroutine bcast_dgemm_3D_3D(state,transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, batch_count)
    use cudafor
    use cublas
    use iso_c_binding
    implicit none
    
    integer, intent(in) :: ldA,ldB, ldC, m,n,k, batch_count
    REAL(DP), dimension(:,:,:) :: A
    REAL(DP), dimension(:,:,:) :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)

    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    type(c_devptr), dimension(batch_count) :: devptr_A, devptr_B, devptr_C
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE), INTENT(in):: state
  
    if (transA == 'N') then
      cublasTransA = CUBLAS_OP_N
    else if (transA == 'T') then
      cublasTransA = CUBLAS_OP_T
    else
      print *, "Invalid transA value"
      stop
    end if
  
    if (transB == 'N') then
      cublasTransB = CUBLAS_OP_N
    else if (transB == 'T') then
      cublasTransB = CUBLAS_OP_T
    else
      print *, "Invalid transB value"
      stop
    end if
  
    !$acc data create(devptr_A, devptr_B, devptr_C)
    CALL broadcast_array(devptr_A, A, batch_count)
    CALL broadcast_array(devptr_B, B, batch_count)
    CALL broadcast_array(devptr_C, C, batch_count)
    !$acc update device(devptr_A, devptr_B, devptr_C)
    print *, devptr_A
    stop
    
    !$acc host_data use_device(devptr_A, devptr_B, devptr_C)
    stat = cublasDgemmBatched(        &
          state%handle,               &
          cublasTransA, cublasTransB, &
          m, n, k,                    &
          alpha,                      &
          devptr_A, ldA,              &
          devptr_B, ldB,              &
          beta,                       &
          devptr_C, ldC,              &
          batch_count)
    !$acc end host_data
    !$acc end data

    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS error: ", stat
    end if
  end subroutine bcast_dgemm_3D_3D
  
END MODULE bcast_mul