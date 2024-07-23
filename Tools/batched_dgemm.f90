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
    TYPE(c_devptr), allocatable :: devptr(:)
    REAL(DP), DIMENSION(:,:), INTENT(IN) :: array
    INTEGER, INTENT(IN) :: batch_size
    INTEGER :: i

    !$acc declare deviceptr(array)
    #if defined(__CUDA)
    attributes(DEVICE) :: array
    #endif

    !$acc parallel loop present(devptr,array) default(present) vector_length(1) num_gangs(1)
    DO i = 1, batch_size
      devptr(i) = transfer(loc(array(1, 1)), devptr(i))
    END DO
    !$acc end parallel loop
  END SUBROUTINE broadcast_array_2D

  SUBROUTINE broadcast_array_3D(devptr, array, batch_size)
    TYPE(c_devptr), allocatable :: devptr(:)
    REAL(DP), DIMENSION(:,:,:), INTENT(IN) :: array
    INTEGER, INTENT(IN) :: batch_size
    INTEGER :: i
    !$acc declare deviceptr(array)
    #if defined(__CUDA)
    attributes(DEVICE) :: array
    #endif

   !$acc parallel loop present(devptr,array) default(present) vector_length(1) num_gangs(1)
    DO i = 1, batch_size
      devptr(i) = transfer(loc(array(1, 1, i)), devptr(i))
    END DO
   !$acc end parallel loop
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
    type(cublasHandle)      :: handle
    type(c_devptr), allocatable, dimension(:) :: devptr_A, devptr_B, devptr_C
  END TYPE DGEMM_BATCHED_STATE

  INTERFACE DGEMM_BATCHED_BROADCAST
    module PROCEDURE bcast_dgemm_2D_3D
    module PROCEDURE bcast_dgemm_3D_2D
    module PROCEDURE bcast_dgemm_2D_2D
    module PROCEDURE bcast_dgemm_3D_3D
  END INTERFACE

  contains

  subroutine init_dgemm_batched_broadcast(state, batch_count)
    implicit none
    type(DGEMM_BATCHED_STATE), intent(inout) :: state
    integer :: stat, batch_count
    
    stat = cublasCreate(state%handle)
    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS initialization error: ", stat
      stop
    end if

    allocate(state%devptr_A(batch_count))
    allocate(state%devptr_B(batch_count))
    allocate(state%devptr_C(batch_count))
    !$acc enter data create(state%devptr_A)
    !$acc enter data create(state%devptr_B)
    !$acc enter data create(state%devptr_C)
  end subroutine init_dgemm_batched_broadcast

  subroutine destroy_dgemm_batched_state(state)
    implicit none
    type(DGEMM_BATCHED_STATE), intent(in) :: state
    integer:: stat

    stat = cublasDestroy(state%handle)
    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS shutdown error: ", stat
    end if
    !$acc exit data delete(state%devptr_A)
    !$acc exit data delete(state%devptr_B)
    !$acc exit data delete(state%devptr_C)
    deallocate(state%devptr_A)
    deallocate(state%devptr_B)
    deallocate(state%devptr_C)
  end subroutine


  subroutine bcast_dgemm_2D_3D(state, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, batch_count)
    use cudafor
    use cublas
    use iso_c_binding
    implicit none
    
    integer, intent(in) :: ldA,ldB, ldC, m,n,k, batch_count
    REAL(DP), dimension(:,:)   :: A
    REAL(DP), dimension(:,:,:) :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)
    type(c_devptr), dimension(batch_count), device :: devptr_A, devptr_B, devptr_C


    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE) :: state
      
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

    CALL broadcast_array(state%devptr_A, A, batch_count)
    CALL broadcast_array(state%devptr_B, B, batch_count)
    CALL broadcast_array(state%devptr_C, C, batch_count)

    !$acc host_data use_device(state%devptr_A, state%devptr_B, state%devptr_C)
    devptr_A = state%devptr_A
    devptr_B = state%devptr_B
    devptr_C = state%devptr_C
    
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
    REAL(DP), dimension(:,:)   :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)
    type(c_devptr), dimension(batch_count), device :: devptr_A, devptr_B, devptr_C


    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE) :: state

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

    CALL broadcast_array(state%devptr_A, A, batch_count)
    CALL broadcast_array(state%devptr_B, B, batch_count)
    CALL broadcast_array(state%devptr_C, C, batch_count)

    !$acc host_data use_device(state%devptr_A, state%devptr_B, state%devptr_C)
    devptr_A = state%devptr_A
    devptr_B = state%devptr_B
    devptr_C = state%devptr_C
    
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
    REAL(DP), dimension(:,:)   :: A
    REAL(DP), dimension(:,:)   :: B 
    REAL(DP), dimension(:,:,:) :: C
    !$acc declare deviceptr(A,B,C)
    type(c_devptr), dimension(batch_count), device :: devptr_A, devptr_B, devptr_C


    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE) :: state    
  
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
  
    CALL broadcast_array(state%devptr_A, A, batch_count)
    CALL broadcast_array(state%devptr_B, B, batch_count)
    CALL broadcast_array(state%devptr_C, C, batch_count)

    !$acc host_data use_device(state%devptr_A, state%devptr_B, state%devptr_C)
    devptr_A = state%devptr_A
    devptr_B = state%devptr_B
    devptr_C = state%devptr_C

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
    type(c_devptr), dimension(batch_count), device :: devptr_A, devptr_B, devptr_C


    REAL(DP), intent(in) :: alpha, beta
    character(len=1), intent(in) :: transA, transB
    #if defined(__CUDA)
    attributes(DEVICE) :: A,B,C
    #endif
    integer :: stat, i
    integer :: cublasTransA, cublasTransB
    type(DGEMM_BATCHED_STATE):: state
  
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
  
    CALL broadcast_array(state%devptr_A, A, batch_count)
    CALL broadcast_array(state%devptr_B, B, batch_count)
    CALL broadcast_array(state%devptr_C, C, batch_count)

    !$acc host_data use_device(state%devptr_A, state%devptr_B, state%devptr_C)
    devptr_A = state%devptr_A
    devptr_B = state%devptr_B
    devptr_C = state%devptr_C

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

    if (stat /= CUBLAS_STATUS_SUCCESS) then
      print *, "CUBLAS error: ", stat
    end if
  end subroutine bcast_dgemm_3D_3D
  
END MODULE bcast_mul