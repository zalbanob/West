
module thrust_kernels
    use iso_c_binding
    use cudafor
    implicit none

    public :: complex_saxpy
    public :: compute_power_sum
    public :: check_ovl_wannier_red

    integer, parameter        :: DP = kind(1.d0)

    interface
      subroutine check_ovl_wannier_red(orb_i, orb_j, size, result) bind(C, name="check_ovl_wannier_red")
          import :: c_double, c_size_t, c_ptr, c_devptr
          implicit none
          type(c_devptr), value :: orb_i, orb_j
          integer(c_size_t), value :: size
          type(c_ptr), value  :: result
        end subroutine check_ovl_wannier_red

        subroutine complex_saxpy(is_real, psic, aux, proj, result, dffts_nnr)
          use iso_c_binding
          use cudafor
          import :: DP
          implicit none
          logical                   :: is_real
          integer                   :: dffts_nnr
          type(c_ptr)      , value  :: psic
          type(c_devptr)   , value  :: aux
          type(c_devptr)   , value  :: proj
          type(c_ptr   )   , value  :: result
        end subroutine complex_saxpy
    end interface
  contains
  function compute_power_sum(d_array, size, constant) result(sum)
    use cudafor
    use iso_c_binding
    implicit none
    interface
        function compute_power_sum_internal(d_array, size, constant) bind(C, name="compute_power_sum")
          use cudafor
          use iso_c_binding
          implicit none
          type(c_devptr), value :: d_array
          integer(c_int), value :: size
          real(c_double), value :: constant
          real(c_double) :: compute_power_sum_internal
        end function compute_power_sum_internal
    end interface


    type(c_devptr), value :: d_array
    integer(c_int), value :: size
    real(8)       , value :: constant

    real(c_double) :: c_constant, c_sum
    real(8)        :: sum
    c_constant   = transfer(constant, c_constant)
    c_sum = compute_power_sum_internal(d_array, size, c_constant)        
    sum   = transfer(c_sum, sum)
  end function compute_power_sum
  
  subroutine complex_saxpy(is_real, psic, aux, proj, result, dffts_nnr)
    use iso_c_binding
    use cudafor
    implicit none
      interface
        subroutine complex_saxpy_internal(real_, psic, aux, proj, result, dffts_nnr) bind(C, name="complex_saxpy")
          use cudafor
          use iso_c_binding
          import :: DP
          implicit none
            logical(c_bool) , value :: real_
            integer(c_int)  , value :: dffts_nnr
            type(c_ptr)     , value :: psic
            type(c_devptr)  , value :: aux
            type(c_devptr)  , value :: proj
            type(c_ptr)     , value :: result
        end subroutine complex_saxpy_internal
      end interface
      logical                   :: is_real
      integer                   :: dffts_nnr
      type(c_ptr   )   , value  :: psic
      type(c_devptr)   , value  :: aux
      type(c_devptr)   , value  :: proj
      type(c_ptr   )   , value  :: result

      logical(c_bool)  :: c_real
      integer(c_int)   :: c_dffts_nnr

      integer          :: ptr

      c_real      = transfer(is_real, c_real)
      c_dffts_nnr = transfer(dffts_nnr, c_dffts_nnr)
      ptr         = transfer(psic, ptr)
      !print *,"BEFORE CALL"
      !print "(z16)", ptr
      call complex_saxpy_internal(c_real, psic , aux, proj, result, c_dffts_nnr)
  end subroutine complex_saxpy

end module thrust_kernels