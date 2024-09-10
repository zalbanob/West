
module thrust_kernels
    use iso_c_binding
    use cudafor
    implicit none

    public :: complex_saxpy
    public :: compute_power_sum
    public :: check_ovl_wannier_red
    public :: update_hg
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

      subroutine update_hg(vr_distr, ag, hg, nbnd_loc, ngk, l2_s, l2_e, kpt_pool_nloc, l1_e, nimage, idx, shape)
        use iso_c_binding
        use cudafor
        implicit none
        type(c_devptr), value :: vr_distr
        type(c_devptr), value :: ag
        type(c_devptr), value :: hg
        type(c_devptr), value :: nbnd_loc
        type(c_devptr), value :: ngk
        type(c_ptr)   , value :: shape
        integer, intent(in) :: l2_s, l2_e, kpt_pool_nloc, l1_e, nimage, idx
        integer(c_int)      :: c_l2_s, c_l2_e, c_kpt_pool_nloc, c_l1_e, c_nimage, c_idx
      end subroutine update_hg

      subroutine reduce_ag_bg(ag     , ag_shape, ag_ndim, &
                              bg     , bg_shape, bg_ndim, & 
                              c_distr, c_distr_shape, c_distr_ndim, &
                              nbnd_loc, ngk, l1_e, l2_s, l2_e, kpt_pool_nloc, nimage, idx, gstar)
        use cudafor
        use iso_c_binding
        implicit none
          type(c_devptr)        :: ag, bg, c_distr
          type(c_devptr)        :: nbnd_loc, ngk
          integer               :: l1_e
          integer               :: l2_s,l2_e
          integer               :: kpt_pool_nloc, nimage, idx, gstart
          integer               :: ag_ndim, bg_ndim, c_distr_ndim
          integer, target       :: ag_shape(*),  bg_shape(*), c_distr_shape(*)
      end subroutine reduce_ag_bg

      subroutine thrust_double_invfft_gamma_c(a1, a2, b, dfft_nl_d, dfft_nlm_d, n, batch_size, nx, nnr) bind(C, name="thrust_double_invfft_gamma_c")
        use cudafor
        use iso_c_binding
        implicit none
        type(c_devptr), value :: a1, a2, b, dfft_nl_d, dfft_nlm_d
        integer(c_int), value :: n, batch_size, nx, nnr
      end subroutine thrust_double_invfft_gamma_c

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
      call complex_saxpy_internal(c_real, psic , aux, proj, result, c_dffts_nnr)
  end subroutine complex_saxpy

  subroutine update_hg(vr_distr, ag, hg, nbnd_loc, ngk, l2_s, l2_e, kpt_pool_nloc, l1_e, nimage, idx, shape)
    use iso_c_binding
    use cudafor
    implicit none
    interface
      subroutine update_hg_internal(vr_distr, ag, hg, nbnd_loc, ngk, l2_s, l2_e, kpt_pool_nloc, l1_e, nimage, idx, shape) bind(C, name="update_hg")
          use cudafor
          implicit none
          type(c_devptr), value :: vr_distr
          type(c_devptr), value :: ag
          type(c_devptr), value :: hg
          type(c_devptr), value :: nbnd_loc
          type(c_devptr), value :: ngk
          type(c_ptr)   , value :: shape

          integer(c_int), value, intent(in) :: l2_s
          integer(c_int), value, intent(in) :: l2_e
          integer(c_int), value, intent(in) :: kpt_pool_nloc
          integer(c_int), value, intent(in) :: l1_e
          integer(c_int), value, intent(in) :: nimage
          integer(c_int), value, intent(in) :: idx
          end subroutine update_hg_internal
    end interface

    
    type(c_devptr), value :: vr_distr
    type(c_devptr), value :: ag
    type(c_devptr), value :: hg
    type(c_devptr), value :: nbnd_loc
    type(c_devptr), value :: ngk
    type(c_ptr)   , value :: shape

    integer, intent(in) :: l2_s, l2_e, kpt_pool_nloc, l1_e, nimage, idx
    integer(c_int)      :: c_l2_s, c_l2_e, c_kpt_pool_nloc, c_l1_e, c_nimage, c_idx

    c_l2_s          = transfer(l2_s,c_l2_s)
    c_l2_e          = transfer(l2_e,c_l2_e)
    c_kpt_pool_nloc = transfer(kpt_pool_nloc,c_kpt_pool_nloc)
    c_l1_e          = transfer(l1_e,c_l1_e)
    c_nimage        = transfer(nimage,c_nimage)
    c_idx           = transfer(idx,c_idx)
    call update_hg_internal(vr_distr, ag, hg, nbnd_loc, ngk, c_l2_s, c_l2_e, c_kpt_pool_nloc, c_l1_e, c_nimage, c_idx, shape)
  end subroutine update_hg


  subroutine thrust_double_invfft_gamma(a1, a2, b, dfft_nl_d, dfft_nlm_d, n, batch_size, nx, nnr)
    use iso_c_binding
    use cudafor
    implicit none
    type(c_devptr), value :: a1, a2, b
    type(c_devptr), value :: dfft_nl_d, dfft_nlm_d
    integer, value :: n, batch_size, nx, nnr
    integer(c_int) :: c_n, c_batch_size, c_nx, c_nnr
    c_n          = transfer(n, c_n) 
    c_batch_size = transfer(batch_size, c_batch_size) 
    c_nx         = transfer(nx , c_nx) 
    c_nnr        = transfer(nnr, c_nnr) 
    call thrust_double_invfft_gamma_c(a1, a2, b, dfft_nl_d, dfft_nlm_d, c_n, c_batch_size, c_nx, c_nnr)
  end subroutine thrust_double_invfft_gamma
  

  subroutine reduce_ag_bg(ag     , ag_shape, ag_ndim, &
                          bg     , bg_shape, bg_ndim, & 
                          c_distr, c_distr_shape, c_distr_ndim, &
                          nbnd_loc, ngk, l1_e, l2_s, l2_e, kpt_pool_nloc, nimage, idx, gstart)
    use cudafor
    use iso_c_binding
    implicit none

    interface
      subroutine reduce_ag_bg_internal(ag, ag_shape, ag_ndim, bg, bg_shape, bg_ndim, &
                          c_distr, c_distr_shape, c_distr_ndim, nbnd_loc, ngk, &
                          l1_e, l2_s, l2_e, kpt_pool_nloc, nimage, idx, gstart) &
                          bind(C, name="reduce_ag_bg")
        use cudafor
        use iso_c_binding
        implicit none
          type(c_devptr), value :: ag
          type(c_ptr), value :: ag_shape
          integer(c_int), value :: ag_ndim
          type(c_devptr), value :: bg
          type(c_ptr), value :: bg_shape
          integer(c_int), value :: bg_ndim
          type(c_devptr), value :: c_distr
          type(c_ptr), value :: c_distr_shape
          integer(c_int), value :: c_distr_ndim
          type(c_devptr), value :: nbnd_loc
          type(c_devptr), value :: ngk
          integer(c_int), value :: l1_e
          integer(c_int), value :: l2_s
          integer(c_int), value :: l2_e
          integer(c_int), value :: kpt_pool_nloc
          integer(c_int), value :: nimage
          integer(c_int), value :: idx
          integer(c_int), value :: gstart
      end subroutine reduce_ag_bg_internal
    end interface

    type(c_devptr) :: ag, bg, c_distr
    integer, target :: ag_shape(*), bg_shape(*), c_distr_shape(*)
    integer :: ag_ndim , bg_ndim , c_distr_ndim
    type(c_devptr) :: nbnd_loc, ngk
    integer :: l1_e
    integer :: l2_s, l2_e
    integer :: kpt_pool_nloc
    integer :: nimage
    integer :: idx
    integer :: gstart

    integer(c_int) :: c_l1_e, c_l2_s, c_l2_e, c_kpt_pool_nloc, c_nimage, c_idx, c_gstart
    integer(c_int) :: c_ag_ndim, c_bg_ndim, c_c_distr_ndim

    c_l1_e          = transfer(l1_e, c_l1_e)
    c_l2_s          = transfer(l2_s, c_l2_s)
    c_l2_e          = transfer(l2_e, c_l2_e)
    c_kpt_pool_nloc = transfer(kpt_pool_nloc, c_kpt_pool_nloc)
    c_nimage        = transfer(nimage, c_nimage)
    c_idx           = transfer(idx, c_idx)
    c_gstart        = transfer(gstart, c_gstart)
    c_ag_ndim       = transfer(ag_ndim, c_ag_ndim)
    c_bg_ndim       = transfer(bg_ndim, c_bg_ndim)
    c_c_distr_ndim  = transfer(c_distr_ndim, c_c_distr_ndim)

    call reduce_ag_bg_internal(ag, c_loc(ag_shape), c_ag_ndim, &
              bg, c_loc(bg_shape), c_bg_ndim, &
              c_distr, c_loc(c_distr_shape), c_c_distr_ndim, &
              nbnd_loc, ngk, &
              c_l1_e, c_l2_s, c_l2_e, c_kpt_pool_nloc, c_nimage, c_idx, c_gstart)

end subroutine reduce_ag_bg

end module thrust_kernels