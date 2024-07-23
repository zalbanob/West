module tpool_bindings
    use iso_c_binding
    implicit none
  
    ! Define a type that corresponds to the tpool_t structure in C
    type, bind(C):: tpool_t
    end type tpool_t

    ! Declare the bindings to the C functions
    interface
      ! tpool_create
      function tpool_create(num) bind(C, name="tpool_create")
        use iso_c_binding
        implicit none
        type(c_ptr) :: tpool_create
        integer(c_size_t), value :: num
      end function tpool_create
  
      ! tpool_destroy
      subroutine tpool_destroy(tp) bind(C, name="tpool_destroy")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: tp
      end subroutine tpool_destroy
  
      ! tpool_add_work
      function tpool_add_work(tp, func, arg) bind(C, name="tpool_add_work")
        use iso_c_binding
        implicit none
        logical(c_bool) :: tpool_add_work
        type(c_ptr), value :: tp
        type(c_funptr), value :: func
        type(c_ptr), value :: arg
      end function tpool_add_work
  
      ! tpool_wait
      subroutine tpool_wait(tp) bind(C, name="tpool_wait")
        use iso_c_binding
        implicit none
        type(c_ptr), value :: tp
      end subroutine tpool_wait
    end interface
  
  end module tpool_bindings
  