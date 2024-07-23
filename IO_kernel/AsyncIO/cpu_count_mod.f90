module cpu_count_mod
    use iso_c_binding, only: c_long
    implicit none
    interface
        function get_cpu_count() bind(C, name="get_cpu_count")
            import c_long
            implicit none
            integer(c_long) :: get_cpu_count
        end function get_cpu_count
    end interface

end module cpu_count_mod