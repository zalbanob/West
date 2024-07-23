MODULE pdep_async_io
    use tpool_bindings
    use cpu_count_mod
    USE kinds,         ONLY : DP,i8b
    USE mp_global,     ONLY : me_bgrp,root_bgrp,nproc_bgrp,intra_bgrp_comm
    USE mp_wave,       ONLY : mergewf
    USE mp_world,      ONLY : mpime,root
    USE westcom,       ONLY : npwq,npwq_g,npwqx,ngq,ngq_g,igq_q
    USE gvect,         ONLY : ig_l2g
    USE control_flags, ONLY : gamma_only
    USE base64_module, ONLY : islittleendian
    USE west_io,       ONLY : HD_LENGTH,HD_VERSION,HD_ID_VERSION,HD_ID_LITTLE_ENDIAN,HD_ID_DIMENSION
    use iso_c_binding
    implicit none

    TYPE:: PDEP_AIO_TASK
        COMPLEX(DP)  , pointer , dimension(:)  :: data 
        INTEGER      , pointer , dimension(:)  :: header 
        character(:) , pointer                 :: fname
    END TYPE PDEP_AIO_TASK

    TYPE :: PDEP_AIO_STATE
        type(c_ptr)       :: pool
        integer(c_size_t) :: num_threads
    END TYPE PDEP_AIO_STATE

    CONTAINS

    function pdep_aio_init(num_threads) result(state)
        use iso_fortran_env, only: int64
        integer(c_size_t)    ,intent(in), optional :: num_threads
        integer(c_size_t)                          :: actual_threads
        type(PDEP_AIO_STATE), intent(out)          :: state
        integer(int64) :: num_io_pool_workers
        character(len=10) :: env_value
        integer :: len, status

        if ( me_bgrp == root_bgrp ) then
            if (present(num_threads) ) then
                If (num_threads > 0) then 
                    actual_threads = num_threads 
                else
                    actual_threads = get_cpu_count()
                endif
            else
                env_value = " "
                call get_environment_variable("NUM_IO_WORKERS", env_value, len, status, .true.)
                if (status == 0) then
                    read(env_value, *) num_io_pool_workers
                    actual_threads = int(num_io_pool_workers, kind=c_size_t)
                else
                    print *, "[WARN]: NUM_IO_WORKERS environment variable not set. Using default value."
                    actual_threads = get_cpu_count() 
                end if
            endif

            state%pool = tpool_create(actual_threads)
            if (c_associated(state%pool)) then
                print "(A, I3, A)", "Thread pool created with ", actual_threads, " Threads"
              else
                print *, "[ERROR]: Failed to create thread pool"
                stop
            end if
            state%num_threads = actual_threads
        endif
    end function pdep_aio_init
    
    SUBROUTINE pdep_merge_and_write_G_async(aio_state, fname, pdepg, iq)
        use nvtx
        implicit none
        integer                            :: stat
        type(PDEP_AIO_STATE),  intent(in)  :: aio_state
        type(PDEP_AIO_TASK),   pointer     :: task
        type(c_ptr),           target      :: task_ptr
        type(c_funptr),        target      :: func
        logical(c_bool)                    :: success

        CHARACTER(LEN=*),INTENT(IN) :: fname
        COMPLEX(DP),INTENT(IN)      :: pdepg(npwqx)
        INTEGER,INTENT(IN),OPTIONAL :: iq

        INTEGER :: ig
        INTEGER :: ndim
        INTEGER :: npwqx_g
        INTEGER,ALLOCATABLE :: igq_l2g_kdip(:)
        INTEGER,ALLOCATABLE :: igq_l2g(:)
        INTEGER,PARAMETER :: default_iq = 1
        INTEGER :: iq_
        call nvtxStartRange("Submitting IO Write")

        IF(PRESENT(iq)) THEN
           iq_ = iq
        ELSE
           iq_ = default_iq
        ENDIF

        allocate(task, stat=stat)
        if (stat /= 0) stop "Allocation failed for task"
        allocate(task%data(npwq_g), stat=stat)
        if (stat /= 0) stop "Allocation failed for data"
        task%data = 0.0_DP
        
        IF(.NOT. gamma_only) THEN
           ndim = ngq_g(iq_)
           npwqx_g = MAXVAL(ngq_g)
           ALLOCATE(igq_l2g_kdip(npwqx_g))
           igq_l2g_kdip = 0
           ALLOCATE(igq_l2g(ngq(iq_)))
           DO ig = 1,ngq(iq_)
              igq_l2g(ig) = ig_l2g(igq_q(ig,iq_))
           ENDDO
           CALL gq_l2gmap_kdip(npwq_g,ngq_g(iq_),ngq(iq_),igq_l2g,igq_l2g_kdip)
           DEALLOCATE(igq_l2g)
           CALL mergewf(pdepg,task%data,npwq,igq_l2g_kdip,me_bgrp,nproc_bgrp,root_bgrp,intra_bgrp_comm)
           DEALLOCATE(igq_l2g_kdip)
        ELSE
           CALL mergewf(pdepg,task%data,npwq,ig_l2g(1:npwq),me_bgrp,nproc_bgrp,root_bgrp,intra_bgrp_comm)
           IF(me_bgrp == root_bgrp) ndim = npwq_g
        ENDIF
       
        if ( me_bgrp == root_bgrp ) then
            allocate(task%header(HD_LENGTH), stat=stat)
            if (stat /= 0) stop "Allocation failed for header"
            allocate(character(LEN=LEN(fname)) :: task%fname, stat=stat)
            if (stat /= 0) stop "Allocation failed for fname"  
            task%fname  = fname
            !PRINT *, "SUBMITTED THE FILE : ", task%fname
            task%header                   = 0
            task%header (HD_ID_VERSION)   = HD_VERSION
            task%header (HD_ID_DIMENSION) = ndim
            IF(islittleendian()) THEN
                task%header(HD_ID_LITTLE_ENDIAN) = 1
            ENDIF
            task_ptr = c_loc(task)
            func     = c_funloc(write_result_c)
            success  = tpool_add_work(aio_state%pool, func, task_ptr)
            !call nvtxStartRange("Sleeping")
            !call sleep_ms(42)
            !call nvtxEndRange
            ! TOOD: Added error check here
        else
            IF (ALLOCATED(task%data))  DEALLOCATE(task%data)
            DEALLOCATE(task)
        endif
        call nvtxEndRange
    END SUBROUTINE pdep_merge_and_write_G_async

    subroutine sleep_ms(milliseconds)
        use iso_c_binding
        implicit none
        type, bind(C) :: timespec
            integer(c_long) :: tv_sec
            integer(c_long) :: tv_nsec
        end type timespec

        integer(c_int), value :: milliseconds
        type(timespec), target :: ts
        interface
            subroutine nanosleep(req, rem) bind(C, name="nanosleep")
              use iso_c_binding
              type(c_ptr), value :: req, rem
            end subroutine nanosleep
        end interface

        ts%tv_sec = milliseconds / 1000
        ts%tv_nsec = mod(milliseconds, 1000) * 1000000

        call nanosleep(c_loc(ts), c_null_ptr)
    end subroutine sleep_ms

    subroutine call_sleep(wait_seconds)
        use,intrinsic                   :: iso_c_binding, only: c_int
        integer(kind=c_int),intent(in)  :: wait_seconds
        integer(kind=c_int)             :: how_long
        interface
            function c_sleep(seconds) bind (C,name="sleep")
                import
                integer(c_int)       :: c_sleep
                integer(c_int), intent(in), VALUE :: seconds
            end function c_sleep
        end interface
        if(wait_seconds>0)then
            how_long=c_sleep(wait_seconds)
        endif
    end subroutine call_sleep

    RECURSIVE SUBROUTINE write_result(task_ptr) bind (C)
        use nvtx
        use iso_c_binding
        implicit none

        type(c_ptr), value, intent(IN) :: task_ptr
        type(PDEP_AIO_TASK), pointer :: task
        integer :: iun, i
        integer(i8b) :: offset
        integer :: ret

        call nvtxStartRange("Write IO")
        call c_f_pointer(task_ptr, task)
        !PRINT *, "FNAME:", loc(task%fname),  " DATA:", loc(task%data), " HEADER:", loc(task%header)
        OPEN(asynchronous='yes', NEWUNIT=iun, FILE=TRIM(task%fname), ACCESS='STREAM', FORM='UNFORMATTED')
        !!PRINT *, "DOING THE FILE : ", task%fname, "UNIT := ", iun
        offset = 1
        WRITE(iun, asynchronous='yes', POS=offset) task%header
        offset = offset + SIZEOF(task%header)
        WRITE(iun, asynchronous='yes', POS=offset) task%data(1:SIZE(task%data))
        offset = offset + SIZEOF(task%data)
        CLOSE(iun)

        IF (ALLOCATED(task%header )) DEALLOCATE(task%header)
        IF (ALLOCATED(task%data ))   DEALLOCATE(task%data)
        DEALLOCATE(task%fname)
        DEALLOCATE(task)
        call nvtxEndRange
    END SUBROUTINE write_result


    subroutine write_result_c(task_ptr)  bind (C)
        use nvtx
        use, intrinsic :: iso_c_binding
        use unix_stdio
        implicit none

        type(c_ptr), value, intent(IN) :: task_ptr
        type(PDEP_AIO_TASK), pointer :: task
        type(c_ptr) :: file
        integer(c_size_t) :: header_size, data_size
        integer(c_size_t) :: offset
        integer(c_int) :: status
        call nvtxStartRange("Write IO")

        call c_f_pointer(task_ptr, task)
        ! Get the sizes of the header and data
        header_size = size(task%header) * sizeof(task%header(1))
        data_size   = size(task%data) * sizeof(task%data(1))

        ! Open the file
        file = c_fopen(trim(task%fname)//c_null_char, 'wb'//c_null_char)
        if (.not. c_associated(file)) then
            print *, "Error opening file: ", task%fname
            return
        endif

        !print *, "FNAME:", loc(task%fname),  " DATA:", loc(task%data), " HEADER:", loc(task%header)

        offset = c_fwrite(c_loc(task%header(1)), int(sizeof(task%header(1)),kind=c_size_t),int(size(task%header),kind=c_size_t), file)
        if (offset /= size(task%header)) then
            print *, "Error writing header to file: ", task%fname
            status = c_fclose(file)
            if (status /= 0) then
                print *, "Error closing file: ", task%fname
            endif
            return
        endif

        offset = c_fwrite(c_loc(task%data(1)), int(sizeof(task%data(1)),kind=c_size_t), int(size(task%data),kind=c_size_t), file)
        if (offset /= size(task%data)) then
            print *, "Error writing data to file: ", task%fname
            status =  c_fclose(file)
            if (status /= 0) then
                print *, "Error closing file: ", task%fname
            endif
            return
        endif

        status = c_fclose(file)
        if (status /= 0) then
            print *, "Error closing file: ", task%fname
        endif
        IF (ALLOCATED(task%header )) DEALLOCATE(task%header)
        IF (ALLOCATED(task%data ))   DEALLOCATE(task%data)
        DEALLOCATE(task%fname)
        DEALLOCATE(task)
        call nvtxEndRange
    end subroutine write_result_c

    SUBROUTINE pdep_aio_waitall(aio_state)
        type(PDEP_AIO_STATE),  intent(in) :: aio_state
        if ( me_bgrp == root_bgrp ) call tpool_wait(aio_state%pool)
    END SUBROUTINE pdep_aio_waitall


    SUBROUTINE pdep_aio_cleanup(aio_state)
        type(PDEP_AIO_STATE),  intent(in) :: aio_state
        if ( me_bgrp == root_bgrp) call tpool_destroy(aio_state%pool)
    END SUBROUTINE pdep_aio_cleanup

END MODULE pdep_async_io
