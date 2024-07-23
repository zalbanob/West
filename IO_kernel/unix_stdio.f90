! unix_stdio.F90
!
! Author:  Philipp Engel
! Licence: ISC
module unix_stdio
    use, intrinsic :: iso_c_binding
    implicit none
    private

    integer(kind=c_int), parameter, public :: EOF = -1

    public :: c_getchar
    public :: c_getline
    public :: c_fclose
    public :: c_fdopen
    public :: c_fflush
    public :: c_fopen
    public :: c_fread
    public :: c_fwrite

    interface
        ! int getchar(void)
        function c_getchar() bind(c, name='getchar')
            import :: c_int
            implicit none
            integer(kind=c_int) :: c_getchar
        end function c_getchar

        ! ssize_t getline(char **lineptr, size_t *n, FILE *stream)
        function c_getline(lineptr, n, stream) bind(c, name='getline')
            import :: c_char, c_ptr, c_size_t
            implicit none
            character(kind=c_char), intent(in)        :: lineptr(*)
            integer(kind=c_size_t), intent(in)        :: n
            type(c_ptr),            intent(in), value :: stream
            integer(kind=c_size_t)                    :: c_getline
        end function c_getline

        ! int fclose(FILE *stream)
        function c_fclose(stream) bind(c, name='fclose')
            import :: c_int, c_ptr
            implicit none
            type(c_ptr), intent(in), value :: stream
            integer(kind=c_int)            :: c_fclose
        end function c_fclose

        ! FILE *fopen(int fd, const char *mode)
        function c_fdopen(fd, mode) bind(c, name='fdopen')
            import :: c_char, c_int, c_ptr
            implicit none
            integer(kind=c_int),    intent(in), value :: fd
            character(kind=c_char), intent(in)        :: mode
            type(c_ptr)                               :: c_fdopen
        end function c_fdopen

        ! int fflush(FILE *stream)
        function c_fflush(stream) bind(c, name='fflush')
           import :: c_int, c_ptr
            implicit none
            type(c_ptr), intent(in), value :: stream
            integer(kind=c_int)            :: c_fflush
        end function c_fflush

        ! FILE *fopen(const char *path, const char *mode)
        function c_fopen(path, mode) bind(c, name='fopen')
            import :: c_char, c_ptr
            implicit none
            character(kind=c_char), intent(in) :: path
            character(kind=c_char), intent(in) :: mode
            type(c_ptr)                        :: c_fopen
        end function c_fopen

        ! size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
        function c_fread(ptr, size, nmemb, stream) bind(c, name='fread')
            import :: c_ptr, c_size_t
            implicit none
            type(c_ptr),            intent(in), value :: ptr
            integer(kind=c_size_t), intent(in), value :: size
            integer(kind=c_size_t), intent(in), value :: nmemb
            type(c_ptr),            intent(in), value :: stream
            integer(kind=c_size_t)                    :: c_fread
        end function c_fread

        ! size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
        function c_fwrite(ptr, size, nmemb, stream) bind(c, name='fwrite')
            import :: c_ptr, c_size_t
            implicit none
            type(c_ptr),            intent(in), value :: ptr
            integer(kind=c_size_t), intent(in), value :: size
            integer(kind=c_size_t), intent(in), value :: nmemb
            type(c_ptr),            intent(in), value :: stream
            integer(kind=c_size_t)                    :: c_fwrite
        end function c_fwrite
    end interface
end module unix_stdio