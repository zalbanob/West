include ../../../make.inc

TARGET = objects

CSRC = check_ovl_wannier_red.cu complex_saxpy.cu power_sum.cu update_hg.cu build_hr.cu gather.cu
FSRC = thrust_kernels.f90 
DEPFLAGS = -M

# Object files
COBJ = $(CSRC:.cu=.o)
FOBJ = $(FSRC:.f90=.o)

FMOD = thrust_kernels.mod

THRUST_KERNEL_OBJS = thrust_kernels.o check_ovl_wannier_red.o power_sum.o complex_saxpy.o update_hg.o build_hr.o gather.o

# Rules
.PHONY: all clean

all: $(COBJ) $(FOBJ) lib_thrust_kernels.a

#	scorep --cuda --thread=pthread --openacc --nocompiler --nomemory --mpp=mpi nvcc -gencode arch=compute_90,code=sm_90 -g -G -O3 -I../../Libraries/cccl/thrust -I../../Libraries/cccl/libcudacxx/include -I../../Libraries/cccl/cub -c $< -o $@

%.o: %.cu
	nvcc -lineinfo  --extended-lambda -gencode arch=compute_90,code=sm_90 -g -O3 -I../../Libraries/cccl/thrust -I../../Libraries/cccl/libcudacxx/include -I../../Libraries/cccl/cub -c $< -o $@

%.o: %.f90
	$(FC) $(FFLAGS) -lstd++ -c -cuda $< -o $@


lib_thrust_kernels.a : $(THRUST_KERNEL_OBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@


 
thrust_kernels.o: check_ovl_wannier_red.o
thrust_kernels.o: power_sum.o
thrust_kernels.o: complex_saxpy.o
thrust_kernels.o: update_hg.o
thrust_kernels.o: build_hr.o
thrust_kernels.o: gather.o


thrust_kernels.mod: thrust_kernels.o

clean:
	rm -f $(COBJ) $(FOBJ) $(FMOD)
