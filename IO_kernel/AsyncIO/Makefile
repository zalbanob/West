include ../../../make.inc

TARGET = objects

# Source files
CSRC = tpool.c cpu_count.c
FSRC = tpool_bindings.f90 cpu_count_mod.f90
DEPFLAGS = -M

# Object files
COBJ = $(CSRC:.c=.o)
FOBJ = $(FSRC:.f90=.o)

# Module files
FMOD = tpool_bindings.mod cpu_count_mod.mod

# Rules
.PHONY: all clean

all: $(COBJ) $(FOBJ)

%.o: %.c
	$(CC) $(CFLAGS)  -lpthread -c $< -o $@

%.o: %.f90
	$(FC) $(FFLAGS) -lpthread -c $< -o $@

tpool_bindings.o: tpool.o
cpu_count_mod.o: cpu_count.o
pdep_async_io.o: cpu_count_mod.o tpool_bindings.o

tpool_bindings.mod: tpool_bindings.o
cpu_count_mod.mod: cpu_count_mod.o

clean:
	rm -f $(COBJ) $(FOBJ) $(FMOD)
