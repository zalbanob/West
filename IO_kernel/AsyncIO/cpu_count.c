#include <unistd.h>
#include <stdio.h>

long get_cpu_count() {
    long r  = sysconf(_SC_NPROCESSORS_ONLN);
    return r;
}
