#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cmath>
#include <iostream>

struct power_functor {
    double constant;
    power_functor(double _constant) : constant(_constant) {}
    __host__ __device__
    double operator()(const double& x) const {
        return pow(x, constant);
    }
};

extern "C"  double compute_power_sum(const double* d_array, int size, double constant){
    thrust::device_ptr<const double> dev_ptr(d_array);
    power_functor func(constant);
    double sum = thrust::transform_reduce(dev_ptr, dev_ptr + size, func, 0.0, thrust::plus<double>());
    return sum;
}
