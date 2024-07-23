
//#if defined(__CUDA)
    #include <thrust/device_ptr.h>
//#else
//    #include <thrust/system/omp/pointer.h>
//#endif

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>
#include <cstddef>



//!$acc parallel loop reduction(+:summ_ib,summ_jb,summ_ij) present(orb_i,orb_j) copy(summ_ib,summ_jb,summ_ij)
//DO ir = 1, dffts_nnr
//   summ_ib = summ_ib + orb_i(ir)**4
//   summ_jb = summ_jb + orb_j(ir)**4
//   summ_ij = summ_ij + orb_i(ir)**2 * orb_j(ir)**2
//ENDDO
//!$acc end parallel

struct compute_sums {
    //#if defined(__CUDA)
    __host__ __device__
    //#endif
    thrust::tuple<double, double, double> operator()(const thrust::tuple<double, double>& t) const {
        double orb_i = thrust::get<0>(t);
        double orb_j = thrust::get<1>(t);
        double partial_sum_ib = orb_i * orb_i * orb_i * orb_i;
        double partial_sum_jb = orb_j * orb_j * orb_j * orb_j;
        double partial_sum_ij = orb_i * orb_i * orb_j * orb_j;
        return thrust::make_tuple(partial_sum_ib, partial_sum_jb, partial_sum_ij);
    }
};

struct tuple_sum {
    //#if defined(__CUDA)
    __host__ __device__
    //#endif
    thrust::tuple<double, double, double> operator()(const thrust::tuple<double, double, double>& a,
                                                     const thrust::tuple<double, double, double>& b) const {
        return thrust::make_tuple(
            thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b),
            thrust::get<2>(a) + thrust::get<2>(b)
        );
    }
};

extern "C" void check_ovl_wannier_red(const double* d_orb_i, const double* d_orb_j, std::size_t size, double* result) {
//#if defined(__CUDA) || defined(_OPENACC)
    thrust::device_ptr<const double> orb_i(d_orb_i);
    thrust::device_ptr<const double> orb_j(d_orb_j);
//#else 
//    thrust::omp::pointer<const double> orb_i(d_orb_i);
//    thrust::omp::pointer<const double> orb_j(d_orb_j);
//#endif
    thrust::tuple<double, double, double> initial(0.0, 0.0, 0.0);
    thrust::tuple<double, double, double> thrust_result = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(orb_i, orb_j)),
        thrust::make_zip_iterator(thrust::make_tuple(orb_i + size, orb_j + size)),
        compute_sums(),
        initial,
        tuple_sum()
    );
    result[0] = thrust::get<0>(thrust_result);
    result[1] = thrust::get<1>(thrust_result);
    result[2] = thrust::get<2>(thrust_result);
}
