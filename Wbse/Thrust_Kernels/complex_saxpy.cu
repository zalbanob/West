#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <complex>
#include <thrust/complex.h>

typedef double DP;
typedef thrust::complex<DP> Complex;


//!$acc parallel loop private(temp1,temp2) reduction(+:val(:), val_2(:)) present(proj, aux) default(present) 
//DO ir = 1, dffts_nnr
//    temp1 = aux(ir) * REAL (psic(ir), KIND=DP)
//    temp2 = aux(ir) * AIMAG(psic(ir))
//    val  (1) = val  (1) + temp1 * proj(ir, 1)
//    val  (2) = val  (2) + temp1 * proj(ir, 2)
//    val  (3) = val  (3) + temp1 * proj(ir, 3)
//    val  (4) = val  (4) + temp1 * proj(ir, 4)
//    val  (5) = val  (5) + temp1 * proj(ir, 5)
//    val  (6) = val  (6) + temp1 * proj(ir, 6)
//ENDDO


template <bool UseReal>
struct saxpy_functor {
    const Complex* psic;
    const DP* aux;
    const DP* proj;
    int dffts_nnr;
    
    saxpy_functor(const Complex* _psic, const DP* _aux, const DP* _proj, int _dffts_nnr) 
        : psic(_psic), aux(_aux), proj(_proj), dffts_nnr(_dffts_nnr) {}

    __device__ __host__
    thrust::tuple<DP, DP, DP, DP, DP, DP> operator()(const int& ir) const {
        DP temp = aux[ir] * (UseReal ? psic[ir].real() : psic[ir].imag() );
        return thrust::make_tuple(
            temp * proj[ir + dffts_nnr * 0], temp * proj[ir + dffts_nnr * 1], 
            temp * proj[ir + dffts_nnr * 2], temp * proj[ir + dffts_nnr * 3], 
            temp * proj[ir + dffts_nnr * 4], temp * proj[ir + dffts_nnr * 5]
        );
    }
};

struct tuple_sum {
    __device__ __host__
    thrust::tuple<DP, DP, DP, DP, DP, DP> operator()(const thrust::tuple<DP, DP, DP, DP, DP, DP>& a,
                                                     const thrust::tuple<DP, DP, DP, DP, DP, DP>& b) const {
        return thrust::make_tuple(
            thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b),
            thrust::get<2>(a) + thrust::get<2>(b),
            thrust::get<3>(a) + thrust::get<3>(b),
            thrust::get<4>(a) + thrust::get<4>(b),
            thrust::get<5>(a) + thrust::get<5>(b)
        );
    }
};

template <bool UseReal>
void compute_saxpy_wrapper(const Complex* psic, const DP* aux, const DP* proj, DP* result, int dffts_nnr) {
    auto result_sum = thrust::transform_reduce(
        thrust::device,
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(dffts_nnr),
        saxpy_functor<UseReal>(psic, aux, proj, dffts_nnr),
        thrust::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tuple_sum()
    );

    result[0] = thrust::get<0>(result_sum);
    result[1] = thrust::get<1>(result_sum);
    result[2] = thrust::get<2>(result_sum);
    result[3] = thrust::get<3>(result_sum);
    result[4] = thrust::get<4>(result_sum);
    result[5] = thrust::get<5>(result_sum);
}



extern "C" void complex_saxpy(bool real, const Complex* psic, const DP* aux, const DP* proj, DP* result, int dffts_nnr) {
    //printf("Function Parameters:\n");
    //printf("  real     : %s\n", real ? "true" : "false");
    //printf("  psic     : %p\n", (void*)psic);
    //printf("  aux      : %p\n", (void*)aux);
    //printf("  proj     : %p\n", (void*)proj);
    //printf("  result   : %p\n", (void*)result);
    //printf("  dffts_nnr: %d\n", dffts_nnr);

    if (real) {
        compute_saxpy_wrapper<true>(psic, aux, proj, result, dffts_nnr);
    } else {
        compute_saxpy_wrapper<false>(psic, aux, proj, result, dffts_nnr);
    }
}