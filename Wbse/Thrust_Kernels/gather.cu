#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

using complex_t = thrust::complex<double>;

struct InvFFTTransformFunctor {
    __host__ __device__
    thrust::tuple<complex_t, complex_t> operator()(const thrust::tuple<complex_t, complex_t>& input) const {
        complex_t a1 = thrust::get<0>(input);
        complex_t a2 = thrust::get<1>(input);
        
        complex_t b_nl = a1 + complex_t(0, 1) * a2;
        complex_t b_nlm = thrust::conj(a1 - complex_t(0, 1) * a2);
        
        return thrust::make_tuple(b_nl, b_nlm);
    }
};

void thrust_double_invfft_gamma_pp(
    const complex_t* a1, const complex_t* a2, complex_t* b,
    const int* dfft_nl_d, const int* dfft_nlm_d,
    int n, int batch_size, int nx, int nnr)
{
    thrust::device_ptr<const complex_t> d_a1(a1);
    thrust::device_ptr<const complex_t> d_a2(a2);
    thrust::device_ptr<complex_t> d_b(b);
    thrust::device_ptr<const int> d_dfft_nl_d(dfft_nl_d);
    thrust::device_ptr<const int> d_dfft_nlm_d(dfft_nlm_d);

    thrust::device_vector<int> indices(2 * n * batch_size);
    thrust::device_vector<complex_t> results(2 * n * batch_size);

    thrust::counting_iterator<int> count_it(0);
    thrust::transform(
        count_it, count_it + n * batch_size,
        indices.begin(),
        [=] __device__ (int idx) {
            int ig = idx % n;
            int ibatch = idx / n;
            return ig + ibatch * nx;
        }
    );
    thrust::copy(indices.begin(), indices.end(), indices.begin() + n * batch_size);

    thrust::device_vector<complex_t> gathered_a1(n * batch_size);
    thrust::device_vector<complex_t> gathered_a2(n * batch_size);
    thrust::gather(indices.begin(), indices.begin() + n * batch_size, d_a1, gathered_a1.begin());
    thrust::gather(indices.begin(), indices.begin() + n * batch_size, d_a2, gathered_a2.begin());

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(gathered_a1.begin(), gathered_a2.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(gathered_a1.end(), gathered_a2.end())),
        thrust::make_zip_iterator(thrust::make_tuple(results.begin(), results.begin() + n * batch_size)),
        InvFFTTransformFunctor()
    );

    thrust::transform(
        count_it, count_it + n * batch_size,
        indices.begin(),
        [=] __device__ (int idx) {
            int ig = idx % n;
            int ibatch = idx / n;
            return d_dfft_nl_d[ig] + ibatch * nnr;
        }
    );
    thrust::transform(
        count_it, count_it + n * batch_size,
        indices.begin() + n * batch_size,
        [=] __device__ (int idx) {
            int ig = idx % n;
            int ibatch = idx / n;
            return d_dfft_nlm_d[ig] + ibatch * nnr;
        }
    );

    thrust::scatter(results.begin(), results.end(), indices.begin(), d_b);
}

extern "C" {
    void thrust_double_invfft_gamma_c(
        const complex_t* a1,
        const complex_t* a2,
        complex_t* b,
        const int* dfft_nl_d,
        const int* dfft_nlm_d,
        int n, int batch_size, int nx, int nnr)
    {
        thrust_double_invfft_gamma_pp(a1, a2, b, dfft_nl_d, dfft_nlm_d, n, batch_size, nx, nnr);
    }
}