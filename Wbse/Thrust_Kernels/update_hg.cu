#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/extrema.h>
#include <omp.h>
#include <chrono>
#include <thrust/complex.h>
#include <cuda_runtime.h>

typedef double DP;
typedef thrust::complex<DP> Complex;

#define warpSize 32


#define Complex double2 
#define imag y
#define real x

__constant__ int d_shape[4];


void print_device_pointers(thrust::device_ptr<int> d_ngk_ptr, thrust::device_ptr<int> d_nbnd_loc_ptr, int size) {
    thrust::host_vector<int> h_ngk(size);
    thrust::host_vector<int> h_nbnd_loc(size);

    thrust::copy(d_ngk_ptr, d_ngk_ptr + size, h_ngk.begin());
    thrust::copy(d_nbnd_loc_ptr, d_nbnd_loc_ptr + size, h_nbnd_loc.begin());

    std::cout << "d_ngk_ptr: " << std::endl;
    for(int i = 0; i < size; i++) std::cout << h_ngk[i] << " ";
    std::cout << std::endl;

    std::cout << "d_nbnd_loc_ptr: " << std::endl;
    for(int i = 0; i < size; i++) std::cout << h_nbnd_loc[i] << " ";
    std::cout << std::endl;
}





__global__
void update_hg_kernel(DP*     __restrict__ vr_distr, 
                     Complex* __restrict__ ag, 
                     Complex* __restrict__ hg,
                      int*    __restrict__ nbnd_loc, 
                      int*    __restrict__ ngk,
                      int l2_s,
                      int l1_e, int nimage, int idx, int kpt_pool_nloc,
                      int loop1, int loop2, int loop3, int loop4) {
    
    //extern __shared__ Complex sdata[];
    __shared__ double sdata_real[32];
    __shared__ double sdata_imag[32];

    for(int wk = 0; wk < blockDim.x; wk++){
        int t = blockIdx.x * blockDim.x + wk;
        int i = ( t % loop1);
        int j = ( t / loop1) % loop2;
        int k = ( t / (loop1 * loop2)) % loop3;
        int l = ( t / (loop1 * loop2 * loop3));

        int il2  = l + l2_s;
        int iks  = k;
        int lbnd = j;
        int il3  = i;

        int nbndval = nbnd_loc[iks];
        int npw = ngk[iks];

        if (lbnd < nbndval && il3 < npw) {
            int tid = threadIdx.x;
            unsigned mask = 0xFFFFFFFFU;
            int lane   = threadIdx.x % warpSize;
            int warpID = threadIdx.x / warpSize;

            double sum_real = 0;
            double sum_imag = 0;

            int base_idx   = il3 + lbnd * d_shape[0] + iks * d_shape[0] * d_shape[1];
            int kpt_offset = d_shape[0] * d_shape[1] * d_shape[2];
            int il2_offset = il2 * kpt_offset;

            #pragma unroll
            for (int il1 = tid; il1 < l1_e; il1 += blockDim.x) {
                int ig1  = nimage * il1 + idx;
                const DP vr          = vr_distr[ig1 + il2 * d_shape[3]];
                const Complex ag_val = ag[(il1 * kpt_offset) + base_idx ];
                sum_real +=  vr * ag_val.real;
                sum_imag +=  vr * ag_val.imag;
            }

            #pragma unroll
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                sum_real += __shfl_down_sync(mask, sum_real, offset);
                sum_imag += __shfl_down_sync(mask, sum_imag, offset);
            }

            if (lane == 0){ 
                sdata_real[warpID] = sum_real;
                sdata_imag[warpID] = sum_imag;
            }
            __syncthreads();

            if (warpID == 0) {
                sum_real = sdata_real[lane];
                sum_imag = sdata_imag[lane];
                #pragma unroll
                for (int offset = 8 >> 1; offset > 0; offset >>= 1){
                    sum_real += __shfl_down_sync(mask, sum_real, offset);
                    sum_imag += __shfl_down_sync(mask, sum_imag, offset);
                }
                if  (tid == 0) {
                    hg[base_idx + il2_offset].real +=  sum_real;
                    hg[base_idx + il2_offset].imag +=  sum_imag;
                }
            }
        }
    }
}

extern "C" 
void update_hg(DP* vr_distr, 
                          Complex* ag, 
                          Complex* hg,
                          int* nbnd_loc, 
                          int* ngk, 
                          int l2_s, int l2_e, 
                          int kpt_pool_nloc, 
                          int l1_e, int nimage, int idx, int * shape) {
    //const auto start = std::chrono::steady_clock::now();
    thrust::device_ptr<int> d_nbnd_loc_ptr = thrust::device_pointer_cast(nbnd_loc);
    thrust::device_ptr<int> d_ngk_ptr      = thrust::device_pointer_cast(ngk);

    //std::cout << "START OF DEBUGGING INFO\n";
    //std::cout << "l2_s: "           << l2_s          << std::endl;
    //std::cout << "l2_e: "           << l2_e          << std::endl;
    //std::cout << "kpt_pool_nloc: "  << kpt_pool_nloc << std::endl;
    //std::cout << "l1_e: "           << l1_e          << std::endl;
    //std::cout << "nimage: "         << nimage        << std::endl;
    //std::cout << "idx: "            << idx           << std::endl;
    //std::cout << "shape: ";
    //for (int i = 0; i < 4; ++i) std::cout << shape[i] << " "; std::cout << std::endl;
    //print_device_pointers(d_ngk_ptr, d_nbnd_loc_ptr, kpt_pool_nloc);
    //std::cout << "END OF DEBUGGING INFO\n";

    cudaMemcpyToSymbol(d_shape, shape, 4 * sizeof(int));
    int nbnd_loc_end  = shape[1]; //*(thrust::max_element(d_nbnd_loc_ptr, d_nbnd_loc_ptr + kpt_pool_nloc));
    int ngk_end       = shape[0]; //*(thrust::max_element(d_ngk_ptr     , d_ngk_ptr + kpt_pool_nloc));
    int total_elements = (l2_e - l2_s + 1) * kpt_pool_nloc * nbnd_loc_end * ngk_end;
    
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    size_t shared_mem_size = 0;//(block_size/32) * sizeof(Complex);
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);


    //cudaEventRecord(start);
    update_hg_kernel<<<grid_size, block_size, shared_mem_size>>>(vr_distr, ag, hg, nbnd_loc, ngk, l2_s - 1, l1_e, nimage, idx, kpt_pool_nloc,
    ngk_end,nbnd_loc_end, kpt_pool_nloc, (l2_e - l2_s + 1));
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Kernel execution time: %f ms\n", milliseconds);
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);

    //const auto stop = std::chrono::steady_clock::now();
    //const std::chrono::duration<double, std::micro> delta = stop - start;
    //std::cout << delta.count() << " us." << '\n';
}