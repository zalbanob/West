#include <cuda_runtime.h>
#include <stdio.h>

#define Complex double2
#define real x
#define imag y

__constant__ int d_ag_shape     [4];
__constant__ int d_bg_shape     [4];
__constant__ int d_c_distr_shape[2];


__global__ 
void reduce_ag_bg_kernel_dyn(Complex*   __restrict__ ag,
                             Complex*   __restrict__ bg,
                             double*    __restrict__ c_distr,
                             const int* __restrict__ nbnd_loc,
                             const int* __restrict__ ngk,
                             int l1_e, int l2_s, int l2_e, 
                             int kpt_pool_nloc, int nimage,int idx, int gstart) {

    extern __shared__ double sdata[];

    int il1 = blockIdx.x;
    int il2 = blockIdx.y + l2_s;
    Complex a_val, b_val;

    if (il1 > l1_e || il2 > l2_e) return;

    double reduce = 0.0;

    int max_npw  = d_ag_shape[0];
    int max_nbnd = d_ag_shape[1];
    int max_kpt  = d_ag_shape[2];

    int c_off =  d_c_distr_shape[0];

    for (int iks = 0; iks < kpt_pool_nloc; iks++) {
        int nbndval = nbnd_loc[iks];
        int npw     = ngk[iks];

        int a_offset = iks * max_npw * max_nbnd + il1 * max_npw * max_nbnd * max_kpt;
        int b_offset = iks * max_npw * max_nbnd + il2 * max_npw * max_nbnd * max_kpt;

        for (int lbnd = 0; lbnd < nbndval; lbnd++) {
            for (int il3 = threadIdx.x; il3 < npw; il3 += blockDim.x) {
                a_val = ag[il3 + lbnd * max_npw + a_offset];
                b_val = bg[il3 + lbnd * max_npw + b_offset];
                const double constant  = 1.0  +  ( il3 != 0 || gstart != 2);
                double a  = a_val.real * b_val.real;
                reduce += constant * a;

                double b = a_val.imag * b_val.imag;
                reduce += 2.0 * b;
            }
        }
    }
    unsigned mask = 0xFFFFFFFFU;
    int lane   = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        reduce += __shfl_down_sync(mask, reduce, offset);
    }
    if (lane == 0) sdata[warpID] = reduce;
    __syncthreads();
    if (warpID == 0) {
        reduce = sdata[lane];
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
            reduce += __shfl_down_sync(mask, reduce, offset);
        }
        if  (threadIdx.x == 0) {
            int ig1 = nimage * il1 + idx ;
            c_distr[ig1 + il2 *c_off ] = reduce;
        }
    }
}


template<int gstart>
__global__ 
void reduce_ag_bg_kernel(Complex*   __restrict__ ag,
                         Complex*   __restrict__ bg,
                         double*    __restrict__ c_distr,
                         const int* __restrict__ nbnd_loc,
                         const int* __restrict__ ngk,
                         int l1_e, int l2_s, int l2_e,
                         int kpt_pool_nloc, int nimage, int idx) {

    extern __shared__ double sdata[];

    int il1 = blockIdx.x;
    int il2 = blockIdx.y + l2_s;
    Complex a_val, b_val;

    if (il1 > l1_e || il2 > l2_e) return;

    register double reduce = 0.0;

    register int max_npw  = d_ag_shape[0];
    register int max_nbnd = d_ag_shape[1];
    register int max_kpt  = d_ag_shape[2];

    register int c_off =  d_c_distr_shape[0];

    for (int iks = 0; iks < kpt_pool_nloc; iks++) {
        int nbndval = nbnd_loc[iks];
        int npw     = ngk[iks];

        register int a_offset = iks * max_npw * max_nbnd + il1 * max_npw * max_nbnd * max_kpt;
        register int b_offset = iks * max_npw * max_nbnd + il2 * max_npw * max_nbnd * max_kpt;

        for (int lbnd = 0; lbnd < nbndval; lbnd++) {
            for (int il3 = threadIdx.x; il3 < npw; il3 += blockDim.x) {
                a_val = ag[il3 + lbnd * max_npw + a_offset];
                b_val = bg[il3 + lbnd * max_npw + b_offset];
                const double constant  = 1.0  +  ( il3 != 0 || gstart != 2);
                double a  = a_val.real * b_val.real;
                reduce += constant * a;

                double b = a_val.imag * b_val.imag;
                reduce += 2.0 * b;
            }
        }
    }
    unsigned mask = 0xFFFFFFFFU;
    int lane   = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        reduce += __shfl_down_sync(mask, reduce, offset);
    }
    if (lane == 0) sdata[warpID] = reduce;
    __syncthreads();
    if (warpID == 0) {
        reduce = sdata[lane];
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
            reduce += __shfl_down_sync(mask, reduce, offset);
        }
        if  (threadIdx.x == 0) {
            int ig1 = nimage * il1 + idx ;
            c_distr[ig1 + il2 *c_off ] = reduce;
        }
    }
}


extern "C" void reduce_ag_bg(Complex* ag, int * ag_shape, int ag_ndim,
                             Complex* bg, int * bg_shape, int bg_ndim,
                             double* c_distr, int * c_distr_shape, int c_distr_ndim,
                             const int* nbnd_loc,
                             const int* ngk,
                             int l1_e, int l2_s, int l2_e,
                             int kpt_pool_nloc,
                             int nimage, int idx, int gstart) {
                                
 /*  int * ag_shape, int ag_ndim,
    int * bg_shape, int bg_ndim,
    int * c_distr , int c_distr_ndim,
 */

    //printf("INSIDE C CALL\n");
    //printf("Address of ag: %p\n", (void*)ag);
    //printf("Address of bg: %p\n", (void*)bg);
    //printf("Address of c_distr: %p\n", (void*)c_distr);
    //printf("Address of nbnd_loc: %p\n", (void*)nbnd_loc);
    //printf("Address of ngk: %p\n", (void*)ngk);
    //printf("Value of l1_e: %d\n", l1_e);
    //printf("Value of l2_s: %d\n", l2_s);
    //printf("Value of l2_e: %d\n", l2_e);
    //printf("Value of kpt_pool_nloc: %d\n", kpt_pool_nloc);
    //printf("Value of nimage: %d\n", nimage);
    //printf("Value of idx: %d\n", idx);
    //printf("Value of gstart: %d\n", gstart);

    cudaMemcpyToSymbol(d_ag_shape, ag_shape, ag_ndim * sizeof(int));
    cudaMemcpyToSymbol(d_bg_shape, bg_shape, bg_ndim * sizeof(int));
    cudaMemcpyToSymbol(d_c_distr_shape, c_distr_shape, c_distr_ndim * sizeof(int));

    dim3 blockDim(1024);
    dim3 gridDim(l1_e, l2_e - l2_s + 1);

    size_t sharedMemSize = (blockDim.x / 32) * sizeof(double);
    if (gstart == 2) {
        reduce_ag_bg_kernel<2><<<gridDim, blockDim, sharedMemSize>>>(ag,
                                                                     bg,
                                                                     c_distr,
                                                                     nbnd_loc, ngk, l1_e, l2_s - 1, l2_e,
                                                                     kpt_pool_nloc, nimage, idx);
    } else {
        reduce_ag_bg_kernel_dyn<<<gridDim, blockDim, sharedMemSize>>>(ag,
                                                                      bg,
                                                                      c_distr, 
                                                                      nbnd_loc, ngk, l1_e, l2_s - 1, l2_e,
                                                                      kpt_pool_nloc, nimage, idx, gstart);
    }

        //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
}
