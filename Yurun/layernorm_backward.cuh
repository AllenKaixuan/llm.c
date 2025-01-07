#include "../llmc/cuda_common.h"
#include "../llmc/cuda_utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

int layernorm_backward_check(float *dinp, float *dweight, float *dbias, float *dinp_ref, float *dweight_ref,
                             float *dbias_ref, int B, int T, int C) {
    int e1 = 0;
    int e2 = 0;
    int e3 = 0;
    for (int i = 0; i < B * T * C; i++) {
        if (fabs(dinp[i] - dinp_ref[i]) > 1e-5) {
            ++e1;
            printf("mismatch at dinp[%d]: %f != %f\n", i, dinp[i], dinp_ref[i]);
            if (e1 > 10) {
                break;
            }
            // return 0;
        }
    }
    for (int i = 0; i < C; i++) {
        if (fabs(dweight[i] - dweight_ref[i]) > 1e-5) {
            ++e2;
            printf("mismatch at dweight[%d]: %f != %f\n", i, dweight[i], dweight_ref[i]);
            if (e2 > 10) {
                break;
            }
            // return 0;
        }
        if (fabs(dbias[i] - dbias_ref[i]) > 1e-5) {
            ++e3;
            printf("mismatch at dbias[%d]: %f != %f\n", i, dbias[i], dbias_ref[i]);
            if (e3 > 10) {
                return 0;
            }
            // return 0;
        }
    }
    return 1;
}

// parallelize over B, T, loop over C
__global__ void layernorm_backward_kernel0(float *dinp, float *dweight, float *dbias, const float *dout,
                                           const float *inp, const float *weight, const float *mean, const float *rstd,
                                           int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = b * T + t
    if (idx >= B * T)
        return;
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;
    const float mean_bt = mean[idx];
    const float rstd_bt = rstd[idx];

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }

    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // Since the bias and weight are shared across batches and time steps, we must use atomicAdd
        atomicAdd(&dbias[i], dout_bt[i]);              // gradient contribution to bias
        atomicAdd(&dweight[i], norm_bti * dout_bt[i]); // gradient contribution to weight
        float dval = 0.0f;                             // gradient contribution to input
        dval += dnorm_i;
        dval -= dnorm_mean;
        dval -= norm_bti * dnorm_norm_mean;
        dval *= rstd_bt;
        dinp_bt[i] += dval;
    }
}

// parallelize over B, T, loop over C, block-level reduction
__global__ void layernorm_backward_kernel1(float *dinp, float *dweight, float *dbias, const float *dout,
                                                 const float *inp, const float *weight, const float *mean,
                                                 const float *rstd, int B, int T, int C) {
    namespace cg = cooperative_groups;

    cg::thread_block block = cg::this_thread_block();

    int idx = blockIdx.x;
    if (idx >= B * T) {
        return;
    }
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;

    float mean_bt = mean[idx];
    float rstd_bt = rstd[idx];

    float thread_sum_dnorm = 0.0f;
    float thread_sum_dnorm_norm = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        thread_sum_dnorm += dnorm_i;
        thread_sum_dnorm_norm += (dnorm_i * norm_bti);
    }

    extern __shared__ float sMem[];
    float *s_dnorm = sMem;
    float *s_dnorm_norm = &sMem[blockDim.x];

    s_dnorm[threadIdx.x] = thread_sum_dnorm;
    s_dnorm_norm[threadIdx.x] = thread_sum_dnorm_norm;
    cg::sync(block);

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_dnorm[threadIdx.x] += s_dnorm[threadIdx.x + offset];
            s_dnorm_norm[threadIdx.x] += s_dnorm_norm[threadIdx.x + offset];
        }
        cg::sync(block);
    }
    float dnorm_mean = s_dnorm[0] / (float)C;
    float dnorm_norm_mean = s_dnorm_norm[0] / (float)C;


    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float dout_bti = dout_bt[i];
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bti;

        atomicAdd(&dbias[i], dout_bti);
        atomicAdd(&dweight[i], norm_bti * dout_bti);

        float dval = dnorm_i;
        dval -= dnorm_mean;
        dval -= norm_bti * dnorm_norm_mean;
        dval *= rstd_bt;
        dinp_bt[i] += dval;
    }
}

// parallelize over B, T, use cooperative groups for wrap-level reduction
__global__ void layernorm_backward_kernel2(float *dinp, float *dweight, float *dbias, const float *dout,
                                           const float *inp, const float *weight, const float *mean, const float *rstd,
                                           int B, int T, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block_tile<32> wrap = cg::tiled_partition<32>(cg::this_thread_block());
    int idx = blockIdx.x * wrap.meta_group_size() + wrap.meta_group_rank();
    if (idx >= B * T)
        return;

    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;
    const float mean_bt = mean[idx];
    const float rstd_bt = rstd[idx];

    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;

    // Read and accumulate the gradients
    for (int i = wrap.thread_rank(); i < C; i += wrap.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    // Reduce within the wrap
    dnorm_mean = cg::reduce(wrap, dnorm_mean, cg::plus<float>());
    dnorm_norm_mean = cg::reduce(wrap, dnorm_norm_mean, cg::plus<float>());

    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = wrap.thread_rank(); i < C; i += wrap.size()) {
        float dout_bti = dout_bt[i];
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bti;
        // accumulate the gradient of the bias
        atomicAdd(&dbias[i], dout_bti);
        // accumulate the gradient of the weight
        atomicAdd(&dweight[i], norm_bti * dout_bti);
        // compute the gradient of the input
        float dval = 0.0f;
        dval += dnorm_i;                    // term 1
        dval -= dnorm_mean;                 // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt;                    // final scale
        dinp_bt[i] += dval;
    }
}

void layernorm_backward_gpu(float *dinp, float *dweight, float *dbias, float *dinp_ref, float *dweight_ref,
                            float *dbias_ref, float *dout, float *inp, float *weight, float *mean, float *rstd, int B,
                            int T, int C) {
    float *d_dinp, *d_dweight, *d_dbias, *d_dout, *d_inp, *d_weight, *d_mean, *d_rstd;
    // Temporary host buffers for validation
    float *dinp_tmp, *dweight_tmp, *dbias_tmp;
    struct timespec start, end;

    // Allocate device memory
    cudaMalloc(&d_dinp, B * T * C * sizeof(float));
    cudaMalloc(&d_dweight, C * sizeof(float));
    cudaMalloc(&d_dbias, C * sizeof(float));
    cudaMalloc(&d_dout, B * T * C * sizeof(float));
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    cudaMalloc(&d_weight, C * sizeof(float));
    cudaMalloc(&d_mean, B * T * sizeof(float));
    cudaMalloc(&d_rstd, B * T * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, B * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rstd, rstd, B * T * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate temporary host buffers for validation
    dinp_tmp = (float *)mallocCheck(B * T * C * sizeof(float));
    dweight_tmp = (float *)mallocCheck(C * sizeof(float));
    dbias_tmp = (float *)mallocCheck(C * sizeof(float));

    // Configure kernel launch parameters
    int block_size = 32;

    // Test kernel0
    int grid_size = CEIL_DIV(B * T, block_size);
    // Reset gradients
    cudaMemcpy(d_dinp, dinp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dweight, dweight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dbias, dbias, C * sizeof(float), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_MONOTONIC, &start);
    layernorm_backward_kernel0<<<grid_size, block_size>>>(d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean,
                                                          d_rstd, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("layernorm_backward_kernel0 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    // Validate results
    cudaMemcpy(dinp_tmp, d_dinp, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dweight_tmp, d_dweight, C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dbias_tmp, d_dbias, C * sizeof(float), cudaMemcpyDeviceToHost);

    if (!layernorm_backward_check(dinp_tmp, dweight_tmp, dbias_tmp, dinp_ref, dweight_ref, dbias_ref, B, T, C)) {
        printf("layernorm_backward_kernel0 failed\n");
    }

    // Test kernel1
    block_size = 256;
    grid_size = CEIL_DIV(B * T, block_size);
    // Reset gradients
    cudaMemcpy(d_dinp, dinp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dweight, dweight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dbias, dbias, C * sizeof(float), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_MONOTONIC, &start);
    layernorm_backward_kernel1<<<grid_size, block_size, 2 * block_size * sizeof(float)>>>(
        d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("layernorm_backward_kernel1 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    // Validate results
    cudaMemcpy(dinp_tmp, d_dinp, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dweight_tmp, d_dweight, C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dbias_tmp, d_dbias, C * sizeof(float), cudaMemcpyDeviceToHost);

    if (!layernorm_backward_check(dinp_tmp, dweight_tmp, dbias_tmp, dinp_ref, dweight_ref, dbias_ref, B, T, C)) {
        printf("layernorm_backward_kernel1 failed\n");
    }

    // Test kernel2
    block_size = 128;
    grid_size = CEIL_DIV(B * T * 32, block_size);
    // Reset gradients
    cudaMemcpy(d_dinp, dinp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dweight, dweight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dbias, dbias, C * sizeof(float), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_MONOTONIC, &start);
    layernorm_backward_kernel2<<<grid_size, block_size>>>(d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean,
                                                          d_rstd, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("layernorm_backward_kernel2 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    // Validate results
    cudaMemcpy(dinp_tmp, d_dinp, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dweight_tmp, d_dweight, C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dbias_tmp, d_dbias, C * sizeof(float), cudaMemcpyDeviceToHost);

    if (!layernorm_backward_check(dinp_tmp, dweight_tmp, dbias_tmp, dinp_ref, dweight_ref, dbias_ref, B, T, C)) {
        printf("layernorm_backward_kernel1 failed\n");
    }

    // Cleanup
    cudaFree(d_dinp);
    cudaFree(d_dweight);
    cudaFree(d_dbias);
    cudaFree(d_dout);
    cudaFree(d_inp);
    cudaFree(d_weight);
    cudaFree(d_mean);
    cudaFree(d_rstd);
    free(dinp_tmp);
    free(dweight_tmp);
    free(dbias_tmp);
}