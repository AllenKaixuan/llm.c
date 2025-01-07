#include "../llmc/cuda_common.h"
#include "../llmc/cuda_utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

int layernorm_forward_check(float *out, float *mean, float *rstd, float *out_ref, float *mean_ref, float *rstd_ref,
                            int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            if (fabs(mean[b * T + t] - mean_ref[b * T + t]) > 1e-5) {
                printf("mismatch at mean[%d,%d]: %f != %f\n", b, t, mean[b * T + t], mean_ref[b * T + t]);
                return 0;
            }
            if (fabs(rstd[b * T + t] - rstd_ref[b * T + t]) > 1e-5) {
                printf("mismatch at rstd[%d,%d]: %f != %f\n", b, t, rstd[b * T + t], rstd_ref[b * T + t]);
                return 0;
            }
            for (int i = 0; i < C; i++) {
                if (fabs(out[b * T * C + t * C + i] - out_ref[b * T * C + t * C + i]) > 1e-5) {
                    printf("mismatch at out[%d,%d,%d]: %f != %f\n", b, t, i, out[b * T * C + t * C + i],
                           out_ref[b * T * C + t * C + i]);
                    return 0;
                }
            }
        }
    }
    return 1;
}

// Parallelize over B, T, loop over C
__global__ void layernorm_forward_kernel0(float *out, float *mean, float *rstd, const float *inp, const float *weight,
                                          const float *bias, int B, int T, int C) {
    // One thread per element in the output tensor

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = b * T + t
    if (idx >= B * T)
        return;
    float m = 0.0f;
    float v = 0.0f;
    // Calculate mean and variance
    const float *x = inp + idx * C;
    for (int i = 0; i < C; i++) {
        m += x[i];
    }
    m = m / C;
    for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
    }
    v = v / C;
    // Calculate the rstd
    float s = 1.0f / sqrtf(v + 1e-5f);
    // Calculate the output
    float *out_bt = out + idx * C;
    for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // Normalize
        float o = n * weight[i] + bias[i]; // Scale
        out_bt[i] = o;
    }
    // Save the mean and rstd
    mean[idx] = m;
    rstd[idx] = s;
}

// Parallelize over B, T, use shared memory for reduction
// Variance = E[X^2] - E[X]^2 to avoid reading from global memory twice
__global__ void mv_kernel(float *mean, float *rstd, const float *inp, int B, int T, int C, int block_size) {
    int idx = blockIdx.x;
    if (idx >= B * T)
        return;
    extern __shared__ float smdata[];
    float *sm2data = smdata + block_size;

    float m = 0.0f;  // sum of x
    float m2 = 0.0f; // sum of x^2
    float xtemp;     // register to hold the input value
    const float *x = inp + idx * C;
    // Accumulate the sum of x and x^2 when reading from global memory
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        xtemp = x[i];
        m += xtemp;
        m2 += xtemp * xtemp;
    }
    // Save the sum of x and x^2 to shared memory
    smdata[threadIdx.x] = m;
    sm2data[threadIdx.x] = m2;
    __syncthreads();
    // Reduce within the block
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            smdata[threadIdx.x] += smdata[threadIdx.x + stride];
            sm2data[threadIdx.x] += sm2data[threadIdx.x + stride];
        }
    }
    // Normalize and scale
    if (threadIdx.x == 0) {
        float m = smdata[0] / C;
        float v = sm2data[0] / C - m * m;
        mean[idx] = m;
        rstd[idx] = rsqrtf(v + 1e-5f);
    }
}

// Parallelize over B, T, C
__global__ void normalization_kernel(float *out, const float *inp, const float *mean, const float *rstd,
                                     const float *weight, const float *bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C)
        return;
    int bt = idx / C;
    int i = idx % C;
    float x = inp[idx];
    float m = mean[bt];
    float s = rstd[bt];
    float n = (x - m) * s;
    out[idx] = n * weight[i] + bias[i];
}

// Parallelize over B, T, but do the reduction twice, once within the wrap and once within the block
__global__ void layernorm_forward_kernel2(float *out, float *mean, float *rstd, const float *inp, const float *weight,
                                          const float *bias, int B, int T, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block_tile<32> wrap = cg::tiled_partition<32>(cg::this_thread_block());
    int idx = blockIdx.x;
    if (idx >= B * T)
        return;
    __shared__ float smdata[32]; // May not use all of them, one for a wrap, max is 1024 / 32 = 32
    __shared__ float sm2data[32];
    float m = 0.0f;  // sum of x
    float m2 = 0.0f; // sum of x^2
    float xtemp;     // register to hold the input value
    const float *x = inp + idx * C;
    // Accumulate the sum of x and x^2 when reading from global memory
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        xtemp = x[i];
        m += xtemp;
        m2 += xtemp * xtemp;
    }

    // Reduce within the wrap
    float wrap_smdata = cg::reduce(wrap, m, cg::plus<float>());
    float wrap_sm2data = cg::reduce(wrap, m2, cg::plus<float>());
    smdata[wrap.meta_group_rank()] = wrap_smdata;
    sm2data[wrap.meta_group_rank()] = wrap_sm2data;
    __syncthreads();
    // Reduce within the block
    // Only the first "wrap_num" threads in the wrap participate in the reduction
    wrap_smdata = (wrap.thread_rank() < wrap.meta_group_size()) ? smdata[wrap.thread_rank()] : 0.0f;
    wrap_sm2data = (wrap.thread_rank() < wrap.meta_group_size()) ? sm2data[wrap.thread_rank()] : 0.0f;
    float block_smdata = cg::reduce(wrap, wrap_smdata, cg::plus<float>());
    float block_sm2data = cg::reduce(wrap, wrap_sm2data, cg::plus<float>());
    if (wrap.thread_rank() < wrap.meta_group_size()) {
        smdata[wrap.thread_rank()] = wrap_smdata;
        sm2data[wrap.thread_rank()] = wrap_sm2data;
    }
    __syncthreads();
    float ma = block_smdata / C;
    float v = block_sm2data / C - ma * ma;
    float s = rsqrtf(v + 1e-5f);

    if (threadIdx.x == 0) {
        mean[idx] = ma;
        rstd[idx] = s;
    }
    // Normalize and scale
    float *out_bt = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float x = inp[idx * C + i];
        float n = (x - ma) * s;
        out_bt[i] = n * weight[i] + bias[i];
    }
}

void layernorm_forwawrd_gpu(float *out, float *mean, float *rstd, float *out_ref, float *mean_ref, float *rstd_ref,
                            float *inp, float *weight, float *bias, int B, int T, int C) {
    float *d_out, *d_mean, *d_rstd, *d_inp, *d_weight, *d_bias;
    float *out_tmp, *mean_tmp, *rstd_tmp;
    struct timespec start, end;

    // Allocate device memory
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    cudaMalloc(&d_mean, B * T * sizeof(float));
    cudaMalloc(&d_rstd, B * T * sizeof(float));
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    cudaMalloc(&d_weight, C * sizeof(float));
    cudaMalloc(&d_bias, C * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate host memory for verification
    out_tmp = (float *)mallocCheck(B * T * C * sizeof(float));
    mean_tmp = (float *)mallocCheck(B * T * sizeof(float));
    rstd_tmp = (float *)mallocCheck(B * T * sizeof(float));

    int block_size = 256;

    // Test Kernel0
    int grid_size = CEIL_DIV(B * T, block_size);
    // Reset output
    cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float)));
    cudaCheck(cudaMemset(d_mean, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemset(d_rstd, 0, B * T * sizeof(float)));

    clock_gettime(CLOCK_MONOTONIC, &start);
    layernorm_forward_kernel0<<<grid_size, block_size>>>(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("layernorm_forward_kernel0 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    // Verify results
    cudaCheck(cudaMemcpy(out_tmp, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(mean_tmp, d_mean, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(rstd_tmp, d_rstd, B * T * sizeof(float), cudaMemcpyDeviceToHost));

    if (!layernorm_forward_check(out_tmp, mean_tmp, rstd_tmp, out_ref, mean_ref, rstd_ref, B, T, C)) {
        printf("layernorm_forward_kernel0 failed\n");
    }

    // Test Kernel1
    block_size = 128;
    grid_size = B * T;
    // Reset output
    cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float)));
    cudaCheck(cudaMemset(d_mean, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemset(d_rstd, 0, B * T * sizeof(float)));

    clock_gettime(CLOCK_MONOTONIC, &start);
    // Step 1: Calculate mean and variance
    mv_kernel<<<grid_size, block_size, 2 * block_size * sizeof(float)>>>(d_mean, d_rstd, d_inp, B, T, C, block_size);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Step 2: Normalize and scale
    block_size = 256;
    grid_size = CEIL_DIV(B * T * C, block_size);
    normalization_kernel<<<grid_size, block_size>>>(d_out, d_inp, d_mean, d_rstd, d_weight, d_bias, B, T, C);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("layernorm_forward_kernel1 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    // Verify results
    cudaCheck(cudaMemcpy(out_tmp, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(mean_tmp, d_mean, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(rstd_tmp, d_rstd, B * T * sizeof(float), cudaMemcpyDeviceToHost));

    if (!layernorm_forward_check(out_tmp, mean_tmp, rstd_tmp, out_ref, mean_ref, rstd_ref, B, T, C)) {
        printf("layernorm_forward_kernel1 failed\n");
    }

    // Test Kernel2
    grid_size = B * T;
    block_size = 128;

    // Reset output
    cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float)));
    cudaCheck(cudaMemset(d_mean, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemset(d_rstd, 0, B * T * sizeof(float)));

    clock_gettime(CLOCK_MONOTONIC, &start);
    layernorm_forward_kernel2<<<grid_size, block_size>>>(d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("layernorm_forward_kernel2 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    // Verify results
    cudaCheck(cudaMemcpy(out_tmp, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(mean_tmp, d_mean, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(rstd_tmp, d_rstd, B * T * sizeof(float), cudaMemcpyDeviceToHost));

    if (!layernorm_forward_check(out_tmp, mean_tmp, rstd_tmp, out_ref, mean_ref, rstd_ref, B, T, C)) {
        printf("layernorm_forward_kernel2 failed\n");
    }

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_rstd);
    cudaFree(d_inp);
    cudaFree(d_weight);
    cudaFree(d_bias);
    free(out_tmp);
    free(mean_tmp);
    free(rstd_tmp);
}