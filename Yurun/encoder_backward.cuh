#include "../llmc/cuda_common.h"
#include "../llmc/cuda_utils.cuh"

int encoder_backward_check(float *dwte, float *dwpe, float *dwte_ref, float *dwpe_ref, int V, int T, int C) {
    for (int i = 0; i < V * C; i++) {
        if (fabs(dwte[i] - dwte_ref[i]) > 1e-5) {
            printf("mismatch at dwte[%d]: %f != %f\n", i, dwte[i], dwte_ref[i]);
            // return 0;
        }
    }
    for (int i = 0; i < T * C; i++) {
        if (fabs(dwpe[i] - dwpe_ref[i]) > 1e-5) {
            printf("mismatch at dwpe[%d]: %f != %f\n", i, dwpe[i], dwpe_ref[i]);
            // return 0;
        }
    }
    return 1;
}

// parallelize over B, T, C
__global__ void encoder_backward_kernel0(float *dwte, float *dwpe, const float *dout, const int *inp, int B, int T,
                                         int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = b * T * C + t * C + i
    if (idx >= B * T * C)
        return;
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int i = idx % C;
    int ix = inp[b * T + t];

    const float *dout_bt = dout + idx;
    float *dwte_ix = dwte + ix * C + i;
    float *dwpe_t = dwpe + t * C + i;

    atomicAdd(dwte_ix, *dout_bt);
    atomicAdd(dwpe_t, *dout_bt);
}

// parallelize over B, T, loop over C to avoid atomicAdd
__global__ void encoder_backward_kernel1(float *dwte, float *dwpe, const float *dout, const int *inp, int B, int T,
                                         int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    for (int i = 0; i < B * T; i++) {
        int t = i % T;
        int ix = inp[i];
        float dout_btc = dout[i * C + c];
        dwte[ix * C + c] += dout_btc;
        dwpe[t * C + c] += dout_btc;
    }
}

// This is the same as encoder_backward_kernel0 for dwte
// Since the inp is not deterministic, we cannot remove atomicAdd for dwte
__global__ void encoder_backward_dwte_kernel(float *dwte, const float *dout, const int *inp, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C)
        return;
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int i = idx % C;
    int ix = inp[b * T + t];

    const float *dout_bt = dout + b * T * C + t * C + i;
    float *dwte_ix = dwte + ix * C + i;

    atomicAdd(dwte_ix, *dout_bt);
}

// parallelize over T, C/4, vectorized, avoid atomicAdd since dwpe is deterministic
__global__ void encoder_backward_dwpe_kernel(float *dwpe, const float *dout, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= T * C)
        return;

    float dwpe_t[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int t = idx / C;
    int c = idx % C;

    for (int b = 0; b < B; b++) {
        const float4 dout_bt = *reinterpret_cast<const float4 *>(dout + b * T * C + idx);
        dwpe_t[0] += dout_bt.x;
        dwpe_t[1] += dout_bt.y;
        dwpe_t[2] += dout_bt.z;
        dwpe_t[3] += dout_bt.w;
    }

    float4 *dwpe_tp = reinterpret_cast<float4 *>(dwpe + t * C + c);
    float4 dwpe_t4 = *dwpe_tp;

    dwpe_t4.x += dwpe_t[0];
    dwpe_t4.y += dwpe_t[1];
    dwpe_t4.z += dwpe_t[2];
    dwpe_t4.w += dwpe_t[3];

    *dwpe_tp = dwpe_t4;
}

void encoder_backward_gpu(float *dwte, float *dwpe, float *dwte_ref, float *dwpe_ref, float *dout, int *inp, int B,
                          int T, int C, int V) {
    float *d_dwte, *d_dwpe, *d_dout, *dwte_tmp, *dwpe_tmp;
    int *d_inp;
    struct timespec start, end;

    // Allocate device memory and copy input data
    cudaCheck(cudaMalloc(&d_dwte, V * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dwpe, T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate temporary host memory for validation
    dwte_tmp = (float *)mallocCheck(V * C * sizeof(float));
    dwpe_tmp = (float *)mallocCheck(T * C * sizeof(float));

    int block_size;

    // Test kernel0
    block_size = 1024;
    int grid_size = CEIL_DIV(B * T * C, block_size);
    // reset the output
    cudaCheck(cudaMemcpy(d_dwte, dwte, V * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dwpe, dwpe, T * C * sizeof(float), cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_MONOTONIC, &start);
    encoder_backward_kernel0<<<grid_size, block_size>>>(d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("encoder_backward_kernel0 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    cudaCheck(cudaMemcpy(dwte_tmp, d_dwte, V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dwpe_tmp, d_dwpe, T * C * sizeof(float), cudaMemcpyDeviceToHost));

    if (!encoder_backward_check(dwte_tmp, dwpe_tmp, dwte_ref, dwpe_ref, V, T, C)) {
        printf("encoder_backward_kernel0 failed\n");
    }

    // Test kernel1
    block_size = 128;
    grid_size = CEIL_DIV(C, block_size);
    // reset the output
    cudaCheck(cudaMemcpy(d_dwte, dwte, V * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dwpe, dwpe, T * C * sizeof(float), cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_MONOTONIC, &start);
    encoder_backward_kernel1<<<grid_size, block_size>>>(d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("encoder_backward_kernel1 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    cudaCheck(cudaMemcpy(dwte_tmp, d_dwte, V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dwpe_tmp, d_dwpe, T * C * sizeof(float), cudaMemcpyDeviceToHost));

    if (!encoder_backward_check(dwte_tmp, dwpe_tmp, dwte_ref, dwpe_ref, V, T, C)) {
        printf("encoder_backward_kernel1 failed\n");
    }

    // Test kernel2
    block_size = 1024;
    grid_size = CEIL_DIV(B * T * C, block_size);
    // reset the output
    cudaCheck(cudaMemcpy(d_dwte, dwte, V * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dwpe, dwpe, T * C * sizeof(float), cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_MONOTONIC, &start);
    encoder_backward_dwte_kernel<<<grid_size, block_size>>>(d_dwte, d_dout, d_inp, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    const int block_size_dwpe = 256;
    const int grid_size_dwpe = CEIL_DIV(T * C / 4, block_size_dwpe);
    encoder_backward_dwpe_kernel<<<grid_size_dwpe, block_size_dwpe>>>(d_dwpe, d_dout, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("encoder_backward_kernel2 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);

    cudaCheck(cudaMemcpy(dwte_tmp, d_dwte, V * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dwpe_tmp, d_dwpe, T * C * sizeof(float), cudaMemcpyDeviceToHost));

    if (!encoder_backward_check(dwte_tmp, dwpe_tmp, dwte_ref, dwpe_ref, V, T, C)) {
        printf("encoder_backward_kernel2 failed\n");
    }

    // Free memory
    cudaCheck(cudaFree(d_dwte));
    cudaCheck(cudaFree(d_dwpe)); 
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    free(dwte_tmp);
    free(dwpe_tmp);
}