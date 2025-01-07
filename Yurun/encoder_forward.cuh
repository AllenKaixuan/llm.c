#include "../llmc/cuda_common.h"
#include "../llmc/cuda_utils.cuh"

int encoder_forward_check(float *out, float *out_ref, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < C; i++) {
                if (fabs(out[b * T * C + t * C + i] - out_ref[b * T * C + t * C + i]) > 1e-5) {
                    printf("mismatch at b=%d, t=%d, i=%d: %f != %f\n", b, t, i, out[b * T * C + t * C + i],
                           out_ref[b * T * C + t * C + i]);
                    return 0;
                }
            }
        }
    }
    return 1;
}

// parallelize over B, T, loop over C
__global__ void encoder_forward_kernel0(float *out, const int *inp, const float *wte, const float *wpe, int B, int T,
                                        int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = b * T + t
    if (idx >= B * T)
        return;
    // int b = idx / T;
    int t = idx % T;
    int ix = inp[idx];

    float *out_bt = out + idx * C;
    const float *wte_ix = wte + ix * C;
    const float *wpe_t = wpe + t * C;
    for (int i = 0; i < C; i++) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
    }
}

// parallelize over B, T, C
__global__ void encoder_forward_kernel1(float *out, const int *inp, const float *wte, const float *wpe, int B, int T,
                                        int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = b * T * C + t * C + c
    if (idx >= B * T * C)
        return;
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;
    int ix = inp[b * T + t];

    float *out_bt = out + idx;
    const float *wte_ix = wte + ix * C + c;
    const float *wpe_t = wpe + t * C + c;
    out_bt[0] = wte_ix[0] + wpe_t[0];
}

// parallelize over B, T, C/4, vectorized
__global__ void encoder_forward_kernel2(float *out, const int *inp, const float *wte, const float *wpe, int B, int T,
                                        int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // idx = b * T * (C / 4) + t * (C / 4) + c
    if (idx >= B * T * (C / 4))
        return;
    int b = idx / (T * (C / 4));
    int t = (idx / (C / 4)) % T;
    int c = idx % (C / 4);
    int ix = inp[b * T + t];

    // Here we use float4 to vectorize the computation, enforce memory alignment
    float4 *out_bt = reinterpret_cast<float4 *>(out + b * T * C + t * C) + c;
    const float4 *wte_ix = reinterpret_cast<const float4 *>(wte + ix * C) + c;
    const float4 *wpe_t = reinterpret_cast<const float4 *>(wpe + t * C) + c;
    out_bt[0] = make_float4(wte_ix->x + wpe_t->x, wte_ix->y + wpe_t->y, wte_ix->z + wpe_t->z, wte_ix->w + wpe_t->w);
}

void encoder_forward_gpu(float *out_ref, int *inp, float *wte, float *wpe, int B, int T, int C, int V, int maxT) {
    // Device pointers
    float *d_out, *d_wte, *d_wpe, *out_tmp;
    int *d_inp;
    struct timespec start, end;

    // Print input dimensions
    printf("B: %d, T: %d, C: %d, V: %d, maxT: %d\n", B, T, C, V, maxT);

    // Allocate device memory and copy input data
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    cudaMalloc(&d_inp, B * T * sizeof(int));
    cudaMalloc(&d_wte, C * V * sizeof(float));
    cudaMalloc(&d_wpe, C * maxT * sizeof(float));
    cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wte, wte, C * V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wpe, wpe, C * maxT * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate temporary host memory for validation
    out_tmp = (float *)mallocCheck(B * T * C * sizeof(float));


    // Test kernel0
    int block_size = 64;
    int grid_size = CEIL_DIV(B * T, block_size);
    cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float))); // Reset output
    clock_gettime(CLOCK_MONOTONIC, &start);
    encoder_forward_kernel0<<<grid_size, block_size>>>(d_out, d_inp, d_wte, d_wpe, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("encoder_forward_kernel0 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);
    cudaCheck(cudaMemcpy(out_tmp, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    if (!encoder_forward_check(out_ref, out_tmp, B, T, C)) {
        printf("encoder_forward_kernel0 failed\n");
    }

    // Test kernel1
    block_size = 512;
    grid_size = CEIL_DIV(B * T * C, block_size);
    cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float))); // Reset output
    clock_gettime(CLOCK_MONOTONIC, &start);
    encoder_forward_kernel1<<<grid_size, block_size>>>(d_out, d_inp, d_wte, d_wpe, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("encoder_forward_kernel1 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);
    cudaCheck(cudaMemcpy(out_tmp, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    if (!encoder_forward_check(out_ref, out_tmp, B, T, C)) {
        printf("encoder_forward_kernel1 failed\n");
    }

    // Test kernel2
    block_size = 256;
    grid_size = CEIL_DIV(B * T * C / 4, block_size);
    cudaCheck(cudaMemset(d_out, 0, B * T * C * sizeof(float))); // Reset output
    clock_gettime(CLOCK_MONOTONIC, &start);
    encoder_forward_kernel2<<<grid_size, block_size>>>(d_out, d_inp, d_wte, d_wpe, B, T, C);
    cudaCheck(cudaPeekAtLastError());
    cudaCheck(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("encoder_forward_kernel2 time: %f ms\n",
           (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1e6);
    cudaCheck(cudaMemcpy(out_tmp, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    if (!encoder_forward_check(out_ref, out_tmp, B, T, C)) {
        printf("encoder_forward_kernel2 failed\n");
    }

    // Free allocated memory
    cudaFree(d_out);
    cudaFree(d_inp);
    cudaFree(d_wte);
    cudaFree(d_wpe);
    free(out_tmp);
}
