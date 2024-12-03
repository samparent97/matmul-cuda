#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#include "gemm.h"
#include "utils.h"

namespace swiftware::hpp {

__global__ void singleRowInput(float* A, float* B, float* C, int m, int n,
                               int k) {
    // one thread handles a whole row af A, therefore computes a whole row of C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // not sharing memory, because threads don't reuse much within blocks
    // pre-tiling
    if (idx < m) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[idx * k + l] * B[l * n + j];
            }
            C[idx * n + j] = sum;
        }
    }
}

float gemmGpuSingleRowDecomp(int m, int n, int k, const float* h_A,
                             const float* h_B, float* h_C, ScheduleParams Sp) {
    const size_t sizeA = m * k;
    const size_t sizeB = k * n;
    const size_t sizeC = m * n;

    // Allocate mem on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));

    // Move mem to gpu
    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // figure out how many threadsPerBlock, size of blocks
    int threadsPerBlock = 1024;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // run kernel
    singleRowInput<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    // synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // get timing measurement
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // get memory back
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed;
}

__global__ void dotProduct(float* A, float* B, float* C, int m, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / n;
    int col = idx % n;
    float sum = 0.0f;

    if (row < m && col < n) {
        for (int l = 0; l < k; ++l) {
            sum += A[row * k + l] * B[l * n + col];
        }
    }
    C[idx] = sum;
}

float gemmGpuSingleElementDecomp(int m, int n, int k, const float* h_A,
                                 const float* h_B, float* h_C,
                                 ScheduleParams Sp) {
    const size_t sizeA = m * k;
    const size_t sizeB = k * n;
    const size_t sizeC = m * n;

    // Allocate mem on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));

    // Move mem to gpu
    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    // figure out how many threadsPerBlock, size of blocks
    dim3 block(1, 1, 1);
    dim3 grid((m * n + block.x - 1) / block.x, 1, 1);

    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);

    // run kernel
    dotProduct<<<grid, block>>>(d_A, d_B, d_C, m, n, k);

    // synchronize
    cudaDeviceSynchronize();
    cudaEventCreate(&stop);

    // get timing measurement
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // get memory back
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return elapsed;
}

template <int TileSize>
__global__ void oneDimTile(float* A, float* B, float* C, int m, int n, int k) {
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Working on C[i,j]
    int i = TileSize * by + ty;
    int j = TileSize * bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TileSize][TileSize];
    __shared__ float sh_B[TileSize][TileSize];

    // Parallel mat mul
    float value = 0;
    for (int tileCount = 0; tileCount < ceil((float)k / TileSize);
         tileCount++) {
        // Load Tiles into shared memory
        if ((i < m) && ((tileCount * TileSize + tx) < k))
            sh_A[ty][tx] = A[(i)*k + tileCount * TileSize + tx];
        else
            sh_A[ty][tx] = 0.0f;

        if (((tileCount * TileSize + ty) < k) && (j < n))
            sh_B[ty][tx] = B[(tileCount * TileSize + ty) * n + j];
        else
            sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TileSize; k++) value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((i < m) && (j < n)) C[i * n + j] = value;
}

float gemmGpuOneDimTile(int m, int n, int k, const float* h_A, const float* h_B,
                        float* h_C, ScheduleParams Sp) {
    const size_t sizeA = m * k;
    const size_t sizeB = k * n;
    const size_t sizeC = m * n;

    // Allocate mem on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));

    // Move mem to gpu
    cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int TileSize = 32;

    // Kernel execution
    dim3 dim_block(TileSize, TileSize, 1);
    dim3 dim_grid(ceil(k / (float)(TileSize)), ceil(m / (float)(TileSize)), 1);

    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);

    // run kernel
    oneDimTile<TileSize><<<dim_grid, dim_block>>>(d_A, d_B, d_C, m, n, k);

    // synchronize
    cudaDeviceSynchronize();
    cudaEventCreate(&stop);

    // get timing measurement
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // get memory back
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // cleanup memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return elapsed;
}

}  // namespace swiftware::hpp