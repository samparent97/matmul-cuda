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

__global__ void multiRow(float* A, float* B, float* C, int m, int n, int k,
                         int unroll) {
    // one thread = handles multiple rows of A, therefore computes multiple rows
    // of C
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * unroll;

    // Process unroll number of rows per thread
    for (int i = 0; i < unroll && (idx + i) < m; i++) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[(idx + i) * k + l] * B[l * n + j];
            }
            C[(idx + i) * n + j] = sum;
        }
    }
}

float gemmGpuMultiRowDecomp(int m, int n, int k, const float* h_A,
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
    // dim3 block(Sp.TileSize1, 1, 1);
    // dim3 grid((m + Sp.TileSize1 - 1) / Sp.TileSize1, 1, 1);
    // Giving up on using dim3 for now, just give it ints
    int threadsPerBlock = 1024;
    int blocksPerGrid = (m + threadsPerBlock * Sp.ChunkSize - 1) /
                        (threadsPerBlock * Sp.ChunkSize);

    // cudaDeviceSynchronize();
    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // run kernel
    multiRow<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k,
                                                 Sp.ChunkSize);

    // synchronize
    // cudaDeviceSynchronize();
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
    //   delete[] cpu_C;

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
    int block_y = blockIdx.y;
    int block_x = blockIdx.x;

    int thread_y = threadIdx.y;
    int thread_x = threadIdx.x;

    int C_i = TileSize * block_y + thread_y;
    int C_j = TileSize * block_x + thread_x;

    // Allocating shared memory
    __shared__ float sh_A[TileSize][TileSize];
    __shared__ float sh_B[TileSize][TileSize];

    // Parallel mat mul
    float value = 0;
    for (int tileCount = 0; tileCount < ceil((float)k / TileSize);
         tileCount++) {
        // Load a tile of A into shared memory
        if ((C_i < m) && ((tileCount * TileSize + thread_x) < k)) {
            sh_A[thread_y][thread_x] =
                A[(C_i)*k + tileCount * TileSize + thread_x];
        } else {
            sh_A[thread_y][thread_x] = 0.0f;
        }

        // Load a tile of B into shared memory
        if ((C_j < n) && ((tileCount * TileSize + thread_y) < k)) {
            sh_B[thread_y][thread_x] =
                B[(tileCount * TileSize + thread_y) * n + C_j];
        } else {
            sh_B[thread_y][thread_x] = 0.0f;
        }

        __syncthreads();

        // Perform the dot product
        for (int k = 0; k < TileSize; k++) {
            value += sh_A[thread_y][k] * sh_B[k][thread_x];
        }

        __syncthreads();
    }

    // Assign the calculated value to the correct position in C
    if ((C_i < m) && (C_j < n)) {
        C[C_i * n + C_j] = value;
    }
}

template <int TileSize>
__global__ void twoDimTile(float* A, float* B, float* C, int m, int n, int k) {
    // TODO
    __shared__ float Asub[TileSize][TileSize];
    __shared__ float Bsub[TileSize][TileSize];

    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TileSize + ty;
    int col = blockIdx.x * TileSize + tx;

    // Accumulator for the result of C[row][col]
    float value = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (k + TileSize - 1) / TileSize; t++) {
        // Load tiles into shared memory
        if (row < m && t * TileSize + tx < k) {
            Asub[ty][tx] = A[row * k + t * TileSize + tx];
        } else {
            Asub[ty][tx] = 0.0f;
        }
        if (col < n && t * TileSize + ty < k) {
            Bsub[ty][tx] = B[(t * TileSize + ty) * n + col];
        } else {
            Bsub[ty][tx] = 0.0f;
        }
        __syncthreads();  // Synchronize to ensure all threads load their tiles

        // Perform computation for this tile
        for (int k = 0; k < TileSize; k++) {
            value += Asub[ty][k] * Bsub[k][tx];
        }
        __syncthreads();  // Synchronize to ensure computation is complete
                          // before loading the next tile
    }

    // Write the result to global memory
    if (row < m && col < n) {
        C[row * n + col] = value;
    }
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
    dim3 dim_grid((n + TileSize - 1) / TileSize, (m + TileSize - 1) / TileSize,
                  1);

    // start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // run kernel
    twoDimTile<TileSize><<<dim_grid, dim_block>>>(d_A, d_B, d_C, m, n, k);

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

    return elapsed;
}

}  // namespace swiftware::hpp