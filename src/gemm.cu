#include "gemm.h"
#include "utils.h"
#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <iostream>

namespace swiftware::hpp {

__global__ void dotProduct(float *A, float *B, float *C, int m, int n, int k) {
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

  // Debugging output
  // valuable reference, let's you print from inside kernel
  // printf("Thread (%d, %d): C[%d][%d] = %f\n", row, col, row, col, sum);
}

// __global__ void singleRow(float *A, float *B, float *C, int m, int n, int k) {
//   // one thread = an entire row decomposition
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int row = idx / n;
// }

template<int TileSize>
__global__ void oneDimTile(float *A, float *B, float *C, int m, int n, int k) {
  // now going to use squared tiles
  // let each thread pull in a value from A
  int block_y = blockIdx.y;
  int block_x = blockIdx.x;

  int thread_y = threadIdx.y;
  int thread_x = threadIdx.x;

  int c_i = block_y*blockDim.y + threadIdx.y;
  int c_j = block_x*blockDim.y + threadIdx.x;

  __shared__ float A_tile[TileSize * TileSize];
  __shared__ float B_tile[TileSize * TileSize];

  for (int tileIdx = 0; tileIdx; < )

  // populate Ax
  Ax[thread_y * TileSize + thread_x] = A[c_i * k + c_j];

  __syncthreads();

  // now do the dot product
  float sum = 0.0f;
  for (int l = 0; l < k; ++l) {
    sum += Ax[thread_y * TileSize + l] * B[l * n + c_j];
  }
  C[c_i * n + c_j] += sum;


  

}
__global__ void twoDimTile(float *A, float *B, float *C, int m, int n, int k) {
  // TODO
}

float gemmGpuSingleRowDecomp(int m, int n, int k, const float *h_A,
                             const float *h_B, float *h_C, ScheduleParams Sp) {
  const size_t sizeA = m * k;
  const size_t sizeB = k * n;
  const size_t sizeC = m * n;

  // Allocate mem on GPU
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  cudaMalloc((void **)&d_B, sizeB * sizeof(float));
  cudaMalloc((void **)&d_C, sizeC * sizeof(float));

  // Move mem to gpu
  cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

  // figure out how many threadsPerBlock, size of blocks
  dim3 block(Sp.TileSize1, 1, 1);
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

  // Verify result against CPU calculation
//   float *cpu_C = new float[sizeC];
//   const float tol = 1e-5;
//   bool correct = true;

//   // CPU matrix multiplication for verification
//   for (int i = 0; i < m; i++) {
//     for (int j = 0; j < n; j++) {
//       float sum = 0.0f;
//       for (int l = 0; l < k; l++) {
//         sum += h_A[i * k + l] * h_B[l * n + j];
//       }
//       cpu_C[i * n + j] = sum;

//       // Compare GPU and CPU results
//       if (std::abs(cpu_C[i * n + j] - h_C[i * n + j]) > tol) {
//         correct = false;
//         std::cout << "Mismatch at position (" << i << "," << j << "): ";
//         std::cout << "CPU=" << cpu_C[i * n + j] << ", GPU=" << h_C[i * n + j]
//                   << std::endl;
//       }
//     }
//   }

//   if (correct) {
//     std::cout << "Matrix multiplication results match within tolerance!"
//               << std::endl;
//   }

  // cleanup memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
//   delete[] cpu_C;

  return elapsed;
}

} // namespace swiftware::hpp