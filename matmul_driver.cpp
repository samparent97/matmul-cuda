// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <benchmark/benchmark.h>
#include <chrono>

#include "err_code.h"
#include "gemm.h"

#define NUM_THREADS 8

static void BM_GEMM(benchmark::State &state,
                    void (*gemmImpl1)(int M, int N, int K, const float *A, const float *B, float *C,
                                      swiftware::hpp::ScheduleParams Sp)) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int t1 = state.range(3);
  int t2 = state.range(4);
  int cs = state.range(5); // chunk size
  int nt = state.range(6); // number of threads
  // TOOO : add other parameters if needed

  auto *A = new swiftware::hpp::DenseMatrix(m, k);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  for (int i = 0; i < m * k; ++i) {
    A->data[i] = 1.0;
  }
  for (int i = 0; i < k * n; ++i) {
    B->data[i] = 1.0;
  }

  for (auto _: state) {
    gemmImpl1(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2, nt, cs));
  }
  delete A;
  delete B;
  delete C;

}

static void BM_MATMUL_CUDA(benchmark::State &state,
                             float (*matmulImpl1)(int m, int n, int k, const float *A, const float *B, float *C, 
                                                 swiftware::hpp::ScheduleParams Sp)) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    swiftware::hpp::ScheduleParams sp(state.range(3), state.range(4), 8, 1);

    size_t size = m * sizeof(float);
    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }

    const char *device_name = "GPU";

    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        float elapsed = matmulImpl1(m, n, k, A->data.data(), B->data.data(), C->data.data(), sp);
        state.SetIterationTime(elapsed);
    }

    delete A;
    delete B;
    delete C;
}


//Args format (m, n, k, Sp.TileSize1, Sp.TileSize2, chunk size, number of threads, samplingRatePercentage)
BENCHMARK_CAPTURE(BM_GEMM, gemm_optimized, swiftware::hpp::gemmEfficientParallel)
    // ->Args({512, 512, 512, 64, 32, 1, NUM_THREADS, 1})
    // ->Args({1024, 1024, 1024, 64, 32, 1, NUM_THREADS, 1})
    // ->Args({2048, 2048, 2048, 64, 32, 1, NUM_THREADS, 1})
    ->Args({4096, 4096, 4096, 64, 32, 1, NUM_THREADS, 1})
    ->Args({10000, 10000, 50, 64, 32, 1, NUM_THREADS, 1})
    ->Args({50, 50, 10000, 64, 32, 1, NUM_THREADS, 1})
;

BENCHMARK_CAPTURE(BM_MATMUL_CUDA, cuda_matmul, swiftware::hpp::gemmGpuSingleRowDecomp)
    // ->Args({512, 512, 512, 256, 32})->UseManualTime()->Iterations(10)
    // ->Args({1024, 1024, 1024, 256, 32})->UseManualTime()->Iterations(10)
    // ->Args({2048, 2048, 2048, 256, 32})->UseManualTime()->Iterations(10)
    // ->Args({64, 64, 64, 32, 32})->UseManualTime()->Iterations(10)

    // ->Args({8, 8, 8, 1, 32})->UseManualTime()->Iterations(10)



    // ->Args({4096, 4096, 4096, 1, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 2, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 4, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 8, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 16, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 32, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 64, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 128, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 256, 32})->UseManualTime()->Iterations(10)
    // ->Args({4096, 4096, 4096, 512, 32})->UseManualTime()->Iterations(10)
    ->Args({4096, 4096, 4096, 1024, 32})
    ->UseManualTime()
    ->Iterations(10)

    // ->Args({10000, 10000, 50, 16, 32})->UseManualTime()->Iterations(10)
    // ->Args({10000, 10000, 50, 32, 32})->UseManualTime()->Iterations(10)
    // ->Args({10000, 10000, 50, 64, 32})->UseManualTime()->Iterations(10)
    // ->Args({10000, 10000, 50, 128, 32})->UseManualTime()->Iterations(10)
    // ->Args({10000, 10000, 50, 256, 32})->UseManualTime()->Iterations(10)
    // ->Args({10000, 10000, 50, 512, 32})->UseManualTime()->Iterations(10)
    // ->Args({10000, 10000, 50, 1024, 32})->UseManualTime()->Iterations(10)

    // ->Args({50, 50, 10000, 16, 32})->UseManualTime()->Iterations(10)
    // ->Args({50, 50, 10000, 32, 32})->UseManualTime()->Iterations(10)
    // ->Args({50, 50, 10000, 64, 32})->UseManualTime()->Iterations(10)
    // ->Args({50, 50, 10000, 128, 32})->UseManualTime()->Iterations(10)
    // ->Args({50, 50, 10000, 256, 32})->UseManualTime()->Iterations(10)
    // ->Args({50, 50, 10000, 512, 32})->UseManualTime()->Iterations(10)
    // ->Args({50, 50, 10000, 1024, 32})->UseManualTime()->Iterations(10)
;

BENCHMARK_CAPTURE(BM_MATMUL_CUDA, cuda_matmul, swiftware::hpp::gemmGpuOneDimTile)
    ->Args({4096, 4096, 4096, 32, 32})->UseManualTime()->Iterations(10)
;


BENCHMARK_MAIN();

