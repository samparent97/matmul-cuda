// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#ifndef LAB2_GEMM_H
#define LAB2_GEMM_H

#include "def.h"

namespace swiftware::hpp {

  // please do not change below
/// \brief Matrix-matrix multiplication
/// \param m Number of rows of A and C
/// \param n Number of columns of B and C
/// \param k Number of columns of A and rows of B
/// \param A Matrix A
/// \param B Matrix B
/// \param C Matrix C
  void gemmEfficientParallel(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);
  float gemmGpuSingleRowDecomp(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);  
  float vectorMultiplyWrapper(float* h_A, float* h_B, float* h_C, int N);
  float gemmGpuOneDimTile(int m, int n, int k, const float *h_A,
                             const float *h_B, float *h_C, ScheduleParams Sp);
}


#endif //LAB2_GEMM_H
