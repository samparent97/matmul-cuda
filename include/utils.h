// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef LAB1_DENSE_MATMUL_UTILS_H
#define LAB1_DENSE_MATMUL_UTILS_H

#include <string>
#include "def.h"

namespace swiftware::hpp {

  //TODO add necessary includes

  // Do not change the following function signatures
  /// \brief Read a CSV file and store it in a DenseMatrix
  /// \param filename Path to the CSV file
  /// \param OutMat Pointer to the DenseMatrix to store the data
  /// \param removeFirstRow Whether to remove the first row of the CSV file
  DenseMatrix * readCSV(const std::string &filename, bool removeFirstRow = false);

  /// \brief Convert a DenseMatrix to a CSR matrix
  /// \param A DenseMatrix to convert
  /// \return Pointer to the CSR matrix
  CSR *dense2CSR(const DenseMatrix *A);

  /// \brief Sample a DenseMatrix with a given sampling rate
  /// \param A DenseMatrix to sample
  /// \param samplingRate Sampling rate: a value between 0 and 1 that specifies the fraction of elements to sample.
  /// For example, a sampling rate of 0.1 means that 10% of the elements will be sampled which makes a sparse matrix
  /// with 90% of zeros (sparsity ratio = 90%).
  /// \return Pointer to the sampled DenseMatrix
  DenseMatrix *samplingDense(const DenseMatrix *A, float samplingRate);

}

#endif //LAB1_DENSE_MATMUL_UTILS_H
