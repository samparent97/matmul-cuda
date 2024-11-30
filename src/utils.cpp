// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include "utils.h"
#include <fstream>
#include <sstream>


namespace swiftware::hpp {


  CSR *dense2CSR(const DenseMatrix *A) {
      auto *OutMat = new CSR(A->m, A->n);
      // TODO implement dense2CSR
      return OutMat;
  }


  // Do not change the following function signatures
  DenseMatrix *readCSV(const std::string &filename, bool removeFirstRow) {
    std::ifstream file(filename);
    std::string line, word;
    // determine number of columns in file
    std::vector<std::string> lines;
    int cntr = 0;
    while (getline(file, line)) {
      lines.push_back(line);
    }
    if (removeFirstRow) {
      lines.erase(lines.begin());
    }
    std::vector<std::vector<float>> valuesPerLine(lines.size());
    for (int i = 0; i < lines.size(); i++) {
      std::stringstream lineStream(lines[i]);
      while (getline(lineStream, word, ',')) {
        valuesPerLine[i].push_back(std::stof(word));
      }
    }
    auto *OutMat = new DenseMatrix(valuesPerLine.size(), valuesPerLine[0].size());
    auto *data = OutMat->data.data();
    int ncol = OutMat->n;
    for (int i = 0; i < valuesPerLine.size(); i++) {
      size_t cols = valuesPerLine[i].size();
      for (uint j = 0; j < cols; j++) {
        data[i * ncol + j] = valuesPerLine[i][j];
      }
    }
    return OutMat;
  }

  DenseMatrix *samplingDense(const DenseMatrix *A, float samplingRate) {
    auto *OutMat = new DenseMatrix(A->m, A->n);
    int stride = 1. / samplingRate;
    for (int i = 0; i < A->m; i++) {
      for (int j = 0; j < A->n; j++) {
        int idx = i * A->m + j;
        if (idx % stride)
          OutMat->data[idx] = 0;
        else
          OutMat->data[idx] = A->data[idx];
      }
    }
    return OutMat;
  }


}