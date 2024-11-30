// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef LAB1_DENSE_MATMUL_DEF_H
#define LAB1_DENSE_MATMUL_DEF_H

#include <vector>


namespace swiftware::hpp {

  struct ScheduleParams {
    int NumThreads;
    int ChunkSize;
    // TODO: Add more parameters if needed
    int TileSize1;
    int TileSize2;
    ScheduleParams(int TileSize1, int TileSize2, int NT, int CS):
                                                                   TileSize1(TileSize1), TileSize2(TileSize2), NumThreads(NT), ChunkSize(CS){}
  };

  // please do not change below lines
  struct DenseMatrix{
    int m;
    int n;
    std::vector<float> data;
    DenseMatrix(int m, int n): m(m), n(n), data(m*n){}
  };

  struct CSR{
    int m;
    int n;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<float> data;
    CSR(int m, int n): m(m), n(n){}
  };

}

#endif //LAB1_DENSE_MATMUL_DEF_H
