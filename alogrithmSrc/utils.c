// @file utils.c
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "utils.h"
#include "common_types.h"

// Use mean square error for comparing two tensors.
#define MEASURE_MSE
void RandInitF32(float *p_data, int N) {
    int k;
    for (k = 0; k < N; k++) {
        float val = rand() % 10;
        //float val = 2*((float)rand() / RAND_MAX) - 1.0;
        p_data[k] = val;
    }
}

void SeqInitF32(float *p_data, int N) {
    int k;
    for (k = 0; k < N; k++) {
        p_data[k] = k;
    }
}


bool TensorCompare(const float *t1, const float *t2, TensorDim dim) {

  bool ret = true;
  int N = TensorSize(dim);
  float mse = 0;
  for (int n = 0; n < dim.n; ++n) {
    for (int c = 0; c < dim.c; c++) {
      for (int h = 0; h < dim.h; ++h) {
        for (int w = 0; w < dim.w; w++) {
          int addr = w + dim.w * (h + dim.h * (c + dim.c * n));
          // printf(" %.0f=%.0f \t", t1[addr], t2[addr]);
#ifdef MEASURE_MSE
          mse += pow(t1[addr] - t2[addr], 2);
#else
          if (fabs(t1[addr] - t2[addr]) > 1e-4) {
            printf("Mismatch at [%d, %d, %d, %d]\n", n, c, h, w);
            printf("Tensor 1 val = %f\t Tensor 2 val = %f\n", t1[addr], t2[addr]);
            ret = false;
          }
#endif
        }
      }
    }
  }
#ifdef MEASURE_MSE
  mse = mse / N;
  if (mse > 1e-6) {
    ret = false;
  }
  printf("\nMSE = %.8f\n", mse);
#endif
  return ret;
}
int TensorSize(TensorDim dim) {
  return dim.n * dim.c * dim.h * dim.w;
}

