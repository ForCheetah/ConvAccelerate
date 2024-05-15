// @file im2col_conv.c
//
//  \date Created on: Sep 30, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <stdbool.h>
#include <cblas.h>
#include "common_types.h"
#include "conv_layers.h"
#include "stdio.h"
#include "string.h"

//From Berkeley Vision's Caffe
static inline bool Is_in_range(int a, int b) {  // 注意 ： 负值比正值大    
  return (unsigned int)a < (unsigned int)(b);
}

void Im2Cube(const float *input, const TensorDim in_dim, float* in_cube, const ColDim in_2D_dim, 
      const CubeDim in_cube_dim, const TensorDim weight_dim, const TensorDim out_dim, const int CUBE_row,  
      const int CUBE_col, const int pad_h, const int pad_w, const int stride_h, const int stride_w){

  int channel_size = in_dim.h * in_dim.w;
  int in_start = 0;
  int row_2D;
  int col_2D;
  int cube_num;
  
  for (int channel = 0; channel < in_dim.c; channel++, in_start += channel_size) {
    for (int kernel_row = 0; kernel_row < weight_dim.h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < weight_dim.w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        row_2D = channel*weight_dim.h*weight_dim.w + kernel_row*weight_dim.w + kernel_col;
        for (int output_rows = 0; output_rows<out_dim.h; output_rows++) {
          if (!Is_in_range(input_row, in_dim.h)){
            for (int output_cols = 0; output_cols<out_dim.w; output_cols++) {
              col_2D = output_rows*out_dim.w + output_cols;
              cube_num = (row_2D/CUBE_row) * in_cube_dim.col + (col_2D/CUBE_col);
              in_cube[cube_num*CUBE_row*CUBE_col + (row_2D%CUBE_row)*CUBE_col + col_2D%CUBE_col] = 0;
              // printf("in_cube[%d] = 0 \n", cube_num*CUBE_row*CUBE_col + (row_2D%CUBE_row)*CUBE_col + col_2D%CUBE_col);
              // *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col;
            for (int output_cols = 0; output_cols<out_dim.w; output_cols++) {
              if (Is_in_range(input_col, in_dim.w)) {
                col_2D = output_rows*out_dim.w + output_cols;
                cube_num = (row_2D/CUBE_row) * in_cube_dim.col + (col_2D/CUBE_col);
                in_cube[cube_num*CUBE_row*CUBE_col + (row_2D%CUBE_row)*CUBE_col + col_2D%CUBE_col] = input[in_start + input_row * in_dim.w + input_col];
                // printf("in_cube[%d] = input[%d] \n", cube_num*CUBE_row*CUBE_col + (row_2D%CUBE_row)*CUBE_col + col_2D%CUBE_col, in_start + input_row * in_dim.w + input_col);
              } else {
                col_2D = output_rows*out_dim.w + output_cols;
                cube_num = (row_2D/CUBE_row) * in_cube_dim.col + (col_2D/CUBE_col);
                in_cube[cube_num*CUBE_row*CUBE_col + (row_2D%CUBE_row)*CUBE_col + col_2D%CUBE_col] = 0;
                // printf("in_cube[%d] = 0 \n", cube_num*CUBE_row*CUBE_col + (row_2D%CUBE_row)*CUBE_col + col_2D%CUBE_col);
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void MemSetZero(float* CalCubeLast, int elemNum){
  for(int i=0; i<elemNum; i++){
    CalCubeLast[i] = 0;
  }
}

void show_cube(float* cube, int CUBE_row, int CUBE_col){
  for(int i=0; i<CUBE_row; i++){
    for(int j=0; j<CUBE_col; j++){
      printf("%.0f ", *cube++);
    }
    printf("\n");
  }
  printf("********************************************************\n");
}

void CubeMatmul(float* weight_cube, const CubeDim weight_cube_dim, float* in_cube, const CubeDim in_cube_dim, 
    float *output, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    float* CalCube, const int CUBE_row, const int CUBE_col){
      int in_cube_index;
      int weight_cube_index;
      int out_start;
      int row_range;
      int copy_range;
      for(int row=0; row<out_cube_dim.row; row++){
        for(int col=0; col<out_cube_dim.col; col++){
          row_range = (row+1)*CUBE_row>out_2D_dim.row?out_2D_dim.row%CUBE_row:CUBE_row;
          copy_range = (col+1)*CUBE_col>out_2D_dim.col?out_2D_dim.col%CUBE_col:CUBE_col;
          MemSetZero(CalCube, CUBE_row*CUBE_col);
          for(int k=0; k<weight_cube_dim.col; k++){
            weight_cube_index = (row*weight_cube_dim.col + k) * CUBE_row*CUBE_col;
            in_cube_index = (k*in_cube_dim.col + col) * CUBE_row*CUBE_col;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                CUBE_row, CUBE_col, CUBE_col, 1.0, weight_cube+weight_cube_index, CUBE_col, in_cube+in_cube_index, CUBE_col, 1.0, CalCube, CUBE_col);
          }
          // show_cube(CalCube, CUBE_row, CUBE_col);
          for(int out_row=0; out_row<row_range; out_row++){
            out_start = (row*CUBE_row+out_row)*out_2D_dim.col + col*CUBE_col;
            memcpy(output+out_start, CalCube+out_row*CUBE_col, sizeof(float)*copy_range);
          }

        }
      }
}

void Weight2Cube(const float *weight, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim, const int CUBE_row, const int CUBE_col){
  
  for(int i=0; i<weight_2D_dim.row; i++){
    int weight_cube_row = i / CUBE_row;
    for(int j=0; j<weight_2D_dim.col; j++){
      int num_cube = j / CUBE_col + weight_cube_row*weight_cube_dim.col;
      weight_cube[num_cube*CUBE_row*CUBE_col+(i%CUBE_row)*CUBE_col+(j%CUBE_col)] = weight[i*weight_2D_dim.col+j];
      // printf("num_cube:%d  weight_cube[%d]=weight[%d]\n",num_cube, num_cube*CUBE_row*CUBE_col+(i%CUBE_row)*CUBE_col+(j%CUBE_col), i*weight_2D_dim.col+j);
    }
  }

}

void Im2CubeConvLayer(const float *input, const TensorDim in_dim, const float *weight, const TensorDim weight_dim, 
    float* bias, float *output, const TensorDim out_dim, 
    float* in_cube, const ColDim in_2D_dim, const CubeDim in_cube_dim, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim,
    float* CalCube, const int CUBE_row, const int CUBE_col, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int group, const int pad, const int stride, const int bias_en) {

    
    Weight2Cube(weight, weight_cube, weight_2D_dim, weight_cube_dim, CUBE_row, CUBE_col);
    Im2Cube(input, in_dim, in_cube, in_2D_dim, in_cube_dim, weight_dim, out_dim, CUBE_row,  
      CUBE_col, pad, pad, stride, stride);
    // show_cube(in_cube, in_cube_dim.row*16*in_cube_dim.col, 16);
    // show_cube(weight_cube, weight_cube_dim.row*16, weight_cube_dim.col*16);

    CubeMatmul(weight_cube, weight_cube_dim, in_cube, in_cube_dim, output, out_2D_dim, out_cube_dim,
      CalCube, CUBE_row, CUBE_col);

    // for(int i=0; i<out_2D_dim.row; i++){
    //   for(int j=0; j<out_2D_dim.col; j++){
    //     printf("%.0f   ", output[i*out_2D_dim.col+j]);
    //   }
    //   printf("\n");
    // }
    // printf("\n========================================\n");
}
