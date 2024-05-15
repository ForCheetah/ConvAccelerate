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
#include "stdlib.h"

//From Berkeley Vision's Caffe
static inline bool Is_in_rangeA(int a, int b) {  // 注意 ： 负值比正值大    
  return (unsigned int)a < (unsigned int)(b);
}

// nchw -> 2D  support: pad
void Im2ColCopy(const float *input, const TensorDim in_dim, float* in_2D, const ColDim in_2D_dim, 
      const CubeDim in_cube_dim, const TensorDim weight_dim, const TensorDim out_dim, const int CUBE_row,  
      const int CUBE_col, const int pad_h, const int pad_w, const int stride_h, const int stride_w){

  int channel_size = in_dim.h * in_dim.w;
  int in_start = 0;
  int in_2D_start = 0;
  float* zero_list = (float*)calloc(out_dim.w, sizeof(float));
  
  for (int channel = 0; channel < in_dim.c; channel++, in_start += channel_size) {
    for (int kernel_row = 0; kernel_row < weight_dim.h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < weight_dim.w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        for (int output_rows = 0; output_rows<out_dim.h; output_rows++) {
          if (!Is_in_rangeA(input_row, in_dim.h)){
            memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*out_dim.w);
            // printf(" copy range::::: :%d   \n", out_dim.w);
            in_2D_start += out_dim.w;
          } else {
            int input_col = -pad_w + kernel_col;
            if(input_col <0){
              memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*(-input_col));
              // printf(" copy range -----:%d   \n", -input_col);
              in_2D_start -= input_col;
              if(input_col+out_dim.w>in_dim.w){
                int overflow_num = input_col+out_dim.w - in_dim.w;
                memcpy(in_2D+in_2D_start, input+in_start+input_row*in_dim.w, sizeof(float)*(out_dim.w-overflow_num+input_col));
                in_2D_start += out_dim.w-overflow_num+input_col;
                // printf(" copy range :%d   \n", out_dim.w-overflow_num+input_col);
                memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*overflow_num);
                in_2D_start += overflow_num;
                // printf(" copy range :%d   \n", overflow_num);
              }else{
                memcpy(in_2D+in_2D_start, input+in_start+input_row*in_dim.w, sizeof(float)*(out_dim.w+input_col));
                in_2D_start += out_dim.w+input_col;
              }
              
              // printf(" copy range :%d   \n", out_dim.w+input_col);
            }else if(input_col+out_dim.w>in_dim.w){
              int overflow_num = input_col+out_dim.w - in_dim.w;
              memcpy(in_2D+in_2D_start, input+in_start+input_row*in_dim.w+input_col, sizeof(float)*(out_dim.w-overflow_num));
              in_2D_start += out_dim.w-overflow_num;
              memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*overflow_num);
              in_2D_start += overflow_num;
              // printf(" copy range :%d   \n", out_dim.w-overflow_num);
            }else{
              memcpy(in_2D+in_2D_start, input+in_start+input_row*in_dim.w+input_col, sizeof(float)*out_dim.w);
              in_2D_start += out_dim.w;
              // printf(" copy range :%d   \n", out_dim.w);
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
  free(zero_list);
}

void Col2CubeCopy(float* in_2D, const ColDim in_2D_dim, float* in_cube, const CubeDim in_cube_dim, const int CUBE_row, const int CUBE_col){
  int in_2D_start;
  int in_cube_start;
  int copy_range;
  int row_range;
  for(int row=0; row<in_cube_dim.row; row ++){
    row_range = (row +1)*CUBE_row > in_2D_dim.row ? in_2D_dim.row-row*CUBE_row : CUBE_row;
    for(int col=0; col<in_cube_dim.col; col++){
      copy_range = (col+1)*CUBE_col > in_2D_dim.col ? in_2D_dim.col-col*CUBE_col : CUBE_col;
      in_cube_start = row*CUBE_row*in_cube_dim.col*CUBE_col + col*CUBE_row*CUBE_col;
      for(int itera=0; itera<row_range; itera++){
        in_2D_start = (row*CUBE_row+itera)*in_2D_dim.col + col*CUBE_col;
        memcpy(in_cube+in_cube_start, in_2D+in_2D_start, sizeof(float)*copy_range);
        // printf(" copy start: %d*%d+%d   tar: %d\n", (row*CUBE_row+itera), in_2D_dim.col, col*CUBE_col, in_cube_start);
        in_cube_start += CUBE_col;
      }
    }
  }
}

void MemSetZeroA(float* CalCubeLast, int elemNum){
  for(int i=0; i<elemNum; i++){
    CalCubeLast[i] = 0;
  }
}

void show_cubeA(float* cube, int CUBE_row, int CUBE_col){
  for(int i=0; i<CUBE_row; i++){
    for(int j=0; j<CUBE_col; j++){
      printf("%.0f  ", *cube++);
    }
    printf("\n");
  }
  printf("********************************************************\n");
}

void CubeMat(float* weight_cube, const CubeDim weight_cube_dim, float* in_cube, const CubeDim in_cube_dim, 
    float *output, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int CUBE_row, const int CUBE_col){
      int in_cube_index;
      int weight_cube_index;
      int out_start;
      int row_range;
      int copy_range;
      float *CalCube = malloc(CUBE_row * CUBE_col * sizeof(float));
      for(int row=0; row<out_cube_dim.row; row++){
        for(int col=0; col<out_cube_dim.col; col++){
          row_range = (row+1)*CUBE_row>out_2D_dim.row?out_2D_dim.row%CUBE_row:CUBE_row;
          copy_range = (col+1)*CUBE_col>out_2D_dim.col?out_2D_dim.col%CUBE_col:CUBE_col;
          MemSetZeroA(CalCube, CUBE_row*CUBE_col);
          for(int k=0; k<weight_cube_dim.col; k++){
            weight_cube_index = (row*weight_cube_dim.col + k) * CUBE_row*CUBE_col;
            in_cube_index = (k*in_cube_dim.col + col) * CUBE_row*CUBE_col;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                CUBE_row, CUBE_col, CUBE_col, 1.0, weight_cube+weight_cube_index, CUBE_col, in_cube+in_cube_index, CUBE_col, 1.0, CalCube, CUBE_col);
          }
          // show_cubeA(CalCube, CUBE_row, CUBE_col);
          for(int out_row=0; out_row<row_range; out_row++){
            out_start = (row*CUBE_row+out_row)*out_2D_dim.col + col*CUBE_col;
            memcpy(output+out_start, CalCube+out_row*CUBE_col, sizeof(float)*copy_range);
          }
        }
      }
      free(CalCube);
}

void Weight2CubeA(const float *weight, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim, const int CUBE_row, const int CUBE_col){
  
  for(int i=0; i<weight_2D_dim.row; i++){
    int weight_cube_row = i / CUBE_row;
    for(int j=0; j<weight_2D_dim.col; j++){
      int num_cube = j / CUBE_col + weight_cube_row*weight_cube_dim.col;
      weight_cube[num_cube*CUBE_row*CUBE_col+(i%CUBE_row)*CUBE_col+(j%CUBE_col)] = weight[i*weight_2D_dim.col+j];
      // printf("num_cube:%d  weight_cube[%d]=weight[%d]\n",num_cube, num_cube*CUBE_row*CUBE_col+(i%CUBE_row)*CUBE_col+(j%CUBE_col), i*weight_2D_dim.col+j);
    }
  }

}

void Im2MemcopyLayer(const float *input, const TensorDim in_dim, const float *weight, const TensorDim weight_dim, 
    float* bias, float *output, const TensorDim out_dim, 
    const ColDim in_2D_dim, const CubeDim in_cube_dim, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim,
    const int CUBE_row, const int CUBE_col, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int group, const int pad, const int stride, const int bias_en) {

    
    Weight2CubeA(weight, weight_cube, weight_2D_dim, weight_cube_dim, CUBE_row, CUBE_col);

    float* in_2D = malloc(in_2D_dim.col * in_2D_dim.row * sizeof(float));
    float* in_cube = (float*)calloc(in_cube_dim.row * in_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));

    Im2ColCopy(input, in_dim, in_2D, in_2D_dim, in_cube_dim, weight_dim, out_dim, CUBE_row,  
      CUBE_col, pad, pad, stride, stride);
    
    // show_cubeA(in_2D, in_2D_dim.row, in_2D_dim.col);
    // show_cubeA(weight_cube, weight_cube_dim.row*16, weight_cube_dim.col*16);
    
    Col2CubeCopy(in_2D, in_2D_dim, in_cube, in_cube_dim, CUBE_row, CUBE_col);

    // show_cubeA(in_cube, in_cube_dim.row*16*in_cube_dim.col, 16);
    CubeMat(weight_cube, weight_cube_dim, in_cube, in_cube_dim, output, out_2D_dim, out_cube_dim,
      CUBE_row, CUBE_col);


    // for(int i=0; i<out_2D_dim.row; i++){
    //   for(int j=0; j<out_2D_dim.col; j++){
    //     printf("%.0f   ", output[i*out_2D_dim.col+j]);
    //   }
    //   printf("\n");
    // }
    // printf("\n========================================\n");

    free(in_2D);
    free(in_cube);
}
