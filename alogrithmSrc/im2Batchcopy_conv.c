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

// nhwc -> 2D  support: stride pad
void Im2ColCopy_nhwc(const float *input, const TensorDim in_dim, float* in_2D, const ColDim in_2D_dim, 
      const CubeDim in_cube_dim, const TensorDim weight_dim, const TensorDim out_dim, const int CUBE_row,  
      const int CUBE_col, const int pad_h, const int pad_w, const int stride_h, const int stride_w){

  int in_start = 0;
  int in_2D_start = 0;
  int start_row;
  int start_col;
  int zero_range;
  int num_range;
  int copy_range = weight_dim.w*weight_dim.c;
  float* zero_list = (float*)calloc(copy_range, sizeof(float));
  
  // int index = 0;

  for(int output_rows=0; output_rows<out_dim.h; output_rows++){
    for(int output_cols=0; output_cols<out_dim.w; output_cols++){
      start_col = output_cols*stride_w-pad_w;
      for(int kernel_row=0; kernel_row<weight_dim.h; kernel_row++){
        start_row = -pad_h + output_rows*stride_h + kernel_row;
        if (!Is_in_rangeA(start_row, in_dim.h)){
            memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*copy_range);
            // printf("index : %d  in_2D_start:%d   copy_zero:%d  \n", index++, in_2D_start, copy_range);
            in_2D_start += copy_range; 
            
        }else{
            if(start_col<0){
                zero_range = (-start_col)*weight_dim.c;
                memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*zero_range);
                // printf("index : %d  in_2D_start:%d   copy_zero:%d  ", index++, in_2D_start, zero_range);
                in_2D_start += zero_range;
                in_start = start_row*in_dim.w*in_dim.c;
                if(start_col+weight_dim.w>in_dim.w){
                    int overflow = start_col+weight_dim.w-in_dim.w;
                    num_range = (weight_dim.w+start_col-overflow)*weight_dim.c;
                    memcpy(in_2D+in_2D_start, input+in_start, sizeof(float)*num_range);
                    in_2D_start += num_range;
                    memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*overflow*weight_dim.c);
                    in_2D_start += overflow*weight_dim.c;
                    // printf(" copy_data:%d   copy_zero \n", overflow*weight_dim.c);
                }else{
                    num_range = (weight_dim.w+start_col)*weight_dim.c;
                    memcpy(in_2D+in_2D_start, input+in_start, sizeof(float)*num_range);
                    in_2D_start += num_range;
                    // printf(" copy_data:%d \n", num_range);

                }
            }else if(start_col+weight_dim.w>in_dim.w){
                int overflow = start_col+weight_dim.w-in_dim.w;
                num_range = (weight_dim.w-overflow)*weight_dim.c;
                in_start = start_row*in_dim.w*in_dim.c + start_col*in_dim.c;
                memcpy(in_2D+in_2D_start, input+in_start, sizeof(float)*num_range);
                // printf("index : %d  in_2D_start:%d   copy_data:%d  ", index++, in_2D_start, num_range);
                in_2D_start += num_range;
                memcpy(in_2D+in_2D_start, zero_list, sizeof(float)*overflow*weight_dim.c);
                // printf(" copy_zero:%d \n", overflow*weight_dim.c);
                in_2D_start += overflow*weight_dim.c;
            }else{
                in_start = start_row*in_dim.w*in_dim.c + start_col*in_dim.c;
                memcpy(in_2D+in_2D_start, input+in_start, sizeof(float)*copy_range);
                // printf("index : %d  in_2D_start:%d   copy_data:%d  \n", index++, in_2D_start, copy_range);
                in_2D_start += copy_range;
            }
        }
      }
    }
  }
  free(zero_list);
}

void Col2CubeCopyC(float* in_2D, const ColDim in_2D_dim, float* in_cube, const CubeDim in_cube_dim, const int CUBE_row, const int CUBE_col){
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

void MemSetZeroC(float* CalCubeLast, int elemNum){
  for(int i=0; i<elemNum; i++){
    CalCubeLast[i] = 0;
  }
}

void show_cubeC(float* cube, int CUBE_row, int CUBE_col){
  for(int i=0; i<CUBE_row; i++){
    for(int j=0; j<CUBE_col; j++){
      printf("%.0f  ", *cube++);
    }
    printf("\n");
  }
  printf("********************************************************\n");
}

void CubeMatC(float* weight_cube, const CubeDim weight_cube_dim, float* in_cube, const CubeDim in_cube_dim, 
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
          MemSetZeroC(CalCube, CUBE_row*CUBE_col);
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

// weight:[out_c,in_c,k_h,k_w]  ->   2D: [k_h*k_w*in_c, out_c]
void Weight2CubeC(const float *weight, float* weight_cube, const TensorDim weight_dim, const ColDim weight_2D_dim, const CubeDim weight_cube_dim, const int CUBE_row, const int CUBE_col){
  int we_addr;
  int c, h, w;
  for(int i=0; i<weight_2D_dim.row; i++){
    int weight_cube_row = i / CUBE_row;
    c = i % weight_dim.c;
    h = i/(weight_dim.w*weight_dim.c);
    w = (i-h*weight_dim.w*weight_dim.c)/weight_dim.c;

    for(int j=0; j<weight_2D_dim.col; j++){
      int num_cube = j / CUBE_col + weight_cube_row*weight_cube_dim.col;
      we_addr = j*weight_2D_dim.row + c*weight_dim.h*weight_dim.w + h*weight_dim.w + w;
      weight_cube[num_cube*CUBE_row*CUBE_col+(i%CUBE_row)*CUBE_col+(j%CUBE_col)] = weight[we_addr];
      // printf("index:%d  weight_cube[%d]=weight[%d]\n",index++, num_cube*CUBE_row*CUBE_col+(i%CUBE_row)*CUBE_col+(j%CUBE_col), we_addr);
    }
  }

}

/*
    input * weight
*/
void Im2BatchcopyLayer(const float *input, const TensorDim in_dim, const float *weight, const TensorDim weight_dim, 
    float* bias, float *output, const TensorDim out_dim, 
    const ColDim in_2D_dim, const CubeDim in_cube_dim, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim,
    const int CUBE_row, const int CUBE_col, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int group, const int pad, const int stride, const int bias_en) {
      printf("   get in \n");

    
    // work in convert stage   nchw->Cube
    Weight2CubeC(weight, weight_cube, weight_dim, weight_2D_dim, weight_cube_dim, CUBE_row, CUBE_col);

    float* in_2D = malloc(in_2D_dim.col * in_2D_dim.row * sizeof(float));
    float* in_cube = (float*)calloc(in_cube_dim.row * in_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));

    // nhwc->2D
    Im2ColCopy_nhwc(input, in_dim, in_2D, in_2D_dim, in_cube_dim, weight_dim, out_dim, CUBE_row,  
      CUBE_col, pad, pad, stride, stride);
    
    // show_cubeB(in_2D, in_2D_dim.row, in_2D_dim.col);
    // show_cubeB(weight_cube, weight_cube_dim.row*16, weight_cube_dim.col*16);
    
    Col2CubeCopyC(in_2D, in_2D_dim, in_cube, in_cube_dim, CUBE_row, CUBE_col);

    // show_cubeA(in_cube, in_cube_dim.row*16*in_cube_dim.col, 16);
    // CubeMatC(weight_cube, weight_cube_dim, in_cube, in_cube_dim, output, out_2D_dim, out_cube_dim,
    //   CUBE_row, CUBE_col);
    CubeMatC(in_cube, in_cube_dim, weight_cube, weight_cube_dim, output, out_2D_dim, out_cube_dim,
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
