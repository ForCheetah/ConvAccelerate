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


// 思路三: 结论公式2与固定已知系数的加法
float GF22[12] = {1, 0, 0,
               0.5, 0.5, 0.5,
               0.5, -0.5, 0.5,
               0, 0, 1};
float GT[12] = {1, 0.5, 0.5, 0,
                0, 0.5, -0.5, 0,
                0, 0.5, 0.5, 1};
//矩阵乘
void dot(float* A, int row_A, int col_A, float* B, int row_B, int col_B, float* C) {
    // assert(col_A == row_B);              // && row_A == col_B
                                         //由矩阵相乘，要求f2=s1，以下用f2
    for (int i = 0; i < row_A; i++)      // i表示第i行
        for (int j = 0; j < col_B; j++)  // j表示第j列
            //  C[i*col_A + j] = 0;        //在这里 result[i][j] = result[i*f2+j];
            for (int p = 0; p < col_A; p++)
                C[i * col_B + j] += A[i * col_A + p] * B[p * col_B + j];
}

void transforme_g(float* g, float* transformed_g, int c, int n) {
    float Gg[12] = {0};
    for (int nn = 0; nn < n; nn++) {
        for (int cc = 0; cc < c; cc++)  //卷积核通道循环
        {
            memset(Gg, 0, 12 * sizeof(float));                //清空
            dot(GF22, 4, 3, nn * c * 9 + cc * 9 + g, 3, 3, Gg);  //先计算得到U=GgGT
            dot(Gg, 4, 3, GT, 3, 4, nn * c * 16 + cc * 16 + transformed_g);
        }
    }
}
//对应点相乘
void multi(float* A, int row_A, int col_A, float* B, int row_B, int col_B, float* C) {
    for (int i = 0; i < row_A; i++)
        for (int j = 0; j < col_A; j++)
            C[col_A * i + j] = A[col_A * i + j] * B[col_A * i + j];
}
/*
tile : 每个tile的尺寸  4
height_col: 纵坐标上有多少个tile
width_col: 横坐标上有多少个tile

    im2col_winograd1(d_5, c / groups, h, w, size, stride, 2, 3, transformed_d_5);   // image分块
*/
void im2col_winograd_F22(const float* data_im, const TensorDim weight_dim, const TensorDim in_dim, int stride, int tile, int height_col, int width_col,
                    int pad, float* data_col) {
    int step = tile - (weight_dim.h - stride);     //相邻tile之间首元素的距离  2

    // 先乘好下面这些变量省的for循环里再重复计算   相当于Imge2Cube 的代码 
    int tile2 = tile * tile;
    int width_col_tile2 = width_col * tile2;
    int height_width_col_tile2 = height_col * width_col * tile2;
    int height_width = in_dim.h * in_dim.w;
    for (int c = 0; c < in_dim.c; c++)              //遍历通道  
        for (int h = 0; h < height_col; h++)        //输入图像里按列遍历每个tile
            for (int w = 0; w < width_col; w++)     //输入图像里按行遍历每个tile
                for (int i = 0; i < tile; i++)      //按列遍历每个tile里的行
                    for (int j = 0; j < tile; j++){ //按行遍历每个tile的元素
                        if(h*step+i-pad<0 || h*step+i-pad>=in_dim.h || w*step+j-pad<0 || w*step+j-pad>=in_dim.w){
                            continue;
                        }
                        data_col[c * height_width_col_tile2 + h * width_col_tile2 + w * tile2 + i * tile + j] =
                                data_im[c * height_width + (h*step+i-pad)*in_dim.w + w*step+j-pad];
                        // printf("data_temp[%d] = data[%d]\n", c*height_width_col_tile2+h*width_col_tile2+w*tile2+i*tile+j, c * height_width + (h*step+i-pad)*in_dim.w + w*step+j-pad);
                    }  
}

/*
At = [[1,1,1,0],
      [0,1,-1,1]]
G = [[1,0,0],
     [1/2,1/2,1/2],
     [1/2,-1/2,1/2],
     [0,0,1]
     ]
bt = [[1,0,-1,0]
     [0,1,1,0],
     [0,-1,1,0],
     [0,1,0,-1]
     ]
*/

void winograd_2d(float* U, float* d, float* result) {
    float BTd[16] = {0};
    float V[16] = {0};
    float UV[16] = {0};
    float ATUV[8] = {0};

    // dot(BT, 4, 4, d, 4, 4, BTd);
    for (int i = 0; i < 4; i++)
        BTd[i] = d[0 + i] - d[8 + i];
    for (int i = 0; i < 4; i++)
        BTd[4 + i] = d[4 + i] + d[8 + i];
    for (int i = 0; i < 4; i++)
        BTd[8 + i] = -d[4 + i] + d[8 + i];
    for (int i = 0; i < 4; i++)
        BTd[12 + i] = d[4 + i] - d[12 + i];

    // dot(BTd, 4, 4, B, 4, 4, V);
    for (int i = 0; i < 4; i++)
        V[0 + i * 4] = BTd[0 + i * 4] - BTd[2 + i * 4];
    for (int i = 0; i < 4; i++)
        V[1 + i * 4] = BTd[1 + i * 4] + BTd[2 + i * 4];
    for (int i = 0; i < 4; i++)
        V[2 + i * 4] = -BTd[1 + i * 4] + BTd[2 + i * 4];
    for (int i = 0; i < 4; i++)
        V[3 + i * 4] = BTd[1 + i * 4] - BTd[3 + i * 4];

    multi(U, 4, 4, V, 4, 4, UV);

    // dot(AT, 2, 4, UV, 4, 4, ATUV);
    for (int i = 0; i < 4; i++)
        ATUV[i] = UV[0 + i] + UV[4 + i] + UV[8 + i];
    for (int i = 0; i < 4; i++)
        ATUV[4 + i] = UV[4 + i] - UV[8 + i] - UV[12 + i];

    result[0] += (ATUV[0] + ATUV[1] + ATUV[2]);
    result[2] += (ATUV[4] + ATUV[5] + ATUV[6]);
    result[1] += (ATUV[1] - ATUV[2] - ATUV[3]);
    result[3] += (ATUV[5] - ATUV[6] - ATUV[7]);
}

void convolutional_winograd(float* transformed_g, float* transformed_d, float* transformed_output, const TensorDim in_dim, const TensorDim weight_dim, int height_col, int width_col) {

    int width_col_16 = width_col * 16;  // 19*16
    int height_col_width_col_16 = height_col * width_col_16;  // 19*19*16
    int width_col_4 = width_col * 4;  // 19*4
    int height_col_width_col_4 = height_col * width_col_4;  // 19*19
    // int channels_height_col_width_col_4 = channels * height_col_width_col_4;
    int temp_U_c, temp_U_h;
    int temp_d_nn,  temp_d_h;

    for (int nn = 0; nn < weight_dim.n; nn++)  //卷积核个数循环 64
    {
        temp_d_nn = nn * height_col_width_col_4;
        for (int c = 0; c < in_dim.c; c++)  //卷积核通道循环 64
        {
            temp_U_c = c * height_col_width_col_16;  // step_channel
            // temp_d_c = c * height_col_width_col_4;
            for (int h = 0; h < height_col; h++) {    //    tile宽  19
                temp_U_h = h * width_col_16; // h*19*16
                temp_d_h = h * width_col_4;  // w*19*4
                for (int w = 0; w < width_col; w++){   //   tile高  19
                    winograd_2d(nn * in_dim.c * 16 + c * 16 + transformed_g, temp_U_c + temp_U_h + w * 16 + transformed_d,
                                 temp_d_nn + temp_d_h + w * 4 + transformed_output);  // temp_U_nn ++ temp_d_c    对 g 和 d 的4*4进行计算
                    // printf("g: \n", nn);
                }   
                    
            }
        }
    }
}


// 相当于以前写的 2D->cube的逆序版，cube->2D
void col2im_winograd(float* transformed_output, int height_col, int width_col, const TensorDim weight_dim, const TensorDim in_dim, int stride, int pad, float* output) {

    int height_map = (in_dim.h + 2 * pad - weight_dim.h) / stride + 1;
    int width_map = (in_dim.w + 2 * pad - weight_dim.w) / stride + 1;
    for(int outc=0; outc<weight_dim.n; outc++){
        for(int row=0; row<height_map; row++){
            for(int col=0; col<width_map; col++){
                output[outc*height_map*width_map+row*width_map+col] = transformed_output[outc*height_col*width_col*4+(row/2*width_col+col/2)*4+row%2*2+col%2];
            // printf("out[%d] = temp[%d] \n", outc*height_map*width_map+row*width_map+col, (row/2*width_col+col/2)*4+col%2);
            }
        }
    }    
}

void MemSetZeroB(float* CalCubeLast, int elemNum){
  for(int i=0; i<elemNum; i++){
    CalCubeLast[i] = 0;
  }
}

void show_cubeB(float* cube, int CUBE_row, int CUBE_col){
  for(int i=0; i<CUBE_row; i++){
    for(int j=0; j<CUBE_col; j++){
      printf("%.0f  ", *cube++);
    }
    printf("\n");
  }
  printf("********************************************************\n");
}


void WinogradF22Layer(const float* in_data, const TensorDim in_dim, float* weight, const TensorDim weight_dim,
      float* output, const TensorDim out_dim, int stride, int pad) {

    int tile_h = (in_dim.h%2)==0?(in_dim.h+pad+pad-2)/2 : (in_dim.h+pad+pad-2) / 2+1;
    int tile_w = (in_dim.w%2)==0?(in_dim.w+pad+pad-2) / 2 : (in_dim.w+pad+pad-2) / 2+1;
    // printf("h:%d   w:%d\n", tile_h, tile_w);
    float* transformed_d = (float*)calloc(tile_h * tile_w * in_dim.c * 16, sizeof(float));
    float* transformed_g = calloc(weight_dim.n * weight_dim.c * 16, sizeof(float));
    float* transformed_output = calloc(tile_h*2 * tile_w*2 * weight_dim.n, sizeof(float));
    transforme_g(weight, transformed_g, weight_dim.c, weight_dim.n);
    im2col_winograd_F22(in_data, weight_dim, in_dim, stride, 4, tile_h, tile_w, pad, transformed_d) ;
    convolutional_winograd(transformed_g, transformed_d, transformed_output, in_dim, weight_dim, tile_h, tile_w);
    col2im_winograd(transformed_output, tile_h, tile_w, weight_dim, in_dim, stride, pad, output);


    free(transformed_g);
    free(transformed_d);
    free(transformed_output);
}
