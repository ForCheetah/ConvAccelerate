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

/*

    G 矩阵
    const float ktm[8][3] = {       // Acutually Matrix G
        {   1.0f,     0.0f,     0.0f},    // G矩阵
        {-2.0f/9,  -2.0f/9,  -2.0f/9},
        {-2.0f/9,   2.0f/9,  -2.0f/9},
        {1.0f/90,  1.0f/45,  2.0f/45},
        {1.0f/90, -1.0f/45,  2.0f/45},
        {1.0f/45,  1.0f/90, 1.0f/180},
        {1.0f/45, -1.0f/90, 1.0f/180},
        {   0.0f,     0.0f,     1.0f}
    }
*/

//  GF63: [8,3]   GT: [3,8]
float GF63[24] = {1.0,    0.0,    0.0,     -2.0f/9,  -2.0f/9,   -2.0f/9,    -2.0f/9,   2.0f/9,  -2.0f/9,
                1.0f/90,  1.0f/45,   2.0f/45,   1.0f/90,   -1.0f/45,   2.0f/45,  1.0f/45,  1.0f/90, 1.0f/180,
                1.0f/45,  -1.0f/90,  1.0f/180,     0.0f,     0.0f,     1.0f};
float GT63[24] = {1.0f, -2.0f/9, -2.0f/9, 1.0f/90, 1.0f/90, 1.0f/45, 1.0f/45,  0.0f,
                0.0f,   -2.0f/9,  2.0f/9, 1.0f/45, -1.0f/45, 1.0f/90, -1.0f/90, 0.0f,
                0.0f,  -2.0f/9, -2.0f/9, 2.0f/45, 2.0f/45, 1.0f/180, 1.0f/180, 1.0f};
//矩阵乘
void dot_mat(float* A, int row_A, int col_A, const float* B, int row_B, int col_B, float* C) {
    // assert(col_A == row_B);              // && row_A == col_B
                                         //由矩阵相乘，要求f2=s1，以下用f2
    for (int i = 0; i < row_A; i++)      // i表示第i行
        for (int j = 0; j < col_B; j++)  // j表示第j列
            //  C[i*col_A + j] = 0;        //在这里 result[i][j] = result[i*f2+j];
            for (int p = 0; p < col_A; p++)
                C[i * col_B + j] += A[i * col_A + p] * B[p * col_B + j];
}

void transforme_g_F63(const float* g, float* transformed_g, int c, int n) {
    float Gg[24] = {0};
    for (int nn = 0; nn < n; nn++) {
        for (int cc = 0; cc < c; cc++) { //卷积核通道循环
            memset(Gg, 0, 24 * sizeof(float));                //清空
            dot_mat(GF63, 8, 3, nn * c * 9 + cc * 9 + g, 3, 3, Gg);  //先计算得到U=GgGT
            dot_mat(Gg, 8, 3, GT63, 3, 8, nn * c * 64 + cc * 64 + transformed_g);
        }
    }
}
//对应点相乘
void dot_multi(float* A, int row_A, int col_A, float* B, int row_B, int col_B, float* C) {
    for (int i = 0; i < row_A; i++)
        for (int j = 0; j < col_A; j++)
            C[col_A * i + j] = A[col_A * i + j] * B[col_A * i + j];
}

/*
tile : 每个tile的尺寸  8
height_col: 纵坐标上有多少个tile
width_col: 横坐标上有多少个tile

    im2col_winograd1(d_5, c / groups, h, w, size, stride, 2, 3, transformed_d_5);   // image分块
*/
void im2col_winograd_F63(const float* data_im, const TensorDim weight_dim, const TensorDim in_dim, int stride, int tile, int height_col, int width_col,
                    int pad, float* data_col) {
    int step = tile - (weight_dim.h - stride);     //相邻tile之间首元素的距离  6

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
    At矩阵
    const float otm[6][8] = {
        {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    };

    G 矩阵
    const float ktm[8][3] = {       // Acutually Matrix G
        {   1.0f,     0.0f,     0.0f},    // G矩阵
        {-2.0f/9,  -2.0f/9,  -2.0f/9},
        {-2.0f/9,   2.0f/9,  -2.0f/9},
        {1.0f/90,  1.0f/45,  2.0f/45},
        {1.0f/90, -1.0f/45,  2.0f/45},
        {1.0f/45,  1.0f/90, 1.0f/180},
        {1.0f/45, -1.0f/90, 1.0f/180},
        {   0.0f,     0.0f,     1.0f}
    }

    Bt变态矩阵
    const float itm[8][8] = {
            0      1     2         3      4       5      6     7
    0    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
    1    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
    2    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
    3    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
    4    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
    5    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
    6    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
    7    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
    };
*/

void winograd_2d_F63(float* U, float* d, float* result) {
    float BTd[64] = {0};
    float V[64] = {0};
    float UV[64] = {0};
    float ATUV[48] = {0};

    // dot(BT, 8, 8, d, 8, 8, BTd);
    for (int i = 0; i < 8; i++)
        BTd[i] = d[0+i] + 5.25*(d[32+i]-d[16+i]) - d[48+i];
    for (int i = 0; i < 8; i++)
        BTd[8+i] = d[8+i] + d[16+i] - 4.25*(d[24+i]+d[32+i]) + d[40+i] + d[48+i];
    for (int i = 0; i < 8; i++)
        BTd[16+i] = -d[8+i] + d[16+i] + 4.25*(d[24+i]-d[32+i]) - d[40+i] + d[48+i];
    for (int i = 0; i < 8; i++)
        BTd[24+i] = 0.5*d[8+i] + 0.25*d[16+i] - 2.5*d[24+i] - 1.25*d[32+i] + 2*d[40+i] + d[48+i];

    for (int i = 0; i < 8; i++)
        BTd[32+i] = -0.5*d[8+i] + 0.25*d[16+i] + 2.5*d[24+i] - 1.25*d[32+i] - 2*d[40+i] + d[48+i];
    for (int i = 0; i < 8; i++)
        BTd[40+i] = 2*d[8+i] + 4*d[16+i] - 2.5*d[24+i] - 5*d[32+i] + 0.5*d[40+i] + d[48+i];
    for (int i = 0; i < 8; i++)
        BTd[48+i] = -2*d[8+i] + 4*d[16+i] + 2.5*d[24+i] - 5*d[32+i] - 0.5*d[40+i] + d[48+i];
    for (int i = 0; i < 8; i++)
        BTd[56+i] = -1*d[8+i] + 5.25*(d[24+i]-d[40+i]) + d[56+i];

/*
    Bt变态矩阵
    const float itm[8][8] = {
            0      1     2         3      4       5      6     7
    0    {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
    1    {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
    2    {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
    3    {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
    4    {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
    5    {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
    6    {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
    7    {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
    };
*/
    // dot(BTd, 8, 8, B, 8, 8, V);
    for (int i = 0; i < 8; i++)
        V[i*8] = BTd[i*8] + 5.25*(BTd[4+i*8]-BTd[2+i*8]) - BTd[6+i*8];
    for (int i = 0; i < 8; i++)
        V[1+i*8] = BTd[1+i*8] + BTd[2+i*8] - 4.25*(BTd[3+i*8]+BTd[4+i*8]) + BTd[5+i*8] + BTd[6+i*8];
    for (int i = 0; i < 8; i++)
        V[2+i*8] = -BTd[1+i*8] + BTd[2+i*8] + 4.25*(BTd[3+i*8]-BTd[4+i*8]) - BTd[5+i*8] + BTd[6+i*8];
    for (int i = 0; i < 8; i++)
        V[3+i*8] = 0.5*BTd[1+i*8] + 0.25*BTd[2+i*8] - 2.5*BTd[3+i*8] - 1.25*BTd[4+i*8] + 2*BTd[5+i*8] + BTd[6+i*8];

    for (int i = 0; i < 8; i++)
        V[4+i*8] = -0.5*BTd[1+i*8] + 0.25*BTd[2+i*8] + 2.5*BTd[3+i*8] - 1.25*BTd[4+i*8] - 2*BTd[5+i*8] + BTd[6+i*8];
    for (int i = 0; i < 8; i++)
        V[5+i*8] = 2*BTd[1+i*8] + 4*BTd[2+i*8] - 2.5*BTd[3+i*8] - 5*BTd[4+i*8] + 0.5*BTd[5+i*8] + BTd[6+i*8];
    for (int i = 0; i < 8; i++)
        V[6+i*8] = -2*BTd[1+i*8] + 4*BTd[2+i*8] + 2.5*BTd[3+i*8] - 5*BTd[4+i*8] - 0.5*BTd[5+i*8] + BTd[6+i*8];
    for (int i = 0; i < 8; i++)
        V[7+i*8] = -BTd[1+i*8] + 5.25*(BTd[3+i*8]-BTd[5+i*8]) + BTd[7+i*8];
    
    dot_multi(U, 8, 8, V, 8, 8, UV);


/*
    const float otm[6][8] = {
        {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    };
*/
    // dot(AT, 6, 8, UV, 8, 8, ATUV);
    for (int i = 0; i < 8; i++)
        ATUV[i] = UV[i] + UV[8+i] + UV[16+i] + UV[24+i] + UV[32+i] + 32*(UV[40+i]+UV[48+i]);
    for (int i = 0; i < 8; i++)
        ATUV[8+i] = UV[8+i] - UV[16+i] + 2*(UV[24+i]-UV[32+i]) + 16*(UV[40+i]-UV[48+i]);

    for (int i = 0; i < 8; i++)
        ATUV[16+i] = UV[8+i] + UV[16+i] + 4*(UV[24+i]+UV[32+i]) + 8*(UV[40+i]+UV[48+i]);
    for (int i = 0; i < 8; i++)
        ATUV[24+i] = UV[8+i] - UV[16+i] + 8*(UV[24+i]-UV[32+i]) + 4*(UV[40+i]-UV[48+i]);

    for (int i = 0; i < 8; i++)
        ATUV[32+i] = UV[8+i] + UV[16+i] + 16*(UV[24+i]+UV[32+i]) + 2*(UV[40+i]+UV[48+i]);
    for (int i = 0; i < 8; i++)
        ATUV[40+i] = UV[8+i] - UV[16+i] + 32*(UV[24+i]-UV[32+i]) + UV[40+i] - UV[48+i] + UV[56+i];

/*
    const float otm[6][8] = {
        {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    };
*/

    for(int i=0; i<6; i++)
        result[i*6] += ATUV[i*8] + ATUV[1+i*8] + ATUV[2+i*8] + ATUV[3+i*8] + ATUV[4+i*8] + 32*(ATUV[5+i*8]+ATUV[6+i*8]);
    for(int i=0; i<6; i++)
        result[1+i*6] += ATUV[1+i*8] - ATUV[2+i*8] + 2*(ATUV[3+i*8]-ATUV[4+i*8]) + 16*(ATUV[5+i*8]-ATUV[6+i*8]);
    for(int i=0; i<6; i++)
        result[2+i*6] += ATUV[1+i*8] + ATUV[2+i*8] + 4*(ATUV[3+i*8]+ATUV[4+i*8]) + 8*(ATUV[5+i*8]+ATUV[6+i*8]);
    for(int i=0; i<6; i++)
        result[3+i*6] += ATUV[1+i*8] - ATUV[2+i*8] + 8*(ATUV[3+i*8]-ATUV[4+i*8]) + 4*(ATUV[5+i*8]-ATUV[6+i*8]);
    for(int i=0; i<6; i++)
        result[4+i*6] += ATUV[1+i*8] + ATUV[2+i*8] + 16*(ATUV[3+i*8]+ATUV[4+i*8]) + 2*(ATUV[5+i*8]+ATUV[6+i*8]);
    for(int i=0; i<6; i++)
        result[5+i*6] += ATUV[1+i*8] - ATUV[2+i*8] + 32*(ATUV[3+i*8]-ATUV[4+i*8]) + ATUV[5+i*8] - ATUV[6+i*8] + ATUV[7+i*8];

    // result[0] += (ATUV[0] + ATUV[1] + ATUV[2]);
    // result[2] += (ATUV[4] + ATUV[5] + ATUV[6]);
    // result[1] += (ATUV[1] - ATUV[2] - ATUV[3]);
    // result[3] += (ATUV[5] - ATUV[6] - ATUV[7]);
}

void convolutional_winograd_F63(float* transformed_g, float* transformed_d, float* transformed_output, const TensorDim in_dim, const TensorDim weight_dim, int height_col, int width_col) {

    int width_col_16 = width_col * 64;  // 19*64
    int height_col_width_col_16 = height_col * width_col_16;  // 19*19*64
    int width_col_4 = width_col * 36;  // 19*8
    int height_col_width_col_4 = height_col * width_col_4;  // 19*19*36
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
                temp_U_h = h * width_col_16; // h*19*64
                temp_d_h = h * width_col_4;  // w*19*36
                for (int w = 0; w < width_col; w++)    //   tile高  19
                    winograd_2d_F63(nn * in_dim.c * 64 + c * 64 + transformed_g, temp_U_c + temp_U_h + w * 64 + transformed_d,
                                 temp_d_nn + temp_d_h + w * 36 + transformed_output);  // temp_U_nn ++ temp_d_c    对 g 和 d 的8*8进行计算
            }
        }
    }
}


// 相当于以前写的 2D->cube的逆序版，cube->2D
void col2im_winograd_F63(float* transformed_output, int height_col, int width_col, const TensorDim weight_dim, const TensorDim in_dim, int stride, int pad, float* output) {

    int height_map = (in_dim.h + 2 * pad - weight_dim.h) / stride + 1;
    int width_map = (in_dim.w + 2 * pad - weight_dim.w) / stride + 1;
    for(int outc=0; outc<weight_dim.n; outc++){
        for(int row=0; row<height_map; row++){
            for(int col=0; col<width_map; col++){
                output[outc*height_map*width_map+row*width_map+col] = transformed_output[outc*height_col*width_col*36+(row/6*width_col+col/6)*36+row%6*6+col%6];
            // printf("out[%d] = temp[%d] \n", outc*height_map*width_map+row*width_map+col, (row/2*width_col+col/2)*4+col%2);
            }
        }
    }    
}


void show_cubeD(float* cube, int CUBE_row, int CUBE_col){
    for(int i=0; i<CUBE_row; i++){
        for(int j=0; j<CUBE_col; j++){
        printf("%.0f  ", *cube++);
        }
        printf("\n");
    }
    printf("********************************************************\n");
}


/*
f(6*6,3*3)实现二维卷积     8*8  3*3  ->  6*6
*/
void WinogradF63Layer(const float* in_data, const TensorDim in_dim, const float* weight, const TensorDim weight_dim,
        float* output, const TensorDim out_dim, int stride, int pad) {

    //pad to 6n+2
    int tile_h = ((in_dim.h+pad+pad-2)%6)==0?(in_dim.h+pad+pad-2)/6 : (in_dim.h+pad+pad-2)/6+1;
    int tile_w = ((in_dim.w+pad+pad-2)%6)==0?(in_dim.w+pad+pad-2)/6 : (in_dim.w+pad+pad-2)/6+1;
    // printf("h:%d   w:%d\n", tile_h, tile_w);
    float* transformed_d = (float*)calloc(tile_h * tile_w * in_dim.c * 64, sizeof(float));
    float* transformed_g = calloc(weight_dim.n * weight_dim.c * 64, sizeof(float));
    float* transformed_output = calloc(tile_h * tile_w * 36 * weight_dim.n, sizeof(float));

    // calculate at convert stage  
    transforme_g_F63(weight, transformed_g, weight_dim.c, weight_dim.n);

    im2col_winograd_F63(in_data, weight_dim, in_dim, stride, 8, tile_h, tile_w, pad, transformed_d) ;
    convolutional_winograd_F63(transformed_g, transformed_d, transformed_output, in_dim, weight_dim, tile_h, tile_w);
    // show_cubeD(transformed_output, tile_h * tile_w * weight_dim.n, 36);
    col2im_winograd_F63(transformed_output, tile_h, tile_w, weight_dim, in_dim, stride, pad, output);


    free(transformed_g);
    free(transformed_d);
    free(transformed_output);
}
