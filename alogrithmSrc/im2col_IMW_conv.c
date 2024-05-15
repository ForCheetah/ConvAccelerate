
#include <stdbool.h>
#include <cblas.h>
#include "common_types.h"
#include "stdio.h"


/*
    这里实现的是  按照 input_tensor_col Matmul weight_col 顺序进行的矩阵相乘
*/

static inline bool If_in_range(int a, int b) {  
    return (unsigned int)a < (unsigned int)(b);
}

void Im2Col_IMW(const float *data_im, const int channels,const int height, const int width, const int kernel_h,
        const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, float *data_col) {
  const int output_h = (height + 2 * pad_h - kernel_h ) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  const int channel_size = height * width; 
    for (int output_rows = 0; output_rows<output_h; output_rows++) {  //纵移
        for (int output_col = 0; output_col<output_w; output_col++){  //横移
        for (int channel = 0; channel<channels; channel++){ //channel
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {  // 位置行
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++){  // 位置列
                int input_row = -pad_h + kernel_row + output_rows*stride_h;
                int input_col = -pad_w + kernel_col + output_col*stride_w;
                if(!If_in_range(input_row, height) || !If_in_range(input_col, width)){
                    *(data_col++) = 0;
                    // index++;
                }else{
                    *(data_col++) = data_im[channel*channel_size + input_row*width + input_col];
                    // printf("index : %d  dim: %d \n", index, channel*channel_size + input_row*width + input_col);
                    // index++;
                }
                }
            }
        }
        }
    }
}

void Weight2Col_IMW(const float* weight, float* weight_col, const int row, const int col){
    for(int r=0; r<row; r++){
        for(int c=0; c<col; c++){
        weight_col[c*row+r] = weight[r*col+c];
        }
    }
}

void show_weight1(const float* weight, int row, int col){
    for(int i=0; i<row; i++){
        for(int j=0; j<col;j++){
        printf("%.0f   ", weight[i*col+j]);
        }
        printf("\n");
    }
}

void show_input(const float* weight, int C, int H, int W){
    for(int i=0; i<C; i++){
        for(int j=0; j<H;j++){
        for(int k=0; k<W; k++){
            printf("%.0f   ", weight[i*H*W+j*W+k]);
        }
        printf("\n");
        }
        printf("\n");
    }
}



void Im2ColConvIMW(const float *input, const float *weight, 
    const float *bias, float *in_col, float *weight_col, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {

    int C = in_dim.c;
    // int N = in_dim.n;
    // printf("chanell::::::::::%d\n", C);
    int H = in_dim.h;
    int W = in_dim.w;
    float alpha = 1;
    float beta = 0;
    Weight2Col_IMW(weight, weight_col, out_dim.c, C*ker_size*ker_size);  // 没有原地算法
    for (int b = 0; b < in_dim.n; b++) {  // 对batch的支持
        int in_offset = b * C * H * W;
        Im2Col_IMW(input + in_offset, C, H, W, ker_size, ker_size, pad, pad, stride,
            stride, in_col);
        // show_weight1(in_col, 25, C*ker_size*ker_size);
    
        int out_offset = b * out_dim.c *  out_dim.h * out_dim.w; 
        
        int m = out_dim.h * out_dim.w;
        int k = C * ker_size * ker_size;
        int n = out_dim.c;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, in_col,
                    k, weight_col, n, beta, output + out_offset, n);
    }
}
