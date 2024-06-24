// @file main.c
//   running ： export LD_LIBRARY_PATH=/home/byd/whs/CPP/libblas/lib
//              make  ./test_app
//

#include <conv_layers.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "common_types.h"
#include "data_reshape.h"
#include "utils.h"






/*
  这里是 weight_col * input_col
*/

void TestIm2FlavorConvLayer() {
    // Configurations
    // Enable kn2row or kn2col
    // bool im2col = true;
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;

    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 1;
    int C = 3;
    int H = 5;
    int W = 5;
    int M = 9;
    int pad = 2;


    TensorDim in_dim = {N, C, H, W};
    TensorDim filt_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
    float *spad = malloc(out_dim.h * out_dim.w * in_dim.c *   
                        filt_dim.w * filt_dim.h * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);
    
    if(print_info){
        printf(" # input tensor: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor: [%d, %d, %d, %d] \n", out_dim.n, out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d \n", group, stride, pad);
        printf(" # spad: [%d, %d, %d, %d, %d] \n", out_dim.h, out_dim.w, in_dim.c, filt_dim.w, filt_dim.h);
    }

    Im2ColConvLayer(in_data, filters, bias, spad, in_dim, out_dim, ker_size,  // image to col 算法
                    group, pad, stride, bias_en, output);


    if (TensorCompare(output, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
    free(spad);
}



/*

  这里是  input_col * weight_col
*/
void show_some(float* data){
    for (int i=0; i<25; i++){
        printf("#%.0f\t", data[i]);
    }
}

void Transpose2D(const float* data, float* data_tranp, const int row, const int col){
    for(int r=0; r<row; r++){
        for(int c=0; c<col; c++){
        data_tranp[c*row+r] = data[r*col+c];
        }
    }
    }

    void TestIm2ColConvIMW() {
    // Configurations
    // Enable kn2row or kn2col
    // bool im2col = true;
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool padding_en = true;
    bool bias_en = false;
    bool print_info = true;

    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 1;
    int C = 3;
    int H = 5;
    int W = 5;
    int M = 9;
    int pad = 0;
    if (padding_en) {
        pad = ker_size / 2;
    }

    TensorDim in_dim = {N, C, H, W};
    TensorDim filt_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
    float *in_col = malloc(out_dim.h * out_dim.w * in_dim.c *   
                        filt_dim.w * filt_dim.h * sizeof(float));
    float *weight_col = (float*)malloc(TensorSize(filt_dim) * sizeof(float));
    float *output_transpose = malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }
    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32(in_data, filters,     
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);

    // ******************************************** 打印输入输出信息 ***********************************************
    if(print_info){
        printf(" # input tensor: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor: [%d, %d, %d, %d] \n", out_dim.n, out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d \n", group, stride, pad);
        printf(" # in_col mat: [%d, %d] \n", out_dim.h*out_dim.w, in_dim.c * filt_dim.w * filt_dim.h);
        printf(" # weight_col mat: [%d, %d] \n", in_dim.c * filt_dim.w * filt_dim.h, out_dim.c);
    }


    // ******************************************** tensor_col * weight_col ***********************************************
    Im2ColConvIMW(in_data, filters, bias, in_col, weight_col, in_dim, out_dim, ker_size,  // image to col 算法
                    group, pad, stride, bias_en, output);

    Transpose2D(output, output_transpose, out_dim.w*out_dim.h, out_dim.c);


    if (TensorCompare(output_transpose, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
    free(in_col);
    free(output_transpose);
    }



    /*
    这里是 weight_cube * input_cube
    分块大小为 16*16
    */

    void TestIm2CubeConvLayer() {
    // Configurations
    // Enable kn2row or kn2col
    // bool im2col = true;
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;


    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 1;
    int C = 3;
    int H = 5;
    int W = 5;
    int M = 9;
    int pad = 2;
    int CUBE_row = 16;
    int CUBE_col = 16;


    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    ColDim in_2D_dim = {in_dim.c*ker_size*ker_size, out_dim.h*out_dim.w};
    ColDim weight_2D_dim = {out_dim.c, in_dim.c*ker_size*ker_size};
    ColDim out_2D_dim = {out_dim.c, out_dim.h*out_dim.w};
    int in_cube_row;
    if(in_2D_dim.row % CUBE_row == 0){
        in_cube_row = in_2D_dim.row / CUBE_row;
    }else{
        in_cube_row = in_2D_dim.row / CUBE_row + 1;
    }
    int in_cube_col;
    if(in_2D_dim.col % CUBE_col == 0){
        in_cube_col = in_2D_dim.col / CUBE_col;
    }else{
        in_cube_col = in_2D_dim.col / CUBE_col + 1;
    }
    int weight_cube_row;
    if(weight_2D_dim.row % CUBE_row == 0){
        weight_cube_row = weight_2D_dim.row / CUBE_row;
    }else{
        weight_cube_row = weight_2D_dim.row / CUBE_row + 1;
    }
    CubeDim in_cube_dim = {in_cube_row, in_cube_col};
    CubeDim weight_cube_dim = {weight_cube_row, in_cube_row};
    CubeDim out_cube_dim = {weight_cube_row, in_cube_col};
    




    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
    float *in_cube = (float*)calloc(in_cube_dim.row * in_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));
    float *weight_cube = (float*)calloc(weight_cube_dim.row * weight_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));
    float *CalCube = malloc(CUBE_row * CUBE_col * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # ****************      Alogrithm Im2Cube      **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor: [%d, %d, %d, %d] \n", out_dim.n, out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   CUBE_row:%d   CUBE_col:%d  \n", group, stride, pad, CUBE_row, CUBE_col);
        printf(" # weight2D_dim: [%d,%d]    in2D_dim: [%d,%d]    out2D_dim:[%d,%d]\n", weight_2D_dim.row, weight_2D_dim.col, in_2D_dim.row, in_2D_dim.col, out_2D_dim.row, out_2D_dim.col);
        printf(" # weight_cube_dim: [%d,%d]    in_cube_dim: [%d,%d]    out_cube_dim:[%d,%d]\n", weight_cube_dim.row, weight_cube_dim.col, in_cube_dim.row, in_cube_dim.col, out_cube_dim.row, out_cube_dim.col);
        printf(" # weight_cube: [%d, %d] \n",  weight_cube_dim.row * CUBE_row, weight_cube_dim.col * CUBE_col);
        printf(" # **************** malloc mem inside Alogrithm **************** #\n");
        printf(" # in_cube: [%d, %d] \n",  in_cube_dim.row * CUBE_row, in_cube_dim.col * CUBE_col);
        printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }

    // // ************************************** weight_cube Matmul in_cube *************************************
    Im2CubeConvLayer(in_data, in_dim, filters, weight_dim, bias, output, out_dim, 
        in_cube, in_2D_dim, in_cube_dim, weight_cube, weight_2D_dim, weight_cube_dim,
        CalCube, CUBE_row, CUBE_col, out_2D_dim, out_cube_dim,
        group, pad, stride, bias_en);


    // ********************************************   校验     ***********************************************
    if (TensorCompare(output, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }


    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
    free(in_cube);
    free(weight_cube);
    free(CalCube);
  // return 0;
}



void TestIm2MencopyConvLayer() {
    // Configurations
    // Enable kn2row or kn2col
    // bool im2col = true;
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;


    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 1;
    int C = 3;
    int H = 5;
    int W = 5;
    int M = 9;
    int pad = 1;
    int CUBE_row = 16;
    int CUBE_col = 16;


    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    ColDim in_2D_dim = {in_dim.c*ker_size*ker_size, out_dim.h*out_dim.w};
    ColDim weight_2D_dim = {out_dim.c, in_dim.c*ker_size*ker_size};
    ColDim out_2D_dim = {out_dim.c, out_dim.h*out_dim.w};
    int in_cube_row;
    if(in_2D_dim.row % CUBE_row == 0){
        in_cube_row = in_2D_dim.row / CUBE_row;
    }else{
        in_cube_row = in_2D_dim.row / CUBE_row + 1;
    }
    int in_cube_col;
    if(in_2D_dim.col % CUBE_col == 0){
        in_cube_col = in_2D_dim.col / CUBE_col;
    }else{
        in_cube_col = in_2D_dim.col / CUBE_col + 1;
    }
    int weight_cube_row;
    if(weight_2D_dim.row % CUBE_row == 0){
        weight_cube_row = weight_2D_dim.row / CUBE_row;
    }else{
        weight_cube_row = weight_2D_dim.row / CUBE_row + 1;
    }
    CubeDim in_cube_dim = {in_cube_row, in_cube_col};
    CubeDim weight_cube_dim = {weight_cube_row, in_cube_row};
    CubeDim out_cube_dim = {weight_cube_row, in_cube_col};
    




    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
    // float *in_cube = (float*)calloc(in_cube_dim.row * in_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));
    float *weight_cube = (float*)calloc(weight_cube_dim.row * weight_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));
    // float *CalCube = malloc(CUBE_row * CUBE_col * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # ****************     Alogrithm Im2Memcopy    **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor: [%d, %d, %d, %d] \n", out_dim.n, out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   CUBE_row:%d   CUBE_col:%d  \n", group, stride, pad, CUBE_row, CUBE_col);
        printf(" # weight2D_dim: [%d,%d]    in2D_dim: [%d,%d]    out2D_dim:[%d,%d]\n", weight_2D_dim.row, weight_2D_dim.col, in_2D_dim.row, in_2D_dim.col, out_2D_dim.row, out_2D_dim.col);
        printf(" # weight_cube_dim: [%d,%d]    in_cube_dim: [%d,%d]    out_cube_dim:[%d,%d]\n", weight_cube_dim.row, weight_cube_dim.col, in_cube_dim.row, in_cube_dim.col, out_cube_dim.row, out_cube_dim.col);
        printf(" # weight_cube: [%d, %d] \n",  weight_cube_dim.row * CUBE_row, weight_cube_dim.col * CUBE_col);
        printf(" # **************** malloc mem inside Alogrithm **************** #\n");
        printf(" # in_2D: [%d, %d] \n",  ker_size*ker_size*C, out_dim.h*out_dim.w);
        printf(" # in_cube: [%d, %d] \n",  in_cube_dim.row * CUBE_row, in_cube_dim.col * CUBE_col);
        printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }
    if(stride > 1){
        printf(" # ERROR: the stride more than one was not alowed in this alogrithm! \n");
        return;
    }

    // // ************************************** weight_cube Matmul in_cube *************************************
    Im2MemcopyLayer(in_data, in_dim, filters, weight_dim, bias, output, out_dim, 
        in_2D_dim, in_cube_dim, weight_cube, weight_2D_dim, weight_cube_dim,
        CUBE_row, CUBE_col, out_2D_dim, out_cube_dim,
        group, pad, stride, bias_en);

    // ********************************************   校验     ***********************************************
    if (TensorCompare(output, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }


    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
    free(weight_cube);
}


/*
    测试 CPU 版 NHWC排布卷积运算结果正确性
*/
void TestCPU_NHWC_conv() {
    bool bias_en = false;

    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 3;
    int group = 1;
    int stride = 2;
    int N = 1;
    int C = 5;
    int H = 24;
    int W = 36;
    int M = 9;
    int pad = 2;


    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32_nhwc(in_data, filters,     //CPU nhwc算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);

    float* data_nchw = malloc(TensorSize(in_dim) * sizeof(float));
    float* data_out_nchw = malloc(TensorSize(out_dim) * sizeof(float));
    float* data_out_nhwc = malloc(TensorSize(out_dim) * sizeof(float));
    NHWC2NCHW(in_data, in_dim.n, in_dim.c, in_dim.h, in_dim.w, data_nchw);
    RefConv2dF32(data_nchw, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, data_out_nchw);
    NCHW2NHWC(data_out_nchw, out_dim.n, out_dim.c, out_dim.h, out_dim.w, data_out_nhwc);
    TensorCompare(data_out_nhwc, ref_output, out_dim);

    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
}

/*
    Im2Col NHWC排布 批量取数
*/
void TestIm2BatchcopyConvLayer() {
    // Configurations
    // Enable kn2row or kn2col
    // bool im2col = true;
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;


    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 3;
    int group = 1;
    int stride = 3;
    int N = 1;
    int C = 64;
    int H = 120;
    int W = 120;
    int M = 2;
    int pad = 1;
    int CUBE_row = 16;
    int CUBE_col = 16;


    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    ColDim in_2D_dim = {out_dim.h*out_dim.w, in_dim.c*ker_size*ker_size};
    ColDim weight_2D_dim = {in_dim.c*ker_size*ker_size, out_dim.c};
    ColDim out_2D_dim = { out_dim.h*out_dim.w, out_dim.c};
    int in_cube_row;
    if(in_2D_dim.row % CUBE_row == 0){
        in_cube_row = in_2D_dim.row / CUBE_row;
    }else{
        in_cube_row = in_2D_dim.row / CUBE_row + 1;
    }
    int in_cube_col;
    if(in_2D_dim.col % CUBE_col == 0){
        in_cube_col = in_2D_dim.col / CUBE_col;
    }else{
        in_cube_col = in_2D_dim.col / CUBE_col + 1;
    }
    int weight_cube_col;
    if(weight_2D_dim.col % CUBE_col == 0){
        weight_cube_col = weight_2D_dim.col / CUBE_col;
    }else{
        weight_cube_col = weight_2D_dim.col / CUBE_col + 1;
    }
    CubeDim in_cube_dim = {in_cube_row, in_cube_col};
    CubeDim weight_cube_dim = {in_cube_col, weight_cube_col};
    CubeDim out_cube_dim = {in_cube_row, weight_cube_col};
    




    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
    // float *in_cube = (float*)calloc(in_cube_dim.row * in_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));
    float *weight_cube = (float*)calloc(weight_cube_dim.row * weight_cube_dim.col * CUBE_row * CUBE_col, sizeof(float));
    // float *CalCube = malloc(CUBE_row * CUBE_col * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32_nhwc(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);


    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # **************     Alogrithm Im2Batchcopy  nhwc   **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor  NHWC: [%d, %d, %d, %d] \n", N, H, W, C);
        printf(" # weight tensor  NCHW: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor  NHWC: [%d, %d, %d, %d] \n", out_dim.n, out_dim.h, out_dim.w, out_dim.c);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   CUBE_row:%d   CUBE_col:%d  \n", group, stride, pad, CUBE_row, CUBE_col);
        printf(" # weight2D_dim: [%d,%d]    in2D_dim: [%d,%d]    out2D_dim:[%d,%d]\n", weight_2D_dim.row, weight_2D_dim.col, in_2D_dim.row, in_2D_dim.col, out_2D_dim.row, out_2D_dim.col);
        printf(" # weight_cube_dim: [%d,%d]    in_cube_dim: [%d,%d]    out_cube_dim:[%d,%d]\n", weight_cube_dim.row, weight_cube_dim.col, in_cube_dim.row, in_cube_dim.col, out_cube_dim.row, out_cube_dim.col);
        printf(" # weight_cube: [%d, %d] \n",  weight_cube_dim.row * CUBE_row, weight_cube_dim.col * CUBE_col);
        printf(" # **************** malloc mem inside Alogrithm **************** #\n");
        printf(" # in_2D: [%d, %d] \n",  out_dim.h*out_dim.w, ker_size*ker_size*C);
        printf(" # in_cube: [%d, %d] \n",  in_cube_dim.row * CUBE_row, in_cube_dim.col * CUBE_col);
        printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }


    // // ************************************** in_cube Matmul weight_cube *************************************
    Im2BatchcopyLayer(in_data, in_dim, filters, weight_dim, bias, output, out_dim, 
        in_2D_dim, in_cube_dim, weight_cube, weight_2D_dim, weight_cube_dim,
        CUBE_row, CUBE_col, out_2D_dim, out_cube_dim,
        group, pad, stride, bias_en);

    // ********************************************   校验     ***********************************************
    if (TensorCompare(output, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }


    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
    free(weight_cube);
}



/*
winograd 算法   F(2*2, 3*3)
仅用于 3*3卷积，stride=1, dilation=1
推荐   w<120 h<120 in_c>16 out_c>16
现已支持pad   之后添加dilation

*/
void TestWinogradF23ConvLayer() {
    // Configurations
    // Enable kn2row or kn2col
    // bool im2col = true;
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;

    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 1;
    int C = 3;
    int H = 4;
    int W = 4;
    int M = 2;
    int pad = 1;


    TensorDim in_dim = {N, C, H, W};
    TensorDim filt_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));

    int tile_h = (in_dim.h%2)==0?(in_dim.h+pad+pad - 2) / 2 : (in_dim.h+pad+pad - 2) / 2+1;
    int tile_w = (in_dim.w%2)==0?(in_dim.w+pad+pad - 2) / 2 : (in_dim.w+pad+pad - 2) / 2+1;

    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # ****************     Alogrithm Winograd F23   **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor: [%d, %d, %d, %d] \n", out_dim.n, out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   \n", group, stride, pad);
        printf(" # tile_h: %d    tile_w: %d    \n", tile_h, tile_w);
        printf(" # **************** malloc mem inside Alogrithm **************** #\n");
        printf(" # transformed d: [%d, %d, %d] \n", in_dim.c, tile_h*4, tile_w*4);
        printf(" # transformed g: [%d, %d, %d, %d] \n", M, in_dim.c, 4, 4);
        printf(" # transformed out: [%d, %d, %d] \n", M, tile_h*2, tile_w*2);
        // printf(" # G: [%d, %d] \n",  ker_size*ker_size*C, out_dim.h*out_dim.w);
        // printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }
    if(stride > 1){
        printf(" # ERROR: the stride more than one was not alowed in this alogrithm! \n");
        return;
    }

    WinogradF22Layer(in_data, in_dim, filters, filt_dim, output, out_dim, stride, pad);


    if (TensorCompare(output, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
}


/*
winograd 算法  nchw   F(6*6, 3*3)    8*8  3*3  ->  6*6
仅用于 3*3卷积，stride=1, dilation=1
推荐   w<120 h<120 in_c>16 out_c>16
现已支持pad   之后添加dilation

*/
void TestWinogradF63ConvLayer() {
    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;

    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 1;
    int C = 128;
    int H = 84;
    int W = 84;
    int M = 256;
    int pad = 1;


    TensorDim in_dim = {N, C, H, W};
    TensorDim filt_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));


    //pad to 6n+2
    int tile_h = ((in_dim.h+pad+pad-2)%6)==0?(in_dim.h+pad+pad-2)/6 : (in_dim.h+pad+pad-2)/6+1;
    int tile_w = ((in_dim.w+pad+pad-2)%6)==0?(in_dim.w+pad+pad-2)/6 : (in_dim.w+pad+pad-2)/6+1;

    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU 算法校验 ***********************************************
    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, ref_output);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # ****************     Alogrithm Winograd F63   **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor: [%d, %d, %d, %d] \n", out_dim.n, out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   \n", group, stride, pad);
        printf(" # tile_h: %d    tile_w: %d    \n", tile_h, tile_w);
        printf(" # **************** malloc mem inside Alogrithm **************** #\n");
        printf(" # transformed d [in_c,tile_h,tile_w,tile] : [%d, %d, %d, %d*%d] \n", in_dim.c, tile_h, tile_w, 8,8);
        printf(" # transformed g [out_c,in_c,tile]: [%d, %d, %d*%d] \n", M, in_dim.c, 8, 8);
        printf(" # transformed out: [%d, %d, %d] \n", M, tile_h*6, tile_w*6);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }
    if(N > 1){
        printf(" # ERROR: havent support batch more than one ! \n");
        return;
    }
    if(stride > 1){
        printf(" # ERROR: the stride more than one was not alowed in this alogrithm! \n");
        return;
    }

    WinogradF63Layer(in_data, in_dim, filters, filt_dim, output, out_dim, stride, pad);

    if (TensorCompare(output, ref_output, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    free(in_data);
    free(bias);
    free(ref_output);
    free(filters);
    free(output);
}


/*
    昇腾卷积加速算法
    现在来看 支持 pad stride     那么group怎么办  dalition怎么办  dalition可以直接扩增weight_tensor
*/
void TestAscendConvLayer() {

    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;


    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 3;
    int group = 1;
    int stride = 1;
    int N = 2;
    // int C = 32;
    int C = 17;
    int H = 28;
    int W = 28;
    // int M = 64;
    int M = 18;

    int pad = 1;
    int CUBE_row = 16;
    int CUBE_col = 16;

    // 原始维度
    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    // 变换维度
    int c1;
    if(C % CUBE_row == 0){
        c1 = C / CUBE_row;
    }else{
        c1 = C / CUBE_row + 1;
    }
    int move;
    if((out_dim.h*out_dim.w) % CUBE_col ==0){
        move = (out_dim.h*out_dim.w) / CUBE_col;
    }else{
        move = (out_dim.h*out_dim.w) / CUBE_col + 1;
    }
    int kernal_cube;
    if(M % CUBE_col == 0){
        kernal_cube = M / CUBE_col;
    }else{
        kernal_cube = M / CUBE_col + 1;
    }
    Ascend5Dim in_5D_dim = {N, c1, H, W, CUBE_row};
    AscendTransform5Dim in_tran5D_dim = {N, move, c1, ker_size*ker_size, CUBE_row*CUBE_col};
    Ascend5Dim we_5D_dim = {c1, ker_size, ker_size, CUBE_row, M};
    AscendTransform5Dim we_tran5D_dim = {c1, ker_size, ker_size, kernal_cube, CUBE_row*CUBE_col};


    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *std_out = malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, std_out);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # **************     Alogrithm Ascend  Method   **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor  NCHW: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor  CCHW: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor  NCHW: [%d, %d, %d, %d] \n", out_dim.n,out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   CUBE_row:%d   CUBE_col:%d  \n", group, stride, pad, CUBE_row, CUBE_col);
        printf(" # **************** temp mem will malloc inside Alogrithm **************** #\n");
        printf(" # Ascend input  5D : [%d, %d, %d, %d, %d] \n",  in_5D_dim.n, in_5D_dim.c1, in_5D_dim.h, in_5D_dim.w, in_5D_dim.c0);
        printf(" # Ascend input transformed 5D : [%d, %d, %d, %d, %d] \n",  in_tran5D_dim.batch, in_tran5D_dim.move, in_tran5D_dim.channel, in_tran5D_dim.LW, in_tran5D_dim.cube);
        printf(" # **************** offline provide weight **************** #\n");
        printf(" # Ascend weight  5D : [%d, %d, %d, %d, %d] \n",  we_5D_dim.n, we_5D_dim.c1, we_5D_dim.h, we_5D_dim.w, we_5D_dim.c0);
        printf(" # Ascend weight transformed 5D : [%d, %d, %d, %d, %d] \n",  we_tran5D_dim.batch, we_tran5D_dim.move, we_tran5D_dim.channel, we_tran5D_dim.LW, we_tran5D_dim.cube);
        printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }
    if(stride > 1){
        printf(" # ERROR: this alogrithm isn't suitable for stride more than one! \n");
        return;
    }


    // // ************************************** Ascend *************************************
    // Im2BatchcopyLayer(in_data, in_dim, filters, weight_dim, bias, output, out_dim, 
    //     in_2D_dim, in_cube_dim, weight_cube, weight_2D_dim, weight_cube_dim,
    //     CUBE_row, CUBE_col, out_2D_dim, out_cube_dim,
    //     group, pad, stride, bias_en);
    Ascend(in_data, in_dim, filters, weight_dim, bias, output, out_dim, group, pad, stride, CUBE_row, CUBE_col,
            in_5D_dim, in_tran5D_dim, we_5D_dim, we_tran5D_dim);



    // ********************************************   校验     ***********************************************
    // if (TensorCompare(output, std_out, out_dim)) {
    //     printf("PASS\n");
    // } else {
    //     printf("FAIL\n");
    // }


    free(in_data);
    free(bias);
    free(std_out);
    free(filters);
    free(output);
}



/*
    昇腾卷积加速算法    NHWC输入 NHWC输出
*/
void TestAscendConvLayerNHWC() {

    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;


    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 3;
    int group = 1;  // 仅支持 1
    int stride = 1; // 仅支持 1
    int N = 4;
    int C = 35;
    int H = 36;
    int W = 20;
    int M = 47;

    int pad = 1;
    int CUBE_row = 16;  // CUBE_row 和 CUBE_col 需要相等  否则可能会报错，因为大概有混用的情况。
    int CUBE_col = 16;

    // 原始维度
    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    // 变换维度
    int c1;
    if(C % CUBE_row == 0){
        c1 = C / CUBE_row;
    }else{
        c1 = C / CUBE_row + 1;
    }
    int move;
    if((out_dim.h*out_dim.w) % CUBE_col ==0){
        move = (out_dim.h*out_dim.w) / CUBE_col;
    }else{
        move = (out_dim.h*out_dim.w) / CUBE_col + 1;
    }
    int kernal_cube;
    if(M % CUBE_col == 0){
        kernal_cube = M / CUBE_col;
    }else{
        kernal_cube = M / CUBE_col + 1;
    }
    Ascend5Dim in_5D_dim = {N, c1, H, W, CUBE_row};
    AscendTransform5Dim in_tran5D_dim = {N, move, c1, ker_size*ker_size, CUBE_row*CUBE_col};
    Ascend5Dim we_5D_dim = {c1, ker_size, ker_size, CUBE_row, M};
    AscendTransform5Dim we_tran5D_dim = {c1, ker_size, ker_size, kernal_cube, CUBE_row*CUBE_col};


    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *std_out = (float*)malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    // RefConv2dF32(in_data, filters,     //CPU算法  校准数据
    //     bias, in_dim.n, in_dim.c, in_dim.h,
    //     in_dim.w, out_dim.c, out_dim.h, out_dim.w,
    //     ker_size, group,
    //     pad, stride, bias_en, std_out);

    RefConv2dF32_nhwc(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, std_out);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # **************     Alogrithm Ascend  Method   **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor  NHWC: [%d, %d, %d, %d] \n", N, H, W, C);
        printf(" # weight tensor  CCHW: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor  NHWC: [%d, %d, %d, %d] \n", out_dim.n, out_dim.h, out_dim.w,out_dim.c);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   CUBE_row:%d   CUBE_col:%d  \n", group, stride, pad, CUBE_row, CUBE_col);
        printf(" # **************** temp mem will malloc inside Alogrithm **************** #\n");
        printf(" # Ascend input  5D : [%d, %d, %d, %d, %d] \n",  in_5D_dim.n, in_5D_dim.c1, in_5D_dim.h, in_5D_dim.w, in_5D_dim.c0);
        printf(" # Ascend input transformed 5D : [%d, %d, %d, %d, %d] \n",  in_tran5D_dim.batch, in_tran5D_dim.move, in_tran5D_dim.channel, in_tran5D_dim.LW, in_tran5D_dim.cube);
        printf(" # Ascend output 5D : [%d, %d, %d, %d, %d] \n",  in_tran5D_dim.batch, in_tran5D_dim.move, we_tran5D_dim.LW, CUBE_row, CUBE_col);
        printf(" # **************** offline provide weight **************** #\n");
        printf(" # Ascend weight  5D : [%d, %d, %d, %d, %d] \n",  we_5D_dim.n, we_5D_dim.c1, we_5D_dim.h, we_5D_dim.w, we_5D_dim.c0);
        printf(" # Ascend weight transformed 5D : [%d, %d, %d, %d, %d] \n",  we_tran5D_dim.batch, we_tran5D_dim.move, we_tran5D_dim.channel, we_tran5D_dim.LW, we_tran5D_dim.cube);
        printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }
    if(stride > 1){
        printf(" # ERROR: this alogrithm isn't suitable for stride more than one! \n");
        return;
    }


    // // ************************************** Ascend *************************************
    Ascend_A(in_data, in_dim, filters, weight_dim, bias, output, out_dim, group, pad, stride, CUBE_row, CUBE_col,
            in_5D_dim, in_tran5D_dim, we_5D_dim, we_tran5D_dim);

    // ********************************************   校验     ***********************************************
    if (TensorCompare(output, std_out, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    free(in_data);
    free(bias);
    free(std_out);
    free(filters);
    free(output);
}


void TestAscendConvLayerNCHW() {

    bool print_outputs = false;
    bool print_inputs = false;
    bool print_weight = false;
    bool bias_en = false;
    bool print_info = true;


    // ******************************************** 卷积基本信息 ***********************************************
    int ker_size = 4;
    int group = 1;  // 仅支持 1
    int stride = 1; // 仅支持 1
    int N = 2;
    int C = 35;
    int H = 29;
    int W = 47;
    int M = 37;

    int pad = 3;
    int CUBE_row = 16;  // CUBE_row 和 CUBE_col 需要相等  否则可能会报错，因为大概有混用的情况。
    int CUBE_col = 16;

    // 原始维度
    TensorDim in_dim = {N, C, H, W};
    TensorDim weight_dim = {M, C, ker_size, ker_size};
    TensorDim out_dim;
    out_dim.w = (in_dim.w + (pad + pad) - weight_dim.w) / stride + 1;
    out_dim.h = (in_dim.h + (pad + pad) - weight_dim.h) / stride + 1;
    out_dim.c = M;
    out_dim.n = in_dim.n;

    // 变换维度
    int c1;
    if(C % CUBE_row == 0){
        c1 = C / CUBE_row;
    }else{
        c1 = C / CUBE_row + 1;
    }
    int move;
    if((out_dim.h*out_dim.w) % CUBE_col ==0){
        move = (out_dim.h*out_dim.w) / CUBE_col;
    }else{
        move = (out_dim.h*out_dim.w) / CUBE_col + 1;
    }
    int kernal_cube;
    if(M % CUBE_col == 0){
        kernal_cube = M / CUBE_col;
    }else{
        kernal_cube = M / CUBE_col + 1;
    }
    Ascend5Dim in_5D_dim = {N, c1, H, W, CUBE_row};
    Cube5DDim in_tran5D_dim = {N, c1, ker_size*ker_size, move, CUBE_row*CUBE_col};
    Ascend5Dim we_5D_dim = {kernal_cube, CUBE_row, c1, ker_size*ker_size, CUBE_col};
    WeightCube5D we_tran5D_dim = {kernal_cube, c1, ker_size, ker_size, CUBE_row*CUBE_col};


    // ******************************************** 开辟所需空间 ***********************************************
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(weight_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *std_out = (float*)malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(weight_dim));
    if (bias_en) {
        RandInitF32(bias, out_dim.n * out_dim.c);
    }

    // ******************************************** CPU计算校验数据 ***********************************************
    RefConv2dF32(in_data, filters,     //CPU算法  校准数据
        bias, in_dim.n, in_dim.c, in_dim.h,
        in_dim.w, out_dim.c, out_dim.h, out_dim.w,
        ker_size, group,
        pad, stride, bias_en, std_out);
    
    // ******************************************** 打印基本信息 ***********************************************
    if(print_info){
        printf(" # **************     Alogrithm Ascend  Method   **************** #\n");
        printf(" # ****************          basic info         **************** #\n");
        printf(" # input tensor  NCHW: [%d, %d, %d, %d] \n", N, C, H, W);
        printf(" # weight tensor  CCHW: [%d, %d, %d, %d] \n", M, C, ker_size, ker_size);
        printf(" # output tensor  NCHW: [%d, %d, %d, %d] \n", out_dim.n,out_dim.c, out_dim.h, out_dim.w);
        printf(" # bias tensor: [%d] \n", out_dim.c);
        printf(" # group: %d    stride: %d    pad:%d   CUBE_row:%d   CUBE_col:%d  \n", group, stride, pad, CUBE_row, CUBE_col);
        printf(" # **************** temp mem will malloc inside Alogrithm **************** #\n");
        printf(" # Ascend input  5D : [%d, %d, %d, %d, %d] \n",  in_5D_dim.n, in_5D_dim.c1, in_5D_dim.h, in_5D_dim.w, in_5D_dim.c0);
        printf(" # Ascend input transformed 5D : [%d, %d, %d, %d, %d] \n",  in_tran5D_dim.batch, in_tran5D_dim.ch_cube, in_tran5D_dim.LW, in_tran5D_dim.move_cube, in_tran5D_dim.cube);
        printf(" # **************** offline provide weight **************** #\n");
        printf(" # Ascend weight  5D : [%d, %d, %d, %d, %d] \n",  we_5D_dim.n, we_5D_dim.c1, we_5D_dim.h, we_5D_dim.w, we_5D_dim.c0);
        printf(" # Ascend weight transformed 5D : [%d, %d, %d, %d, %d] \n",  we_tran5D_dim.num_cube, we_tran5D_dim.ch_cube, we_tran5D_dim.KH, we_tran5D_dim.KW, we_tran5D_dim.cube);
        printf(" # ClaCube: [%d, %d]  fixed \n",  CUBE_row, CUBE_col);
    }
    if(group > 1){
        printf(" # ERROR: the group more than one has not been supported! \n");
        return;
    }
    if(stride > 1){
        printf(" # ERROR: this alogrithm isn't suitable for stride more than one! \n");
        return;
    }


    // // ************************************** Ascend *************************************
    Ascend_B(in_data, in_dim, filters, weight_dim, bias, output, out_dim, group, pad, stride, CUBE_row, CUBE_col,
            in_5D_dim, in_tran5D_dim, we_5D_dim, we_tran5D_dim);

    // ********************************************   校验     ***********************************************
    if (TensorCompare(output, std_out, out_dim)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    
    free(in_data);
    free(bias);
    free(std_out);
    free(filters);
    free(output);
}

int main(void) {


    //1.  测试 weight_col matmul input_tensor_col
    TestIm2FlavorConvLayer();

    //2. 测试 input_tensor_col matmul weight_col
    // TestIm2ColConvIMW();

    //3.  测试 weight_cube matmul input_cube  逐个取数  NCHW
    // TestIm2CubeConvLayer();

    //4.  测试 image -> col_2D -> cube with multi-mem-copy  批量取数
    // TestIm2MencopyConvLayer();

    //5.  测试 image -> col_2D -> cube with multi-mem-copy  批量取数  NHWC 版本
    // TestIm2BatchcopyConvLayer();

    //6.  测试 winograd alogrithm  F(2*2, 3*3)  
    // TestWinogradF23ConvLayer();

    //7.  测试 winograd alogrithm  F(6*6, 3*3)  
    // TestWinogradF63ConvLayer();

    //8.  测试 NHWC 排布 CPU卷积算法的正确性
    // TestCPU_NHWC_conv();

    //9.  测试 昇腾 卷积算法加速
    // TestAscendConvLayer();

    //10.  测试 昇腾 卷积算法加速
    // TestAscendConvLayerNCHW();

    //11. 测试 昇腾卷积算法加速 NHWC
    // TestAscendConvLayerNHWC();
    return 0;
}
