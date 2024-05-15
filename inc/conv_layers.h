
#ifndef INC_CONV_LAYERS_H_
#define INC_CONV_LAYERS_H_
#include <stdbool.h>
#include <common_types.h>

void RefConv2dF32(const float *input, const float *weight,
    const float *bias, const int in_n, const int in_c, const int in_h,
    const int in_w, const int out_c, const int out_h, const int out_w,
    const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output);

void RefConv2dF32_nhwc(const float *input, const float *weight,
    const float *bias, const int in_n, const int in_c, const int in_h,
    const int in_w, const int out_c, const int out_h, const int out_w,
    const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output);


void Im2ColConvLayer(const float *input, const float *weight,
    const float *bias, float *scratchpad, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output);

void Im2ColConvIMW(const float *input, const float *weight,
    const float *bias, float *scratchpad, float *weight_col, const TensorDim in_dim,
    const TensorDim out_dim, const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output);

void Im2CubeConvLayer(const float *input, const TensorDim in_dim, const float *weight, const TensorDim weight_dim, 
    float *bias, float *output, const TensorDim out_dim, 
    float* in_cube, const ColDim in_2D_dim, const CubeDim in_cube_dim, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim,
    float* CalCube, const int CUBE_row, const int CUBE_col, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int group, const int pad, const int stride, const int bias_en);

void Im2MemcopyLayer(const float *input, const TensorDim in_dim, const float *weight, const TensorDim weight_dim, 
    float *bias, float *output, const TensorDim out_dim, 
    const ColDim in_2D_dim, const CubeDim in_cube_dim, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim,
    const int CUBE_row, const int CUBE_col, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int group, const int pad, const int stride, const int bias_en);

void Im2BatchcopyLayer(const float *input, const TensorDim in_dim, const float *weight, const TensorDim weight_dim, 
    float* bias, float *output, const TensorDim out_dim, 
    const ColDim in_2D_dim, const CubeDim in_cube_dim, float* weight_cube, const ColDim weight_2D_dim, const CubeDim weight_cube_dim,
    const int CUBE_row, const int CUBE_col, const ColDim out_2D_dim, const CubeDim out_cube_dim,
    const int group, const int pad, const int stride, const int bias_en);

void WinogradF22Layer(const float* in_data, const TensorDim in_dim, float* weight, const TensorDim weight_dim,
    float* output, const TensorDim out_dim, int stride, int pad);

void WinogradF63Layer(const float* in_data, const TensorDim in_dim, const float* weight, const TensorDim weight_dim,
    float* output, const TensorDim out_dim, int stride, int pad);



void Ascend(const float*in_data, const TensorDim in_dim, const float* filters, const TensorDim weight_dim,
        float* bias, float* output, const TensorDim out_dim, int group, int pad, int stride,  int CUBE_row, int CUBE_col,
        Ascend5Dim in_5D_dim, AscendTransform5Dim in_tran5D_dim, Ascend5Dim we_5D_dim, AscendTransform5Dim we_tran5D_dim);
#endif  // INC_CONV_LAYERS_H_
