
//
#include <cblas.h>
#include <stdbool.h>
#include "common_types.h"
#include "conv_layers.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "data_reshape.h"



// 以 we_5D_dim 为参考 从4D维数据向we_tran5D_dim拷贝数据   大Z小N排布
void WeightTrans_A(const float* filters, const TensorDim weight_dim, Ascend5Dim we_5D_dim, float* we_tran5D, 
            AscendTransform5Dim we_tran5D_dim, int CUBE_row, int CUBE_col){
    int lastdim4 = we_tran5D_dim.move * we_tran5D_dim.channel * we_tran5D_dim.LW * we_tran5D_dim.cube;
    int lastdim3 = we_tran5D_dim.channel * we_tran5D_dim.LW * we_tran5D_dim.cube;
    int lastdim2 = we_tran5D_dim.LW * we_tran5D_dim.cube;
    int single_filter_num = weight_dim.c * weight_dim.h * weight_dim.w;
    int single_filter_channel = weight_dim.h * weight_dim.w;


    for(int ch_cube=0; ch_cube<we_tran5D_dim.batch; ch_cube++){  //通道方向块   ch_cube
        int index_1 = ch_cube * lastdim4;
        for(int hk=0; hk<we_tran5D_dim.move; hk++){  // filter 长  
            int index_2 = index_1 + hk * lastdim3;
            for(int wk=0; wk<we_tran5D_dim.channel; wk++){  // filter 宽
                int index_3 = index_2 + wk * lastdim2;
                for(int cout_cube=0; cout_cube<we_tran5D_dim.LW; cout_cube++){ // cout方向块 
                    int index_4 = index_3 + cout_cube*we_tran5D_dim.cube;
                    for(int cube_row=0; cube_row<CUBE_row; cube_row++){
                        for(int cube_col=0; cube_col<CUBE_col; cube_col++){
                            int index = index_4 + cube_row*CUBE_col + cube_col;  // we_tran5D 的连续顺序
                            
                            if((cout_cube*CUBE_col+cube_row)>=weight_dim.n  || (ch_cube*CUBE_col+cube_col)>=weight_dim.c){
                                we_tran5D[index] = 0;
                            }else{
                                int index_from = (cout_cube*CUBE_col+cube_row)*single_filter_num + (ch_cube*CUBE_col+cube_col)*single_filter_channel + hk*weight_dim.w+ wk;
                                
                                we_tran5D[index] = filters[index_from];
                            }

                        }
                    }
                }
            }
        }
    }
}

// 4D 转 5D 在线软件进行，或者只在输入特征为 4D 时进行  [n,h,w,c] -> [n, c/k_cube, h, w, k_cube] 
void Input4D25D_A(const float*in_data, const TensorDim in_dim, int pad, int stride, int CUBE_row, 
            int CUBE_col, Ascend5Dim in_5D_dim, float* in_5D){
    
    int single_input_num = in_dim.c * in_dim.h * in_dim.w;  
    int singel_layer_num = in_dim.h * in_dim.w;   
    int lastdim4 = in_5D_dim.c1*in_5D_dim.h*in_5D_dim.w*in_5D_dim.c0; // 2*28*28*16 = 25088
    int lastdim3 = in_5D_dim.h*in_5D_dim.w*in_5D_dim.c0;  // 28*28*16 = 12544
    int lastdim2 = in_5D_dim.w*in_5D_dim.c0; // 448

    for(int batch=0; batch<in_5D_dim.n; batch++){
        int index_1 = batch * lastdim4;
        for(int c1=0; c1<in_5D_dim.c1; c1++){
            int index_2 = index_1 + c1 * lastdim3;
            for(int h=0; h<in_5D_dim.h; h++){
                int index_3 = index_2 + h * lastdim2;
                for(int w=0; w<in_5D_dim.w; w++){
                    int index_4 = index_3 + w * in_5D_dim.c0;
                    for(int c0=0; c0<in_5D_dim.c0; c0++){
                        int index = index_4 + c0;
                        // batch channel h w 
                        if((c1*CUBE_row+c0)>=in_dim.c){
                            in_5D[index] = 0;
                        }else{
                            int index_from = batch*single_input_num + (h*in_dim.w + w)*in_dim.c + c1*CUBE_row+c0;
                            in_5D[index] = in_data[index_from];
                        }
                        
                    }
                }
            }
        }
    }
}


bool Is_in_range_A(int a, int b) {  // 注意 ： 负值比正值大    
    return (unsigned int)a < (unsigned int)(b);
}


void Input5D2Cube_A(float* in_5D, Ascend5Dim in_5D_dim, const TensorDim weight_dim, float* in_tran5D, 
    AscendTransform5Dim in_tran5D_dim, const TensorDim out_dim, int pad, int stride, int CUBE_row, int CUBE_col){
    float* zeroBank = (float*)calloc(256, sizeof(float));
    int singleBatch = in_5D_dim.c1 * in_5D_dim.h * in_5D_dim.w * in_5D_dim.c0;
    int singleCCube = in_5D_dim.h * in_5D_dim.w * in_5D_dim.c0;
    int singleWC0 = in_5D_dim.w * in_5D_dim.c0;
    int in_tran5D_index = 0;
    int zero_range;
    int data_range_front;
    int data_range_tail;
    int from_index;
    int data_start;
    int data_end;

    for(int batch=0; batch<in_tran5D_dim.batch; batch++){
        int index_1 = batch * singleBatch;
        for(int move_cube=0; move_cube<in_tran5D_dim.move; move_cube++){
            for(int chan_cube=0; chan_cube<in_tran5D_dim.channel; chan_cube++){
                int index_2 = index_1 + chan_cube*singleCCube;
                for(int kh=0; kh<weight_dim.h; kh++){  // kernal纵向
                    int line_start = (move_cube*CUBE_row)/out_dim.w - pad + kh;  // 所属行 index   这里有bug  但是还不想改  后续改一改  问题
                    int line_end = ((move_cube+1)*CUBE_row-1)/out_dim.w - pad + kh; // 所属行结束点 index
                    if(line_end - line_start > 1){
                        printf(" # Hey, you know what ?  FXXK you ! \n");
                        return;
                    }
                    for(int kw=0; kw<weight_dim.w; kw++){  // kernal横向 
                        int move_start = (move_cube*CUBE_row)%out_dim.w + kw -pad;  // 所属列 index
                        int move_end = ((move_cube+1)*CUBE_row-1)%out_dim.w + kw - pad;  // 所属列结束点 index
                        data_start = kw-pad>0?kw-pad:0;
                        data_end = weight_dim.w-kw-1>pad?weight_dim.w-kw-1-pad:0;  // kw后面还有几个元素
                        if((!Is_in_range_A(line_start, in_5D_dim.h)) && (!Is_in_range_A(line_end, in_5D_dim.h))){  // 1. 起始终止行列均在范围外
                            memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*CUBE_row*CUBE_col);
                            in_tran5D_index += CUBE_row*CUBE_col;
                            
                        }else if(!Is_in_range_A(line_start, in_5D_dim.h)){    // 2.起始行在范围外 终止行在范围内
                            zero_range =  in_5D_dim.w - move_start - data_end;
                            
                            if(move_start>=in_5D_dim.w){  // 起始index位于右边 pad
                                    zero_range = 0;
                            }
                            data_range_tail = move_end+1-data_start;
                            if(move_end<0){     // 终止index位于左边 pad
                                data_range_tail = 0;
                            }
                            zero_range += CUBE_col - zero_range - data_range_tail;  // 加上中间pad长度 
                            memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                            in_tran5D_index += zero_range*CUBE_col;
                            from_index = index_2+ line_end*singleWC0+data_start*in_5D_dim.c0;
                            memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_tail*CUBE_col);
                            in_tran5D_index += data_range_tail*CUBE_col;
                        }else if(Is_in_range_A(line_start, in_5D_dim.h) && Is_in_range_A(line_end, in_5D_dim.h)){  // 3. 起始终止行都在范围内
                            if(line_end == line_start){ // 3.1 起始终止行位于同一行
                                if(move_start<0){  // 3.1.1 前面pad
                                    zero_range = -move_start;
                                    memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                                    in_tran5D_index += zero_range*CUBE_col;
                                    data_range_tail = move_end +1;
                                    from_index = index_2+ line_end*singleWC0+ data_start*in_5D_dim.c0;
                                    memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_tail*CUBE_col);
                                    in_tran5D_index += data_range_tail*CUBE_col;

                                }else if(move_end>=in_5D_dim.w){  // 3.1.2 后面pad
                                    data_range_front = in_5D_dim.w - move_start - data_end;
                                    from_index = index_2+ line_start*singleWC0+ move_start*in_5D_dim.c0;
                                    memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_front*CUBE_col);
                                    in_tran5D_index += data_range_front*CUBE_col;
                                    zero_range = CUBE_col-data_range_front;
                                    memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                                    in_tran5D_index += zero_range*CUBE_col;
                                }else{  // 3.1.2 不需要pad
                                    data_range_front = CUBE_col;
                                    from_index = index_2+ line_start*singleWC0+ move_start*in_5D_dim.c0;
                                    memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_front*CUBE_col);
                                    in_tran5D_index += data_range_front*CUBE_col;
                                }
                            }else{      // 3.2 起始终止行位于上下两行
                                data_range_front = in_5D_dim.w - move_start - data_end;
                                if(move_start>=in_5D_dim.w){  // 起始index位于右边 pad
                                    data_range_front = 0;
                                }
                                data_range_tail = move_end+1-data_start;
                                if(move_end<0){   // 终止index位于左边 pad
                                    data_range_tail = 0;
                                }
                                zero_range = CUBE_col - data_range_front - data_range_tail;

                                from_index = index_2+ line_start*singleWC0+ move_start*in_5D_dim.c0;
                                memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_front*CUBE_col);
                                
                                in_tran5D_index += data_range_front*CUBE_col;
                                memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);

                                in_tran5D_index += zero_range*CUBE_col;
                                from_index = index_2+ line_end*singleWC0+ data_start*in_5D_dim.c0;
                                memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_tail*CUBE_col);
                                in_tran5D_index += data_range_tail*CUBE_col;
                            }
                        }
                        else{  // 4. 起始行在范围内 终止行在范围外
                            data_range_front = in_5D_dim.w - move_start - data_end;
                            if(move_start>=in_5D_dim.w){  // 起始index位于右边 pad
                                    data_range_front = 0;
                            }
                            from_index = index_2+ line_start*singleWC0+ move_start*in_5D_dim.c0;
                            memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_front*CUBE_col);
                            in_tran5D_index += data_range_front*CUBE_col;
                            zero_range = CUBE_col - data_range_front;
                            memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                            in_tran5D_index += zero_range*CUBE_col;
                        }
                    }
                }
            }
        }
    }
}

void CubeMemSetZero_A(float* CalCubeLast, int elemNum){
    for(int i=0; i<elemNum; i++){
        CalCubeLast[i] = 0;
    }
}

void CubeMatrixMul_A(float* in_tran5D, AscendTransform5Dim in_tran5D_dim, float* we_tran5D, AscendTransform5Dim we_tran5D_dim, 
        float* out_tran, const TensorDim out_dim, int CUBE_row, int CUBE_col){
    
    float *CalCube = (float*)malloc(CUBE_row * CUBE_col * sizeof(float));
    int singleBatch = in_tran5D_dim.move *in_tran5D_dim.channel*in_tran5D_dim.LW*in_tran5D_dim.cube;
    int singleCCube = in_tran5D_dim.channel*in_tran5D_dim.LW*in_tran5D_dim.cube;
    int singleLW = in_tran5D_dim.LW*in_tran5D_dim.cube;

    int outSingleBatch = out_dim.h*out_dim.w*out_dim.c;
    int outSingleMove = CUBE_row*out_dim.c;
    int weSingleCOut = we_tran5D_dim.LW*we_tran5D_dim.cube;

    int row_copy_range;
    int col_copy_range;
    int out_index;

    for(int batch=0; batch<in_tran5D_dim.batch; batch++){
        int out_index1 = batch * outSingleBatch;
        int in_index1 = batch*singleBatch;
        for(int move_cube=0; move_cube<in_tran5D_dim.move; move_cube++){
            int out_index2 = out_index1+ move_cube*outSingleMove;
            int in_index2 = in_index1 + move_cube*singleCCube;
            row_copy_range = (move_cube+1)*CUBE_row>(out_dim.h*out_dim.w)?(out_dim.h*out_dim.w)%CUBE_row : CUBE_row;
            for(int out_cube=0; out_cube<we_tran5D_dim.LW; out_cube++){  // 输出块
                col_copy_range = (out_cube+1)*CUBE_col>out_dim.c?out_dim.c%CUBE_col:CUBE_col;
                CubeMemSetZero_A(CalCube, CUBE_row*CUBE_col);
                for(int horizon_cube=0; horizon_cube<in_tran5D_dim.channel*in_tran5D_dim.LW; horizon_cube++){  // 横向块
                    int in_index3 = in_index2 + horizon_cube*in_tran5D_dim.cube;
                    int we_index1 = horizon_cube*weSingleCOut + out_cube*we_tran5D_dim.cube;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        CUBE_row, CUBE_row, CUBE_col, 1.0, in_tran5D+in_index3,
                        CUBE_row, we_tran5D+we_index1, CUBE_col, 1.0, CalCube, CUBE_row);
                }
                
                // 将计算结果直接拷贝进 [n,h,w,c]  也就是[2,28,28,18]  每次只能拷贝16个数字  还得计算拷贝的范围
                for(int row_copy_index=0; row_copy_index<row_copy_range; row_copy_index++){
                    
                    
                    out_index = out_index2 + row_copy_index*out_dim.c + out_cube*CUBE_col;
                    memcpy(out_tran+out_index, CalCube+row_copy_index*CUBE_col, sizeof(float)*col_copy_range);
                }
                
                
            }
        }
    }


    free(CalCube);
}



void transNhwc2Nchw_A(float* target, float* source, int N, int C, int H, int W){
    int singleBatch = C*H*W;
    int singleHW = H*W;
    for(int batch=0; batch<N; batch++){
        int tar_index1 = batch*singleBatch;
        for(int ch=0; ch<C; ch++){
            int tar_index2 = tar_index1+ ch*singleHW;
            for(int h=0; h<H; h++){
                int tar_index3 = tar_index2 + h *W;
                for(int w=0; w<W; w++){
                    target[tar_index3+w] = source[tar_index1+ (h*W+w)*C + ch];
                }
            }
        }
    }
}


void Ascend_A(const float* in_data, const TensorDim in_dim, const float* filters, const TensorDim weight_dim,
        float* bias, float* output, const TensorDim out_dim, int group, int pad, int stride, int CUBE_row, int CUBE_col,
        Ascend5Dim in_5D_dim, AscendTransform5Dim in_tran5D_dim, Ascend5Dim we_5D_dim, AscendTransform5Dim we_tran5D_dim){
        
        // 权重 4D转5D 矩阵重排  离线进行 
        float* we_tran5D = (float*)malloc(sizeof(float)*we_tran5D_dim.batch*we_tran5D_dim.move*we_tran5D_dim.channel*we_tran5D_dim.LW*we_tran5D_dim.cube);
        WeightTrans_A(filters, weight_dim, we_5D_dim, we_tran5D, we_tran5D_dim, CUBE_row, CUBE_col);
        
        // 输入tensor  4D转5D 在线软件完成(或者只在输入为4D时完成)
        float* in_5D = (float*)malloc(sizeof(float)*in_5D_dim.n*in_5D_dim.c1*in_5D_dim.h*in_5D_dim.w*in_5D_dim.c0);
        float* in_tran5D = (float*)malloc(sizeof(float)*in_tran5D_dim.batch*in_tran5D_dim.move*in_tran5D_dim.channel*in_tran5D_dim.LW*in_tran5D_dim.cube);
        
        // float* out_tran = (float*)malloc(sizeof(float)*in_tran5D_dim.batch*in_tran5D_dim.move*we_tran5D_dim.LW*CUBE_row*CUBE_col);
        // float* out_tran = (float*)malloc(sizeof(float)*out_dim.n*out_dim.h*out_dim.w*out_dim.c);  // 现在获得的是 nhwc 排布
        Input4D25D_A(in_data, in_dim, pad, stride, CUBE_row, CUBE_col, in_5D_dim, in_5D);
        
        // 输入tensor  5D Im2Col(2D)+矩阵重排(大Z小Z)  在线硬件完成
        Input5D2Cube_A(in_5D, in_5D_dim, weight_dim, in_tran5D, in_tran5D_dim, out_dim, pad, stride, CUBE_row, CUBE_col);
        
        // 分块矩阵乘运算 最后一步
        CubeMatrixMul_A(in_tran5D, in_tran5D_dim, we_tran5D, we_tran5D_dim, output, out_dim, CUBE_row, CUBE_col);
        
        free(we_tran5D);
        free(in_tran5D);
    printf("  #  info : this is ascend method .\n");
};
