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
#include "common_types.h"
#include "conv_layers.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
// 以 we_5D_dim 为参考 从4D维数据向we_tran5D_dim拷贝数据
void WeightTrans(const float* filters, const TensorDim weight_dim, Ascend5Dim we_5D_dim, float* we_tran5D, 
            AscendTransform5Dim we_tran5D_dim, int CUBE_row, int CUBE_col){
    int lastdim4 = we_tran5D_dim.move * we_tran5D_dim.channel * we_tran5D_dim.LW * we_tran5D_dim.cube;
    int lastdim3 = we_tran5D_dim.channel * we_tran5D_dim.LW * we_tran5D_dim.cube;
    int lastdim2 = we_tran5D_dim.LW * we_tran5D_dim.cube;
    int single_filter_num = weight_dim.c * weight_dim.h * weight_dim.w;
    int single_filter_channel = weight_dim.h * weight_dim.w;

    for(int ch_cube=0; ch_cube<we_tran5D_dim.batch; ch_cube++){  //通道方向块 
        int index_1 = ch_cube * lastdim4;
        for(int hk=0; hk<we_tran5D_dim.move; hk++){  // filter 长
            int index_2 = index_1 + hk * lastdim3;
            for(int wk=0; wk<we_tran5D_dim.channel; wk++){  // filter 宽
                int index_3 = index_2 + wk * lastdim2;
                for(int cout_cube=0; cout_cube<we_tran5D_dim.LW; cout_cube++){ // cout方向块
                    int index_4 = index_3 + cout_cube*we_tran5D_dim.cube;
                    for(int cube_row=0; cube_row<CUBE_row; cube_row++){
                        for(int cube_col=0; cube_col<CUBE_col; cube_col++){
                            int index = index_4 + cube_row*CUBE_col + cube_col;
                            
                            if((cout_cube*CUBE_col+cube_col)>=weight_dim.n  || (ch_cube*CUBE_row+cube_row)>=weight_dim.c){
                                we_tran5D[index] = 0;
                                // printf("tran[%d] =0      ", index);
                                // if(index == 8961){
                                //     printf("tran[%d] = 0 \n", index);
                                // }
                            }else{
                                // 第几个filter  第几个通道  第几行  第几列
                                int index_from = (cout_cube*CUBE_col+cube_col)*single_filter_num + (ch_cube*CUBE_row+cube_row)*single_filter_channel + hk*weight_dim.w+ wk;
                                we_tran5D[index] = filters[index_from];
                                // printf("tran[%d] = F[%d] ", index, index_from);
                                // if(index == 8961){
                                //     printf("tran[%d] = F[%d] \n", index, index_from);
                                // }
                            }
                            // if(index == 8961){
                            //     printf(" ===  ch_cube:%d   hk:%d  wk:%d  cout_cube:%d  cube_row:%d  cube_col:%d \n",ch_cube, hk, wk, cout_cube, cube_row, cube_col);
                            // }
                            
                        
                        }
                    }
                }
            }
        }
    }
}

// 4D 转 5D 在线软件进行，或者只在输入特征为 4D 时进行   
// 这里的4D还是 NCHW   没有快速方法  pad是什么时候完成的,这里完成的话很方便后面
void Input4D25D(const float*in_data, const TensorDim in_dim, int pad, int stride, int CUBE_row, 
            int CUBE_col, Ascend5Dim in_5D_dim, float* in_5D){
    
    int single_input_num = in_dim.c * in_dim.h * in_dim.w;
    int singel_layer_num = in_dim.h * in_dim.w;
    int lastdim4 = in_5D_dim.c1*in_5D_dim.h*in_5D_dim.w*in_5D_dim.c0;
    int lastdim3 = in_5D_dim.h*in_5D_dim.w*in_5D_dim.c0;
    int lastdim2 = in_5D_dim.w*in_5D_dim.c0;

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
                            // printf("in_5D[%d] = [%d]  ", index, 0);
                            // if(index==12545){
                            //     printf("in_5D[%d] = [%d]  ", index, 0);
                            // }
                        }else{
                            int index_from = batch*single_input_num + (c1*CUBE_row+c0)*singel_layer_num + h*in_dim.w + w;
                            in_5D[index] = in_data[index_from];
                            // printf("in_5D[%d] = [%d]  ", index, index_from);
                            // if(index==12545){
                            //     printf("in_5D[%d] = [%d]  ", index, index_from);
                            // }
                        }
                        
                    }
                }
            }
        }
    }
}


bool Is_in_range(int a, int b) {  // 注意 ： 负值比正值大    
    return (unsigned int)a < (unsigned int)(b);
}


// 一步完成  Im2Col(2D)+矩阵重排(大Z小Z) 一次能够拷贝256个数据
// 咱就是说，这个卷积算法真的高效吗？这 TM 是搞笑吧，算法这么复杂，最后说不定还没我以前写的速度快。
// 当 stride>1时，一次性最多拷贝16个数 ？？？？？   stride>1时效率比我以前的低很多
// stride=1时，也不见得效率高啊  关键是我不能把stride忽略掉，因为 .... 算了 stride>1根本不适合使用这种算法  所以这里只有stride=1的解决方案
// 怎么才能确定 这 256 个数呢？ 前后不连续（纵移），中间不连续（pad），甚至当输出特征图的宽小于16的时候，会出现多个前中后问题，
// 多少个还是不确定的，这不就是非得每16个数都要判断一下拷贝位置么，这...  已经比我以前写的东西效率低了吧。


void Input5D2Cube(float* in_5D, Ascend5Dim in_5D_dim, const TensorDim weight_dim, float* in_tran5D, 
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

    for(int batch=0; batch<in_tran5D_dim.batch; batch++){
        int index_1 = batch * singleBatch;
        for(int move_cube=0; move_cube<in_tran5D_dim.move; move_cube++){
            for(int chan_cube=0; chan_cube<in_tran5D_dim.channel; chan_cube++){
                int index_2 = index_1 + chan_cube*singleCCube;
                for(int kh=0; kh<weight_dim.h; kh++){  // kernal纵向
                    int line_start = (move_cube*CUBE_row)/out_dim.w - pad;  // 所属行 index   这里有bug  但是还不想改  后续改一改  问题
                    int line_end = ((move_cube+1)*CUBE_row-1)/out_dim.w - pad; // 所属行结束点 index
                    if(line_end - line_start > 1){
                        // 这种也不适合使用 这种方法， 因为跨及多行，不太行  得循环的判断加载数据，且效率不高
                        printf(" # Hey, you know what ?  FXXK you ! \n");
                        return;
                    }
                    for(int kw=0; kw<weight_dim.w; kw++){  // kernal横向 
                        int move_start = (move_cube*CUBE_row)%out_dim.w + kw -pad;  // 所属列 index
                        int move_end = ((move_cube+1)*CUBE_row)%out_dim.w + kw - 1 - pad;  // 所属列结束点 index
                        data_start = kw-pad>0?kw-pad:0;
                        if((!Is_in_range(line_start, in_5D_dim.h)) && (!Is_in_range(line_end, in_5D_dim.h))){  // 1. 起始终止行列均在范围外
                            memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*CUBE_row*CUBE_col);
                            in_tran5D_index += CUBE_row*CUBE_col;
                            printf("1.两行均在范围外   copy:%d  copyfrom:zero bank  dataget:%d     linestart:%d lineend:%d\n", CUBE_row, in_tran5D_index, line_start, line_end);
                        }else if(!Is_in_range(line_start, in_5D_dim.h)){    // 2.起始行在范围外 终止行在范围内
                            zero_range = in_5D_dim.w + pad - (weight_dim.w-kw-1) - move_start;  // 后续长度   【输入tensor宽度 + 右边的pad -（卷积核中该位置扫描不到的长度）- 起始index】
                            data_range_tail = move_end+1-kw; // 下一行数据长度
                            zero_range += CUBE_col - zero_range - data_range_tail;  // 加上中间pad长度 
                            memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                            in_tran5D_index += zero_range*CUBE_col;
                            from_index = index_2+ line_end*singleWC0+data_start*in_5D_dim.c0;
                            memcmp(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_tail*CUBE_col);
                            in_tran5D_index += data_range_tail*CUBE_col;
                        }else if(Is_in_range(line_start, in_5D_dim.h) && Is_in_range(line_end, in_5D_dim.h)){  // 3. 起始终止行都在范围内
                            if(line_end == line_start){ // 3.1 起始终止行位于同一行
                                if(move_start<0){  // 3.1.1 前面pad
                                    zero_range = -move_start;
                                    memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                                    in_tran5D_index += zero_range*CUBE_col;
                                    data_range_tail = move_end +1;
                                    from_index = index_2+ line_end*singleWC0+ data_start*in_5D_dim.c0;
                                    memcmp(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_tail*CUBE_col);
                                    in_tran5D_index += data_range_tail*CUBE_col;
                                }else if(move_end>=in_5D_dim.w){  // 3.1.2 后面pad
                                    data_range_front = in_5D_dim.w + pad - (weight_dim.w-kw-1) - move_start;
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
                                data_range_front = in_5D_dim.w + pad - (weight_dim.w-kw-1) - move_start;  // 前部数据长度
                                data_range_tail = move_end+1-kw;
                                zero_range = CUBE_col - data_range_front - data_range_tail;
                                from_index = index_2+ line_start*singleWC0+ move_start*in_5D_dim.c0;
                                memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_front*CUBE_col);
                                in_tran5D_index += data_range_front*CUBE_col;
                                memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range*CUBE_col);
                                in_tran5D_index += zero_range*CUBE_col;
                                from_index = index_2+ line_end*singleWC0+ data_start*in_5D_dim.c0;
                                memcmp(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_tail*CUBE_col);
                                in_tran5D_index += data_range_tail*CUBE_col;
                            }
                        }
                        else{  // 4. 起始行在范围内 终止行在范围外
                            data_range_front = in_5D_dim.w + pad - (weight_dim.w-kw-1) - move_start;
                            from_index = index_2+ line_start*singleWC0+ move_start*in_5D_dim.c0;
                            memcpy(in_tran5D+in_tran5D_index, in_5D+from_index, sizeof(float)*data_range_front);
                            in_tran5D_index += data_range_front;
                            zero_range = CUBE_col - data_range_front;
                            memcpy(in_tran5D+in_tran5D_index, zeroBank, sizeof(float)*zero_range);
                            in_tran5D_index += zero_range;
                        }
                        // if(move)
                    }
                }
            }
        }
    }
}


void Ascend(const float*in_data, const TensorDim in_dim, const float* filters, const TensorDim weight_dim,
        float* bias, float* output, const TensorDim out_dim, int group, int pad, int stride, int CUBE_row, int CUBE_col,
        Ascend5Dim in_5D_dim, AscendTransform5Dim in_tran5D_dim, Ascend5Dim we_5D_dim, AscendTransform5Dim we_tran5D_dim){
        

        // 权重 4D转5D 矩阵重排  离线进行 
        float* we_tran5D = (float*)malloc(sizeof(float)*we_tran5D_dim.batch*we_tran5D_dim.move*we_tran5D_dim.channel*we_tran5D_dim.LW*we_tran5D_dim.cube);
        WeightTrans(filters, weight_dim, we_5D_dim, we_tran5D, we_tran5D_dim, CUBE_row, CUBE_col);

        // 输入tensor  4D转5D 在线软件完成(或者只在输入为4D时完成)
        float* in_5D = (float*)malloc(sizeof(float)*in_5D_dim.n*in_5D_dim.c1*in_5D_dim.h*in_5D_dim.w*in_5D_dim.c0);
        float* in_tran5D = (float*)malloc(sizeof(float)*in_tran5D_dim.batch*in_tran5D_dim.move*in_tran5D_dim.channel*in_tran5D_dim.LW*in_tran5D_dim.cube);
        Input4D25D(in_data, in_dim, pad, stride, CUBE_row, CUBE_col, in_5D_dim, in_5D);

        // 输入tensor  5D Im2Col(2D)+矩阵重排(大Z小Z)  在线硬件完成
        Input5D2Cube(in_5D, in_5D_dim, weight_dim, in_tran5D, in_tran5D_dim, out_dim, pad, stride, CUBE_row, CUBE_col);
        


        free(we_tran5D);
        free(in_tran5D);
    printf("  #  info : this is ascend method .\n");
};
