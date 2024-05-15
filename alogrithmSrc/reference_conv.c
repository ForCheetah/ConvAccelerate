// @file reference_conv.c
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <stdint.h>
#include <stdio.h>

static inline uint32_t is_a_ge_zero_and_a_lt_b(int a, int b) {
  
  return (unsigned int) a < (unsigned int) b;
}

void RefConv2dF32(const float *input, const float *weight,
    const float *bias, const int in_n, const int in_c, const int in_h,
    const int in_w, const int out_c, const int out_h, const int out_w,
    const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {
    
  int imap_offset, omap_offset;
  for(int batch =0; batch<in_n; batch++){
    for (int g = 0; g < group; ++g) {
      imap_offset = g * (in_c / group);
      omap_offset = g * (out_c / group);  // 第几个group
        int s = 0;
        while (s < out_c / group) {  // 第几个核
          int in_row = -pad;
          for (int out_row = 0; out_row < out_h; ++out_row) {
            int in_col = -pad;
            for (int out_col = 0; out_col < out_w; ++out_col) {
              register float sum = 0.0;
              for (int imap = 0; imap < in_c / group; ++imap) {
                int in_addr_base = (imap_offset + imap) * in_h + in_row;
                int wt_addr_base = ((omap_offset + s) * in_c / group + imap);
                for (int kr = 0; kr < ker_size; ++kr) {

                  int wt_addr0 = (wt_addr_base * ker_size + kr) * ker_size;
                  int in_addr0 = (in_addr_base + kr) * in_w + in_col;
                  // printf("wt_addr0:%d in_addr0:%d  \n", wt_addr0, in_addr0);

                  for (int kc = 0; kc < ker_size; ++kc) {
                    if (is_a_ge_zero_and_a_lt_b(in_row + kr,  // 如果位于图外的话，不加就是加零（padding的原因）
                        in_h) & is_a_ge_zero_and_a_lt_b(in_col + kc, in_w)) {
                      int in_addr = batch*in_c*in_h*in_w + in_addr0 + kc;
                      int wt_addr = wt_addr0 + kc;
                      sum += weight[wt_addr] * input[in_addr];
                    }
                  }
                }
              }
              if (bias_en) {
                sum += bias[omap_offset + s];
              }
              int out_addr = ((batch*out_c + omap_offset + s) * out_h + out_row) * out_w + out_col;
              
              output[out_addr] = sum;
              // printf("  =%.1f %d ", output[out_addr], out_addr);
              in_col += stride;
            }
            in_row += stride;
          }
          s++;
      }
    }
  }
}


//  calculate  nhwc * out_c,in_c,h,w
void RefConv2dF32_nhwc(const float *input, const float *weight,
    const float *bias, const int in_n, const int in_c, const int in_h,
    const int in_w, const int out_c, const int out_h, const int out_w,
    const int ker_size, const int group,
    const int pad, const int stride, const int bias_en, float *output) {
    
  int in_addr, we_addr;
  for(int batch =0; batch<in_n; batch++){
          int in_row = -pad;
          for (int out_row = 0; out_row < out_h; ++out_row) {
            int in_col = -pad;
            for (int out_col = 0; out_col < out_w; ++out_col) {
              for(int out_channel=0; out_channel<out_c; out_channel++){
                
                float sum = 0.0;
                for(int k_h=0; k_h<ker_size; k_h++){
                  for(int k_w=0; k_w<ker_size; k_w++){
                    in_addr = batch*in_c*in_h*in_w + ((in_row+k_h)*in_w+in_col+k_w) * in_c; 
                    we_addr = out_channel*in_c*ker_size*ker_size;
                    for(int ch=0; ch<in_c; ch++){
                      if(is_a_ge_zero_and_a_lt_b(in_row + k_h, in_h) && is_a_ge_zero_and_a_lt_b(in_col + k_w, in_w)){
                        sum += input[in_addr+ch] * weight[we_addr+ch*ker_size*ker_size+k_h*ker_size+k_w];
                        // printf("sum = input[%d] * weight[%d] \n", in_addr+ch, we_addr+ch*ker_size*ker_size+k_h*ker_size+k_w);
                        
                      }
                    }
                  }


                }

                // if (bias_en) {
                //   sum += bias[omap_offset];
                // }
                int out_addr = batch*out_h*out_w*out_c + out_row*out_w*out_c + out_col*out_c + out_channel;
                output[out_addr] = sum;
                // printf("  =%.1f %d ", output[out_addr], out_addr);
              }
              in_col += stride;
            }
            in_row += stride;
          }
  }
}
