#include "simulator.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "parser.h"
#include "utils.h"

/*newly added function*/
// Calculate mac ops of matrix_a(a_h * a_w) * matrix_b(b_h * b_w)
struct mv_operations matmul_ops(int a_h, int a_w, int b_h, int b_w){
  struct mv_operations mv_ops;
  if(a_w != b_h){
    error("Matmul not appropriate!");
  }else{
    mv_ops.mac_ops = a_w * b_w * a_h;
    mv_ops.vec_ops = 0;
  }
  return mv_ops;
}

float matmul_mem(int a_h, int a_w, int b_h, int b_w, int mac_dtype, int vec_dtype){
  float mem = 0;
  mem += mac_dtype * (a_h * a_w + b_h * b_w);
  mem += vec_dtype * a_h * b_w;
  return mem;
}

struct mv_operations softmax_ops(int height, int width, int lut){
  struct mv_operations mv_ops;
  if(lut){
    // lut for e^x, then add all for norm(h*w), then value/sum (h*w)
    mv_ops.vec_ops = height * width;
    mv_ops.mac_ops = height * width;
  }else{
    //e^x = 1 + x + (1/2)x^2 (2 of mac operation)
    mv_ops.mac_ops = 3 * height * width;  // 2 for taylor and 1 for division
    mv_ops.vec_ops = height * width;  // add all elements in the matrix
  }
  return mv_ops;
}

float softmax_mem(int height, int width, int lut, int mac_dtype, int vec_dtype){
  float mem = 0;
  if(lut){
    mem += vec_dtype * height * width * 2; // Load matrix and search for lut
    mem += vec_dtype * height * width; // Store matrix
  }else{
    mem += vec_dtype * height * width * 2; // Load & Store
  }
  return mem;
}

struct mv_operations self_attention_ops(int height, int width, int d_model, int lut){
  struct mv_operations mv_ops;
  mv_ops.mac_ops = 0;
  mv_ops.vec_ops = 0;
  // transformation of input matrix to Q,K,V
  mv_ops.mac_ops += 3 * matmul_ops(height, width, width, d_model).mac_ops;

  // Q * K^T
  mv_ops.mac_ops += matmul_ops(height, d_model, d_model, height).mac_ops;

  // Scale(Q * K^T) : (Q * K^T)/(d_model^1/2)
  mv_ops.mac_ops += height * height;  //mac operation for division

  // Softmax( Scale(Q * K^T) )
  mv_ops.mac_ops += softmax_ops(height, height, lut).mac_ops;
  mv_ops.vec_ops += softmax_ops(height, height, lut).vec_ops;

  // Softmax( Scale(Q * K^T) ) * V
  mv_ops.mac_ops += matmul_ops(height, height, height, d_model).mac_ops;

  return mv_ops;
}

float self_attention_mem(int height, int width, int d_model, int lut, int mac_dtype, int vec_dtype){
  float mem = 0;
  // transformation of input matrix to Q,K,V
  mem += 3 * matmul_mem(height, width, width, d_model, mac_dtype, vec_dtype);

  // Q * K^T
  mem += matmul_mem(height, d_model, d_model, height, mac_dtype, vec_dtype);

  // Scale(Q * K^T) : (Q * K^T)/(d_model^1/2)
  mem += (mac_dtype + vec_dtype) * height * height;  //mac operation for division, load and store

  // Softmax( Scale(Q * K^T) )
  mem += softmax_mem(height, height, lut, mac_dtype, vec_dtype);

  // Softmax( Scale(Q * K^T) ) * V
  mem += matmul_mem(height, height, height, d_model, mac_dtype, vec_dtype);

  return mem;
}

struct mv_operations multihead_attention_ops(int height, int width, int d_model, int lut, int N_head){
  struct mv_operations mv_ops;
  mv_ops.mac_ops = 0;
  mv_ops.vec_ops = 0;
  // N_head of self_attention
  mv_ops.mac_ops += N_head * self_attention_ops(height, width, d_model, lut).mac_ops;
  mv_ops.vec_ops += N_head * self_attention_ops(height, width, d_model, lut).vec_ops;

  // Linear transformation for concated matrix
  mv_ops.mac_ops += matmul_ops(height, d_model*N_head, d_model*N_head, width).mac_ops;

  return mv_ops;
}

float multihead_attention_mem(int height, int width, int d_model, int lut, int N_head, int mac_dtype, int vec_dtype){
  float mem = 0;
  // N_head of self_attention
  mem += N_head * self_attention_mem(height, width, d_model, lut, mac_dtype, vec_dtype);

  // Linear transformation for concated matrix
  mem += matmul_mem(height, d_model*N_head, d_model*N_head, width, mac_dtype, vec_dtype);

  return mem;
}

struct mv_operations feed_forward_ops(int height, int width, int d_ff){
  struct mv_operations mv_ops;
  mv_ops.mac_ops = 0;
  mv_ops.vec_ops = 0;
  // First full connected ( XW1 + b1 )
  mv_ops.mac_ops += matmul_ops(height, width, width, d_ff).mac_ops;  // weight
  mv_ops.vec_ops += height * d_ff;  //bias

  // Relu ( Relu(XW1 + b1) )
  mv_ops.vec_ops += height * d_ff;

  // Second full connected( Relu(XW1 + b1)W2 + b2 )
  mv_ops.mac_ops += matmul_ops(height, d_ff, d_ff, width).mac_ops;
  mv_ops.vec_ops += height * width;

  return mv_ops;
}

float feed_forward_mem(int height, int width, int d_ff, int mac_dtype, int vec_dtype){
  float mem = 0;
  // First full connected ( XW1 + b1 )
  mem += matmul_mem(height, width, width, d_ff, mac_dtype, vec_dtype);  //weight
  mem += 3 * vec_dtype * height * d_ff;  //bias(2 for loading two matrix, 1 for store result)

  // Relu ( Relu(XW1 + b1) )
  mem += 2 * vec_dtype * height * d_ff;

  // Second full connected( Relu(XW1 + b1)W2 + b2 )
  mem += matmul_mem(height, d_ff, d_ff, width, mac_dtype, vec_dtype);
  mem += 3 * vec_dtype * height * width;

  return mem;
}

struct mv_operations layer_norm_ops(int height, int width){
  struct mv_operations mv_ops;
  mv_ops.mac_ops = 0;
  mv_ops.vec_ops = 0;
  // Mean
  mv_ops.vec_ops += height * width;
  
  // Var
  mv_ops.vec_ops += height * width;
  mv_ops.mac_ops += height * width;
  
  // Division
  mv_ops.mac_ops += height * width;

  // Scale and bias
  mv_ops.mac_ops += height * width;

  return mv_ops;
}

float layer_norm_mem(int height, int width, int mac_dtype, int vec_dtype){
  float mem = 0;
  // Mean
  mem += vec_dtype * height * width * 2;
  
  // Var
  mem += mac_dtype * height * width;
  
  // Division
  mem += vec_dtype * height * width;

  // Scale and bias
  mem += mac_dtype * height * width;
  mem += vec_dtype * height * width;
  
  return mem;
}

void operations(char *asicfile, char *cfgfile) {
  asic *hardware = (asic*)xmalloc(sizeof(asic));
  parse_hardware_cfg(asicfile, hardware);
  int mac_dtype;
  int vec_dtype;
  int surpass_dtype;
  printf("\n===========processor info=================\n");
  printf("Tensor Alu Number            : %d\n", hardware->mac_num);
  if(hardware->mac_dtype == 1) {
    printf("Tensor Alu Dtype             : half\n");
    mac_dtype = 2;
  } else if(hardware->mac_dtype == 2) {
    printf("Tensor Alu Dtype             : float\n");
    mac_dtype = 4;
  }
  if(hardware->mac_pipeline == 1) {
    printf("Tensor Alu Is Full Pipeline  : yes\n");
  } else {
    printf("Tensor Alu Is Full Pipeline  : no\n");
    printf("Tensor Alu Stall Cycle       : %d\n",hardware->mac_stall_cycle);
  }
  printf("Vector Alu Number            : %d\n", hardware->vec_num);
  if(hardware->vec_dtype == 1) {
    printf("Vector Alu Dtype             : half\n");
    vec_dtype = 2;
  } else if(hardware->vec_dtype == 2) {
    printf("Vector Alu Dtype             : float\n");
    vec_dtype = 4;
  }
  if(hardware->vec_pipeline == 1) {
    printf("Vector Alu Is Full Pipeline  : yes\n");
  } else {
    printf("Vector Alu Is Full Pipeline  : no\n");
    printf("Vector Alu Stall Cycle       : %d\n",hardware->vec_stall_cycle);
  }
  if(hardware->surpass_num > 0) {
    printf("Surpass Alu Number           : %d\n", hardware->surpass_num);
    if(hardware->surpass_dtype == 1) {
      printf("Surpass Alu Dtype            : half\n");
      surpass_dtype = 2;
    } else if(hardware->surpass_dtype == 2) {
      printf("Surpass Alu Dtype            : float\n");
      surpass_dtype = 4;
    }
  } else {
    printf("Surpass Alu Supported        : no\n");
  }
  printf("Power                        : %.5f W\n", hardware->pwr);
  printf("Area                         : %.5f mm^2\n", hardware->area);
  printf("Offchip Bandwidth            : %.5f GB/s\n", hardware->off_bw);
  printf("Offchip Latency              : %.5f us\n", hardware->latency);
  printf("Frequency                    : %.5f GHz\n", hardware->freq);
  printf("Average Alu Efficiency       : %.5f%%\n", hardware->ave_alu_eff);
  printf("Average Bandwidth Efficiency : %.5f%%\n", hardware->ave_bw_eff);
  if(hardware->surpass_num > 0) {
    printf("Surpass Efficiency           : %.5f%%\n", hardware->surpass_eff);
  }
  printf("===========processor info=================\n");
  printf("\n\n===========operator info==================\n");

  network net = parse_network_cfg(cfgfile);
  int i;
  float mem = 0;
  float alu_perf;
  float mem_perf;
  int alu_bottleneck; 
  float peak_perf;
  float worst_perf;
  struct mv_operations mv_ops;
  mv_ops.mac_ops = 0;
  mv_ops.vec_ops = 0;

  //advanced usage
  float vec_alu_pipe_eff = hardware->vec_pipeline == 1 ? 1 : 1/(hardware->vec_stall_cycle + 1);
  float mac_alu_pipe_eff = hardware->mac_pipeline == 1 ? 1 : 1.0/(hardware->mac_stall_cycle + 1);

  for(i = 0; i < net.n; ++i) {
    layer l = net.layers[i];
    if(l.type == CONVOLUTIONAL) {
      //ops
      mv_ops.mac_ops += 2 * l.n * l.size * l.size * l.c * l.out_h * l.out_w;
      // filter_num * filter_size^2 * channels * out_h * out_w

      //mem
      mem += mac_dtype * l.w * l.h * l.c;
      mem += mac_dtype * l.size * l.size * l.c * l.n;
      mem += vec_dtype * l.n * l.out_h * l.out_w;

      //perf
      //完成一次conv的时间
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == BATCHNORM) {
      //ops
      mv_ops.vec_ops += l.w * l.h * l.c; //for mean
      mv_ops.vec_ops += l.w * l.h * l.c * 4; //for var
      mv_ops.vec_ops += l.w * l.h * l.c; //for scale
      mv_ops.vec_ops += l.w * l.h * l.c * 2; //for bias

      //mem
      mem += l.w * l.h * l.c * 2 * vec_dtype;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == ACTIVE) {
      if(hardware->surpass_num > 0) {  //using surpass alu
        //ops
        mv_ops.vec_ops += l.inputs;

        //mems
        mem += 2 * surpass_dtype * l.inputs;

        //perf
        alu_perf = (((((mv_ops.vec_ops / hardware->surpass_num)) / hardware->freq) / (1000))) / (hardware->surpass_eff/100);
        mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
      } else { //using taylor expansion, 1/(1+e^(-x)) = 1/2 + (1/4)*x - (1/48)*x^3
        //ops
        mv_ops.vec_ops += 3 * l.inputs + 2 * l.inputs + 4 * l.inputs;;

        //mems
        mem += 2 * vec_dtype * l.inputs;

        //perf
        alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
        mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
      }
    } else if(l.type == RELU) {
      //ops
      mv_ops.vec_ops += l.inputs;

      //mems
      mem += 2 * vec_dtype * l.inputs;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / (hardware->ave_alu_eff/100);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == AVGPOOL) {
      //ops
      mv_ops.vec_ops += 2 * l.size * l.size * l.c * l.out_h * l.out_w;

      //mem
      mem += vec_dtype * l.c * l.w * l.h;
      mem += vec_dtype * l.out_c * l.out_w * l.out_h;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == MAXPOOL){
      //ops
      mv_ops.vec_ops += 2 * l.size * l.size * l.c * l.out_h * l.out_w;

      //mem
      mem += vec_dtype * l.c * l.w * l.h;
      mem += vec_dtype * l.out_c * l.out_w * l.out_h;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == CONNECTED) {
      //ops
      mv_ops.mac_ops += 2 * l.inputs * l.outputs;

      //mem
      mem += mac_dtype * l.inputs;
      mem += mac_dtype * l.inputs * l.outputs;
      mem += vec_dtype * l.outputs;

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if (l.type == RNN) {
      //ops
      mv_ops.mac_ops += 2 * l.input_layer->inputs * l.input_layer->outputs;
      mv_ops.mac_ops += 2 * l.self_layer->inputs * l.self_layer->outputs;
      mv_ops.mac_ops += 2 * l.output_layer->inputs * l.output_layer->outputs;
      mv_ops.mac_ops *= l.n;

      //mem
      mem += mac_dtype * l.input_layer->inputs;
      mem += mac_dtype * l.input_layer->inputs * l.input_layer->outputs;
      mem += mac_dtype * l.self_layer->inputs * l.self_layer->outputs;
      mem += mac_dtype * l.output_layer->inputs * l.output_layer->outputs;
      mem += vec_dtype * l.input_layer->outputs;

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if (l.type == LSTM) {
      //ops
      mv_ops.mac_ops += 2 * l.uf->inputs * l.uf->outputs;
      mv_ops.mac_ops += 2 * l.ui->inputs * l.ui->outputs;
      mv_ops.mac_ops += 2 * l.ug->inputs * l.ug->outputs;
      mv_ops.mac_ops += 2 * l.uo->inputs * l.uo->outputs;
      mv_ops.mac_ops += 2 * l.wf->inputs * l.wf->outputs;
      mv_ops.mac_ops += 2 * l.wi->inputs * l.wi->outputs;
      mv_ops.mac_ops += 2 * l.wg->inputs * l.wg->outputs;
      mv_ops.mac_ops += 2 * l.wo->inputs * l.wo->outputs;

      //mem
      mem += mac_dtype * l.uf->inputs;
      mem += mac_dtype * l.uf->inputs * l.uf->outputs;
      mem += mac_dtype * l.ui->inputs * l.ui->outputs;
      mem += mac_dtype * l.ug->inputs * l.ug->outputs;
      mem += mac_dtype * l.uo->inputs * l.uo->outputs;
      mem += mac_dtype * l.wf->inputs * l.wf->outputs;
      mem += mac_dtype * l.wi->inputs * l.wi->outputs;
      mem += mac_dtype * l.wg->inputs * l.wg->outputs;
      mem += vec_dtype * l.wo->outputs;

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == LRN) {
      float x = 100/hardware->surpass_eff;
      if (hardware->surpass_num == 0) {  //taylor expansion, 1/x
        x = 10; //approximation
      }
      //ops
      mv_ops.vec_ops += l.c * l.h * l.w * (2 * l.n * l.n * x + 2);

      //mem
      mem += 2 * vec_dtype * l.c * l.w * l.h;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == DECONV) {
      //ops
      mv_ops.vec_ops += 2 * l.n * l.size * l.size * l.c * l.h * l.w;

      //mem
      mem += vec_dtype * l.w * l.h * l.c;
      mem += vec_dtype * l.size * l.size * l.c * l.n;
      mem += vec_dtype * l.n * l.out_h * l.out_w;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == UNPOOL) {
      //ops
      mv_ops.vec_ops += l.size * l.size * l.c * l.out_h * l.out_w;

      //mem
      mem += vec_dtype * l.w * l.h * l.c;
      mem += vec_dtype * l.out_c * l.out_h * l.out_w;

      //perf
      alu_perf = (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    }else if(l.type == SOFTMAX){
      //ops
      mv_ops.mac_ops += softmax_ops(l.h, l.w, l.lut).mac_ops;
      mv_ops.vec_ops += softmax_ops(l.h, l.w, l.lut).vec_ops;

      //mem
      mem += softmax_mem(l.h, l.w, l.lut, mac_dtype, vec_dtype);

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff) \
               + (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    }else if (l.type == SELF_ATTENTION){
      //ops
      mv_ops.mac_ops += self_attention_ops(l.h, l.w, l.d_model, l.lut).mac_ops;
      mv_ops.vec_ops += self_attention_ops(l.h, l.w, l.d_model, l.lut).vec_ops;

      //mem
      mem += self_attention_mem(l.h, l.w, l.d_model, l.lut, mac_dtype, vec_dtype);

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff) \
               + (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    }else if (l.type == MULTIHEAD_ATTENTION){
      //ops
      mv_ops.mac_ops += multihead_attention_ops(l.h, l.w, l.d_model, l.lut, l.N_head).mac_ops;
      mv_ops.vec_ops += multihead_attention_ops(l.h, l.w, l.d_model, l.lut, l.N_head).vec_ops;

      //mem
      mem += multihead_attention_mem(l.h, l.w, l.d_model, l.lut, l.N_head, mac_dtype, vec_dtype);

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff) \
               + (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    }else if(l.type == FEED_FORWARD){
      //ops
      mv_ops.mac_ops += feed_forward_ops(l.h, l.w, l.d_ff).mac_ops;
      mv_ops.vec_ops += feed_forward_ops(l.h, l.w, l.d_ff).vec_ops;

      //mem
      mem += feed_forward_mem(l.h, l.w, l.d_ff, mac_dtype, vec_dtype);

      //perf
      alu_perf = (((((mv_ops.mac_ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff) \
               + (((((mv_ops.vec_ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    }    
    
  }

  alu_bottleneck = (alu_perf-mem_perf) > 0.0000001 ? 1 : 0;
  peak_perf = (alu_bottleneck  == 1) ? alu_perf : mem_perf;
  worst_perf  = alu_perf + mem_perf;

  printf("Total Compute Operations : %f GOPs\n", (mv_ops.mac_ops + mv_ops.vec_ops)/(1000*1000*1000));
  printf("Total Data Sizes         : %f MB\n", mem/(1024*1024));
  printf("===========operator info==================\n\n\n");
  printf("===========performance====================\n");
  printf("Peak Performance       : %.5f us\n", peak_perf);
  printf("Worst Performance      : %.5f us\n", worst_perf);
  printf("Peak Power Efficiency  : %.5f\n", 1/(hardware->pwr*peak_perf));
  printf("Worst Power Efficiency : %.5f\n", 1/(hardware->pwr*worst_perf));
  printf("Peak Area Efficiency   : %.5f\n", 1/(hardware->area*peak_perf));
  printf("Worst Area Efficiency  : %.5f\n", 1/(hardware->area*worst_perf));
  if(alu_bottleneck == 1) {
    printf("\nBottleneck is COMPUTATION!\n");
  } else {
    printf("\nBottleneck is MEMORY ACCESS!\n");
  }
  printf("===========performance====================\n\n\n");

  free(hardware);
}


int main(int argc, char **argv) {
  int i;
  for (i = 0; i < argc; ++i) {
    if (!argv[i]) continue;
    strip_args(argv[i]);
  }

  operations(argv[1],argv[2]);

  return 0;
}
