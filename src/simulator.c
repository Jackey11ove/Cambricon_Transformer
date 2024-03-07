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
  float ops = 0;
  float mem = 0;
  float alu_perf;
  float mem_perf;
  int alu_bottleneck; 
  float peak_perf;
  float worst_perf;

  //advanced usage
  //问题在于这样的评估方式是否合理，直接用1/(阻塞排数+1)来表示流水效率
  float vec_alu_pipe_eff = hardware->vec_pipeline == 1 ? 1 : 1/(hardware->vec_stall_cycle + 1);
  float mac_alu_pipe_eff = hardware->mac_pipeline == 1 ? 1 : 1.0/(hardware->mac_stall_cycle + 1);

  for(i = 0; i < net.n; ++i) {
    layer l = net.layers[i];
    if(l.type == CONVOLUTIONAL) {
      //ops
      ops += 2 * l.n * l.size * l.size * l.c * l.out_h * l.out_w;
      // filter_num * filter_size^2 * channels * out_h * out_w

      //mem
      mem += mac_dtype * l.w * l.h * l.c;
      mem += mac_dtype * l.size * l.size * l.c * l.n;
      mem += vec_dtype * l.n * l.out_h * l.out_w;

      //perf
      //完成一次conv的时间
      alu_perf = (((((ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == BATCHNORM) {
      //ops
      ops += l.w * l.h * l.c; //for mean
      ops += l.w * l.h * l.c * 4; //for var
      ops += l.w * l.h * l.c; //for scale
      ops += l.w * l.h * l.c * 2; //for bias

      //mem
      mem += l.w * l.h * l.c * 2 * vec_dtype;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == ACTIVE) {
      if(hardware->surpass_num > 0) {  //using surpass alu
        //ops
        ops += l.inputs;

        //mems
        mem += 2 * surpass_dtype * l.inputs;

        //perf
        alu_perf = (((((ops / hardware->surpass_num)) / hardware->freq) / (1000))) / (hardware->surpass_eff/100);
        mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
      } else { //using taylor expansion, 1/(1+e^(-x)) = 1/2 + (1/4)*x - (1/48)*x^3
        //ops
        ops += 3 * l.inputs + 2 * l.inputs + 4 * l.inputs;;

        //mems
        mem += 2 * vec_dtype * l.inputs;

        //perf
        alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
        mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
      }
    } else if(l.type == RELU) {
      //ops
      ops += l.inputs;

      //mems
      mem += 2 * vec_dtype * l.inputs;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / (hardware->ave_alu_eff/100);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == AVGPOOL) {
      //ops
      ops += 2 * l.size * l.size * l.c * l.out_h * l.out_w;

      //mem
      mem += vec_dtype * l.c * l.w * l.h;
      mem += vec_dtype * l.out_c * l.out_w * l.out_h;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == MAXPOOL){
      //ops
      ops += 2 * l.size * l.size * l.c * l.out_h * l.out_w;

      //mem
      mem += vec_dtype * l.c * l.w * l.h;
      mem += vec_dtype * l.out_c * l.out_w * l.out_h;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == CONNECTED) {
      //ops
      ops += 2 * l.inputs * l.outputs;

      //mem
      mem += mac_dtype * l.inputs;
      mem += mac_dtype * l.inputs * l.outputs;
      mem += vec_dtype * l.outputs;

      //perf
      alu_perf = (((((ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if (l.type == RNN) {
      //ops
      ops += 2 * l.input_layer->inputs * l.input_layer->outputs;
      ops += 2 * l.self_layer->inputs * l.self_layer->outputs;
      ops += 2 * l.output_layer->inputs * l.output_layer->outputs;
      ops *= l.n;

      //mem
      mem += mac_dtype * l.input_layer->inputs;
      mem += mac_dtype * l.input_layer->inputs * l.input_layer->outputs;
      mem += mac_dtype * l.self_layer->inputs * l.self_layer->outputs;
      mem += mac_dtype * l.output_layer->inputs * l.output_layer->outputs;
      mem += vec_dtype * l.input_layer->outputs;

      //perf
      alu_perf = (((((ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if (l.type == LSTM) {
      //ops
      ops += 2 * l.uf->inputs * l.uf->outputs;
      ops += 2 * l.ui->inputs * l.ui->outputs;
      ops += 2 * l.ug->inputs * l.ug->outputs;
      ops += 2 * l.uo->inputs * l.uo->outputs;
      ops += 2 * l.wf->inputs * l.wf->outputs;
      ops += 2 * l.wi->inputs * l.wi->outputs;
      ops += 2 * l.wg->inputs * l.wg->outputs;
      ops += 2 * l.wo->inputs * l.wo->outputs;

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
      alu_perf = (((((ops / hardware->mac_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * mac_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == LRN) {
      float x = 100/hardware->surpass_eff;
      if (hardware->surpass_num == 0) {  //taylor expansion, 1/x
        x = 10; //approximation
      }
      //ops
      ops += l.c * l.h * l.w * (2 * l.n * l.n * x + 2);

      //mem
      mem += 2 * vec_dtype * l.c * l.w * l.h;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == DECONV) {
      //ops
      ops += 2 * l.n * l.size * l.size * l.c * l.h * l.w;

      //mem
      mem += vec_dtype * l.w * l.h * l.c;
      mem += vec_dtype * l.size * l.size * l.c * l.n;
      mem += vec_dtype * l.n * l.out_h * l.out_w;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    } else if(l.type == UNPOOL) {
      //ops
      ops += l.size * l.size * l.c * l.out_h * l.out_w;

      //mem
      mem += vec_dtype * l.w * l.h * l.c;
      mem += vec_dtype * l.out_c * l.out_h * l.out_w;

      //perf
      alu_perf = (((((ops / hardware->vec_num)) / hardware->freq) / (1000))) / ((hardware->ave_alu_eff/100) * vec_alu_pipe_eff);
      mem_perf = (((mem / (1024 * 1024 * 1024)) / hardware->off_bw) * 1000 * 1000 ) / (hardware->ave_bw_eff/100);// + hardware->latency;
    }
  }

  alu_bottleneck = (alu_perf-mem_perf) > 0.0000001 ? 1 : 0;
  peak_perf = (alu_bottleneck  == 1) ? alu_perf : mem_perf;
  worst_perf  = alu_perf + mem_perf;

  printf("Total Compute Operations : %f GOPs\n", ops/(1000*1000*1000));
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
