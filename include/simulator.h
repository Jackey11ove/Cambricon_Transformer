#ifndef SIMULATOR_API
#define SIMULATOR_API

#if defined(_MSC_VER) && _MSC_VER < 1900
  #define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
  #define _CRTDBG_MAP_ALLOC
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#ifndef LIB_API
  #ifdef LIB_EXPORTS
    #if defined(_MSC_VER)
      #define LIB_API __declspec(dllexport)
    #else
      #define LIB_API __attribute__((visibility("default")))
    #endif
  #else
    #if defined(_MSC_VER)
      #define LIB_API
    #else
      #define LIB_API
    #endif
  #endif
#endif

#define SECRET_NUM -1234

typedef enum { UNUSED_DEF_VAL } UNUSED_ENUM_TYPE;

#ifdef __cplusplus
  extern "C" {
#endif

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct asic;
typedef struct asic asic;

// layer.h
typedef enum {
    CONVOLUTIONAL,
    DECONV,
    CONNECTED,
    MAXPOOL,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    ACTIVE,
    RNN,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    RELU,
    LRN,
    UNPOOL,
    EMPTY,
    BLANK
} LAYER_TYPE;

// layer.h
struct layer {
    LAYER_TYPE type;
    layer *share_layer;
    int inputs;
    int hidden;
    int outputs;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int group_id;
    int size;
    int side;
    int stride;
    int stride_x;
    int stride_y;
    int dilation;
    int antialiasing;
    int maxpool_depth;
    int out_channels;
    int peephole;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

};


// network.h
typedef struct network {
    int n;
    int h, w, c;
    int inputs;
    int time_steps;
    layer *layers;
} network;


// hardware.h
typedef struct asic {
    int mac_num;         //tensor alu number
    int mac_dtype;       //tensor data type
    int mac_pipeline;    //1 means no stall, 0 means have stalls
    int mac_stall_cycle; //when full_pipeline is 0, this is the number of stall cycles of each cycle
    int vec_num;         //vector alu number
    int vec_dtype;       //vector data type
    int vec_pipeline;    //1 means no stall, 0 means have stalls
    int vec_stall_cycle; //when full_pipeline is 0, this is the number of stall cycles of each cycle
    int surpass_num;     //surpass alu number
    int surpass_dtype;   //surpass alu data type
    float pwr;           //power(in W)
    float area;          //area(in mm^2)
    float off_bw;        //total bandwidth with DDR(in GB/s)
    float freq;          //frequency
    float ave_alu_eff;   //average alu efficiency(in %)
    float ave_bw_eff;    //average bandwidth efficiency(in %)
    float surpass_eff;   //surpass alu efficiency
    float latency;       //offchip latency (in us)
} asic;



// -----------------------------------------------------


// parser.c
LIB_API void free_network(network net);

// network.h
LIB_API layer* get_network_layer(network* net, int i);

// utils.h
LIB_API void free_ptrs(void **ptrs, int n);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // SIMULATOR_API
