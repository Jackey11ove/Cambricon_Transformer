#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "option.h"
#include "parser.h"
#include "utils.h"
#include "network.h"

typedef struct{
    char *type;
    list *options;
}section;


list *read_cfg(char *filename);

// 将对应的算子字符串转为枚举类别
LAYER_TYPE string_to_layer_type(char * type) {

    if (strcmp(type, "[conv]")==0
        || strcmp(type, "[convolutional]")==0)   return CONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0)         return ACTIVE;
    if (strcmp(type, "[net]")==0
        || strcmp(type, "[network]")==0)         return NETWORK;
    if (strcmp(type, "[lstm]")==0)               return LSTM;
    if (strcmp(type, "[rnn]")==0)                return RNN;
    if (strcmp(type, "[conn]")==0
        || strcmp(type, "[connected]")==0)       return CONNECTED;
    if (strcmp(type, "[max]")==0
        || strcmp(type, "[maxpool]")==0)         return MAXPOOL;
    if (strcmp(type, "[avg]")==0
        || strcmp(type, "[avgpool]")==0)         return AVGPOOL;
    if (strcmp(type, "[lrn]")==0)                return LRN;
    if (strcmp(type, "[batchnorm]")==0)          return BATCHNORM;
    if (strcmp(type, "[relu]")==0)               return RELU;
    if (strcmp(type, "[deconvolutional]")==0)    return DECONV;
    if (strcmp(type, "[unpool]") == 0)           return UNPOOL;
    if (strcmp(type, "[softmax]")==0 )           return SOFTMAX;
    if (strcmp(type, "[self_attention]")==0 )    return SELF_ATTENTION;
    if (strcmp(type, "[multihead_attention]")==0 ) return MULTIHEAD_ATTENTION;
    if (strcmp(type, "[empty]") == 0)            return EMPTY;
    return BLANK;
}

// 释放section结构体占据的内存
void free_section(section *s) {
  free(s->type);
  node *n = s->options->front;
  while(n){
    kvp *pair = (kvp *)n->val;
    free(pair->key);
    free(pair);
    node *next = n->next;
    free(n);
    n = next;
  }
  free(s->options);
  free(s);
}

//@conv
layer parse_convolutional(list *options, size_params params) {
  layer l;

  int n = option_find_int(options, "filters",1);
  int size = option_find_int(options, "size",1);
  int stride = -1;
  int stride_x = option_find_int_quiet(options, "stride_x", -1);
  int stride_y = option_find_int_quiet(options, "stride_y", -1);
  if (stride_x < 1 || stride_y < 1) {
    stride = option_find_int(options, "stride", 1);
    if (stride_x < 1) stride_x = stride;
    if (stride_y < 1) stride_y = stride;
  }
  else {
    stride = option_find_int_quiet(options, "stride", 1);
  }
  l.antialiasing = option_find_int_quiet(options, "antialiasing", 0);

  int share_index = option_find_int_quiet(options, "share_index", -1000000000);
  layer *share_layer = NULL;
  if(share_index >= 0) share_layer = &params.net.layers[share_index];
  else if(share_index != -1000000000) share_layer = &params.net.layers[params.index + share_index];

  l.type = CONVOLUTIONAL;
  l.h = params.h;
  l.w = params.w;
  l.c = params.c;
  l.size = size;
  l.out_h = (params.h - size) / stride_y + 1;
  l.out_w = (params.w - size) / stride_x + 1;
  l.n = n;
  l.stride_x = stride_x;
  l.stride_y = stride_y;

  return l;
}


//@rnn
layer parse_rnn(list *options, size_params params) {
  layer l;

  int output = option_find_int(options, "output",1);
  int hidden = option_find_int(options, "hidden",1);

  int inputs = params.inputs;
  l.type = RNN;
  l.inputs = inputs;
  l.hidden = hidden;
  l.out_w = 1;
  l.out_h = 1;
  l.out_c = output;
  l.n = params.time_steps;

  l.input_layer = (layer*)xcalloc(1, sizeof(layer));
  l.input_layer->type = CONNECTED;
  l.input_layer->inputs = inputs;
  l.input_layer->outputs = hidden;
  l.input_layer->h = 1;
  l.input_layer->w = 1;
  l.input_layer->c = inputs;
  l.input_layer->out_h = 1;
  l.input_layer->out_w = 1;
  l.input_layer->out_c = hidden;
  l.input_layer->n = hidden;
  l.input_layer->size = 1;
  l.input_layer->stride_x = 1;
  l.input_layer->stride_y = 1;

  l.self_layer = (layer*)xcalloc(1, sizeof(layer));
  l.self_layer->type = CONNECTED;
  l.self_layer->inputs = hidden;
  l.self_layer->outputs = hidden;
  l.self_layer->h = 1;
  l.self_layer->w = 1;
  l.self_layer->c = hidden;
  l.self_layer->out_h = 1;
  l.self_layer->out_w = 1;
  l.self_layer->out_c = hidden;
  l.self_layer->n = hidden;
  l.self_layer->size = 1;
  l.self_layer->stride_x = 1;
  l.self_layer->stride_y = 1;

  l.output_layer = (layer*)xcalloc(1, sizeof(layer));
  l.output_layer->type = CONNECTED;
  l.output_layer->inputs = hidden;
  l.output_layer->outputs = output;
  l.output_layer->h = 1;
  l.output_layer->w = 1;
  l.output_layer->c = hidden;
  l.output_layer->out_h = 1;
  l.output_layer->out_w = 1;
  l.output_layer->out_c = output;
  l.output_layer->n = output;
  l.output_layer->size = 1;
  l.output_layer->stride_x = 1;
  l.output_layer->stride_y = 1;

  l.outputs = output;
  return l;
}

//@lstm
layer parse_lstm(list *options, size_params params) {
    int output = option_find_int(options, "output",1);

    layer l;

    l.type = LSTM;
    l.inputs = params.inputs;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = output;
    l.outputs = output;

    l.uf = (layer*)xcalloc(1, sizeof(layer));
    l.uf->type = CONNECTED;
    l.uf->inputs = params.inputs;
    l.uf->outputs = output;
    l.uf->h = 1;
    l.uf->w = 1;
    l.uf->c = params.inputs;
    l.uf->out_h = 1;
    l.uf->out_w = 1;
    l.uf->out_c = output;
    l.uf->n = output;
    l.uf->size = 1;
    l.uf->stride_x = 1;
    l.uf->stride_y = 1;

    l.ui = (layer*)xcalloc(1, sizeof(layer));
    l.ui->type = CONNECTED;
    l.ui->inputs = params.inputs;
    l.ui->outputs = output;
    l.ui->h = 1;
    l.ui->w = 1;
    l.ui->c = params.inputs;
    l.ui->out_h = 1;
    l.ui->out_w = 1;
    l.ui->out_c = output;
    l.ui->n = output;
    l.ui->size = 1;
    l.ui->stride_x = 1;
    l.ui->stride_y = 1;

    l.ug = (layer*)xcalloc(1, sizeof(layer));
    l.ug->type = CONNECTED;
    l.ug->inputs = params.inputs;
    l.ug->outputs = output;
    l.ug->h = 1;
    l.ug->w = 1;
    l.ug->c = params.inputs;
    l.ug->out_h = 1;
    l.ug->out_w = 1;
    l.ug->out_c = output;
    l.ug->n = output;
    l.ug->size = 1;
    l.ug->stride_x = 1;
    l.ug->stride_y = 1;

    l.uo = (layer*)xcalloc(1, sizeof(layer));
    l.uo->type = CONNECTED;
    l.uo->inputs = params.inputs;
    l.uo->outputs = output;
    l.uo->h = 1;
    l.uo->w = 1;
    l.uo->c = params.inputs;
    l.uo->out_h = 1;
    l.uo->out_w = 1;
    l.uo->out_c = output;
    l.uo->n = output;
    l.uo->size = 1;
    l.uo->stride_x = 1;
    l.uo->stride_y = 1;

    l.wf = (layer*)xcalloc(1, sizeof(layer));
    l.wf->type = CONNECTED;
    l.wf->inputs = output;
    l.wf->outputs = output;
    l.wf->h = 1;
    l.wf->w = 1;
    l.wf->c = output;
    l.wf->out_h = 1;
    l.wf->out_w = 1;
    l.wf->out_c = output;
    l.wf->n = output;
    l.wf->size = 1;
    l.wf->stride_x = 1;
    l.wf->stride_y = 1;

    l.wi = (layer*)xcalloc(1, sizeof(layer));
    l.wi->type = CONNECTED;
    l.wi->inputs = output;
    l.wi->outputs = output;
    l.wi->h = 1;
    l.wi->w = 1;
    l.wi->c = output;
    l.wi->out_h = 1;
    l.wi->out_w = 1;
    l.wi->out_c = output;
    l.wi->n = output;
    l.wi->size = 1;
    l.wi->stride_x = 1;
    l.wi->stride_y = 1;

    l.wg = (layer*)xcalloc(1, sizeof(layer));
    l.wg->type = CONNECTED;
    l.wg->inputs = output;
    l.wg->outputs = output;
    l.wg->h = 1;
    l.wg->w = 1;
    l.wg->c = output;
    l.wg->out_h = 1;
    l.wg->out_w = 1;
    l.wg->out_c = output;
    l.wg->n = output;
    l.wg->size = 1;
    l.wg->stride_x = 1;
    l.wg->stride_y = 1;

    l.wo = (layer*)xcalloc(1, sizeof(layer));
    l.wo->type = CONNECTED;
    l.wo->inputs = output;
    l.wo->outputs = output;
    l.wo->h = 1;
    l.wo->w = 1;
    l.wo->c = output;
    l.wo->out_h = 1;
    l.wo->out_w = 1;
    l.wo->out_c = output;
    l.wo->n = output;
    l.wo->size = 1;
    l.wo->stride_x = 1;
    l.wo->stride_y = 1;

    return l;
}


//@fc
layer parse_connected(list *options, size_params params) {
    int output = option_find_int(options, "output",1);
    layer l;

    l.type = CONNECTED;

    l.inputs = params.inputs;
    l.outputs = output;
    l.h = 1;
    l.w = 1;
    l.c = params.inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = output;
    l.n = l.out_c;
    l.size = 1;
    l.stride_x = l.stride_y = 1;

    return l;
}


//@bn
layer parse_batchnorm(list *options, size_params params) {
    layer l;

    l.type = BATCHNORM;
    l.h = params.h;
    l.w = params.w;
    l.c = params.c;

    return l;
}


//@active, only sigmoid
layer parse_activation(list *options, size_params params) {
    layer l;

    l.type = ACTIVE;

    l.inputs = params.inputs;
    l.outputs = l.inputs;

    return l;
}


//@relu
layer parse_relu(list *options, size_params params) {
    layer l;

    l.type = RELU;

    l.inputs = params.inputs;
    l.outputs = l.inputs;

    return l;
}

//@maxpool
layer parse_maxpool(list *options, size_params params) {
  layer l;

  l.type = MAXPOOL;

  l.stride = option_find_int_quiet(options, "stride", 1);
  l.size = option_find_int(options, "size",1);

  l.h = params.h;
  l.w = params.w;
  l.c = params.c;

  l.out_h = (params.h - l.size) / l.stride + 1;
  l.out_w = (params.w - l.size) / l.stride + 1;
  l.out_c = params.c;
 
  return l;
}

//@avgpool
layer parse_avgpool(list *options, size_params params) {
  layer l;

  l.type = AVGPOOL;

  l.stride = option_find_int_quiet(options, "stride", 1);
  l.size = option_find_int(options, "size",1);

  l.h = params.h;
  l.w = params.w;
  l.c = params.c;

  l.out_h = (params.h - l.size) / l.stride + 1;
  l.out_w = (params.w - l.size) / l.stride + 1;
  l.out_c = params.c;
 
  return l;
}


//@lrn
layer parse_lrn(list *options, size_params params) {
  layer l;
  l.type = LRN;

  l.n = option_find_int_quiet(options, "n", 1);

  l.h = params.h;
  l.w = params.w;
  l.c = params.c;

  return l;
}


//@deconv
layer parse_deconv(list *options, size_params params) {
  layer l;

  l.type = DECONV;

  l.stride = option_find_int_quiet(options, "stride", 1);
  l.size = option_find_int(options, "size",1);
  l.c = option_find_int(options, "filters",1);

  l.out_h = params.h;
  l.out_w = params.w;
  l.out_c = params.c;
  l.n = l.c;

  l.h = (params.h - l.size) / l.stride + 1;
  l.w = (params.w - l.size) / l.stride + 1;
 
  return l;
}


//@unpool
layer parse_unpool(list *options, size_params params) {
  layer l;

  l.type = UNPOOL;

  l.stride = option_find_int_quiet(options, "stride", 1);
  l.size = option_find_int(options, "size",1);

  l.out_h = params.h;
  l.out_w = params.w;
  l.out_c = params.c;
  l.c = l.out_c;

  l.h = (params.h - l.size) / l.stride + 1;
  l.w = (params.w - l.size) / l.stride + 1;
 
  return l;
}


//@softmax
layer parse_softmax(list *options, size_params params){
  layer l;

  l.type = SOFTMAX;
  l.h = params.h;
  l.w = params.w;
  l.lut = option_find_int(options, "lut", 0);

  return l;
}

//@self_attention
layer parse_self_attention(list *options, size_params params){
  layer l;

  l.type = SELF_ATTENTION;
  l.h = params.h;
  l.w = params.w;
  l.lut = option_find_int(options, "lut", 0);
  l.d_model = option_find_int(options, "d_model", 1);

  l.out_h = params.h;
  l.out_w = l.d_model;

  return l;
}

//@multihead_attention
layer parse_multihead_attention(list *options, size_params params){
  layer l;

  l.type = MULTIHEAD_ATTENTION;
  l.h = params.h;
  l.w = params.w;
  l.lut = option_find_int(options, "lut", 0);
  l.d_model = option_find_int(options, "d_model", 1);
  l.N_head = option_find_int(options, "N_head", 8);

  l.out_h = params.h;
  l.out_w = params.w;

  return l;
}



//=============================================================
void parse_net_options(list *options, network *net) {
  net->h = option_find_int_quiet(options, "height",0);
  net->w = option_find_int_quiet(options, "width",0);
  net->c = option_find_int_quiet(options, "channels",0);
  net->inputs = option_find_int_quiet(options, "inputs",0);
  net->time_steps= option_find_int_quiet(options, "time_steps",0);
}

network parse_network_cfg(char *filename) {
  // 这里读取的文件应该是模型的配置文件
  list *sections = read_cfg(filename);
  node *n = sections->front;
  if(!n) error("Config file has no sections");
  // network包含了网络的参数，以及net中各个层的参数，net->n表示该网络有多少层
  network net = make_network(sections->size - 1);
  size_params params;

  section *s = (section *)n->val;
  list *options = s->options;
  parse_net_options(options, &net);

  params.h = net.h;
  params.w = net.w;
  params.c = net.c;
  params.inputs = net.inputs;
  params.time_steps= net.time_steps;

  int avg_outputs = 0;
  int avg_counter = 0;
  size_t max_inputs = 0;
  size_t max_outputs = 0;

  n = n->next;
  int count = 0;
  free_section(s);
  while(n){
    params.index = count;
    s = (section *)n->val;
    options = s->options;
    layer l = { (LAYER_TYPE)0 };
    LAYER_TYPE lt = string_to_layer_type(s->type);
    if(lt == CONVOLUTIONAL){
        l = parse_convolutional(options, params);
    }else if(lt == RNN){
      l = parse_rnn(options, params);
    }else if(lt == LSTM){
      l = parse_lstm(options, params);
    }else if(lt == CONNECTED){
      l = parse_connected(options, params);
    }else if(lt == BATCHNORM){
      l = parse_batchnorm(options, params);
    }else if(lt == ACTIVE){
      l = parse_activation(options, params);
    }else if(lt == MAXPOOL){
      l = parse_maxpool(options, params);
    }else if(lt == AVGPOOL){
      l = parse_avgpool(options, params);
    }else if(lt == RELU){
      l = parse_relu(options, params);
    }else if (lt == LRN) {
      l = parse_lrn(options, params);
    }else if (lt == DECONV) {
      l = parse_deconv(options, params);
    }else if (lt == UNPOOL) {
      l = parse_unpool(options, params);
    }else if (lt == SOFTMAX){
      l = parse_softmax(options, params);
    }else if (lt == SELF_ATTENTION){
      l = parse_self_attention(options, params);
    }else if(lt == MULTIHEAD_ATTENTION){
      l = parse_multihead_attention(options, params);
    }
    else{
      fprintf(stderr, "Type not recognized: %s\n", s->type);
    }

    option_unused(options);
    net.layers[count] = l;
    if (l.inputs > max_inputs) max_inputs = l.inputs;
    if (l.outputs > max_outputs) max_outputs = l.outputs;
    free_section(s);
    n = n->next;
    ++count;
    if(n){
      if (l.antialiasing) {
        params.h = l.input_layer->out_h;
        params.w = l.input_layer->out_w;
        params.c = l.input_layer->out_c;
        params.inputs = l.input_layer->outputs;
      }
      else {
        params.h = l.out_h;
        params.w = l.out_w;
        params.c = l.out_c;
        params.inputs = l.outputs;
      }
    }
    if (l.w > 1 && l.h > 1) {
      avg_outputs += l.outputs;
      avg_counter++;
    }
  }

  free_list(sections);

  return net;
}


//@hardware info
void parse_hardware_cfg(char *filename, asic *hardware) {
  list *sections = read_cfg(filename);
  node *n = sections->front;
  if(!n) error("Config file has no sections");
  // 硬件配置信息全部存在asic结构体hardware中

  section *s = (section *)n->val;
  list *options = s->options;
  hardware->mac_num = option_find_int_quiet(options, "mac_num",1);
  hardware->mac_dtype = option_find_int_quiet(options, "mac_dtype",1);
  hardware->mac_pipeline = option_find_int_quiet(options, "mac_pipeline",1);
  hardware->mac_stall_cycle = option_find_int_quiet(options, "mac_stall_cycle",0);
  hardware->vec_num = option_find_int_quiet(options, "vec_num",1);
  hardware->vec_dtype = option_find_int_quiet(options, "vec_dtype",1);
  hardware->surpass_num = option_find_int_quiet(options, "surpass_num",0);
  hardware->surpass_dtype = option_find_int_quiet(options, "surpass_dtype",0);
  hardware->vec_pipeline = option_find_int_quiet(options, "vec_pipeline",1);
  hardware->vec_stall_cycle = option_find_int_quiet(options, "vec_stall_cycle",0);
  hardware->pwr = option_find_float_quiet(options, "power",100000);
  hardware->area = option_find_float_quiet(options, "area",100000);
  hardware->off_bw = option_find_float_quiet(options, "offchip_bandwidth",0.0001);
  hardware->latency = option_find_float_quiet(options, "offchip_latency",0);
  hardware->freq = option_find_float_quiet(options, "frequency",1);
  hardware->ave_alu_eff = option_find_float_quiet(options, "average_alu_efficiency",100);
  hardware->ave_bw_eff = option_find_float_quiet(options, "average_bandwidth_efficiency",100);
  hardware->surpass_eff = option_find_float_quiet(options, "surpass_efficiency",100);

  free_list(sections);
}


list *read_cfg(char *filename) {
  FILE *file = fopen(filename, "r");
  if(file == 0) file_error(filename);
  char *line;
  int nu = 0;
  list *sections = make_list();
  section *current = 0;
  while((line=fgetl(file)) != 0){
    ++ nu;
    strip(line);
    switch(line[0]){
      case '[':
        current = (section*)xmalloc(sizeof(section));
        list_insert(sections, current);
        current->options = make_list();
        current->type = line;
        break;
      case '\0':
      case '#':
      case ';':
        free(line);
        break;
      default:
        if(!read_option(line, current->options)){
          fprintf(stderr, "Config file error line %d, couldn't parse: %s\n", nu, line);
          free(line);
        }
        break;
    }
  }
  fclose(file);
  return sections;
}
