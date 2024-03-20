#ifndef PARSER_H
#define PARSER_H

#include "simulator.h"
#include "list.h"

#ifdef __cplusplus
extern "C" {
#endif

// parameters related, maybe need to add something new for the new operator
typedef struct size_params{
    int batch;
    int inputs;
    int h;  //height
    int w;  //width
    int c;  //channels
    int index;
    int time_steps;
    int train;
    network net;
} size_params;

layer parse_convolutional(list *options, size_params params);
layer parse_rnn(list *options, size_params params);
layer parse_lstm(list *options, size_params params);
layer parse_connected(list *options, size_params params);
layer parse_batchnorm(list *options, size_params params);
layer parse_activation(list *options, size_params params);
layer parse_maxpool(list *options, size_params params);
layer parse_avgpool(list *options, size_params params);
layer parse_relu(list *options, size_params params);
layer parse_lrn(list *options, size_params params);
layer parse_deconv(list *options, size_params params);
layer parse_unpool(list *options, size_params params);
layer parse_softmax(list *options, size_params params);
layer parse_self_attention(list *options, size_params params);
layer parse_multihead_attention(list *options, size_params params);
layer parse_feed_forward(list *options, size_params params);

network parse_network_cfg(char *filename);
void parse_hardware_cfg(char *filename, asic *hardware);

#ifdef __cplusplus
}
#endif
#endif
