#ifndef NETWORK_H
#define NETWORK_H
#include "simulator.h"

#ifdef __cplusplus
extern "C" {
#endif

network make_network(int n);
void free_sublayer(layer *l);
void free_layer(layer l);
void free_network(network net);

#ifdef __cplusplus
}
#endif

#endif
