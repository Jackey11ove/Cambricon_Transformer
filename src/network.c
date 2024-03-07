#include "simulator.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "utils.h"

network make_network(int n) {
  network net = {0};
  net.n = n;
  net.layers = (layer*)xcalloc(net.n, sizeof(layer));

  return net;
}

void free_sublayer(layer *l) {
  if (l) {
    free_layer(*l);
    free(l);
  }
}

void free_layer(layer l) {
  if (l.share_layer != NULL) return;    // don't free shared layers
  if (l.antialiasing) {
    free_sublayer(l.input_layer);
  }
  if (l.type == LSTM) {
    free_sublayer(l.wf);
    free_sublayer(l.wi);
    free_sublayer(l.wg);
    free_sublayer(l.wo);
    free_sublayer(l.uf);
    free_sublayer(l.ui);
    free_sublayer(l.ug);
    free_sublayer(l.uo);
  }
  if (l.type == CRNN) {
    free_sublayer(l.input_layer);
    free_sublayer(l.self_layer);
    free_sublayer(l.output_layer);
  }
}

void free_network(network net) {
  int i;
  for (i = 0; i < net.n; ++i) {
    free_layer(net.layers[i]);
  }
  free(net.layers);

}
