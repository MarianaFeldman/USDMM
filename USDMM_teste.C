#include <stdio.h>

#include <stdlib.h>

#include <string.h>

 

typedef struct WOMC {

    int nlayer;

    int wlen;

    int wsize;

    int train_size;

    int val_size;

    int test_size;

    char* error_type;

    int neighbors_sample;

    int epoch_f;

    int epoch_w;

    int batch;

    char* path_results;

    char* name_save;

    int seed;

    int parallel;

    int early_stop_round_f;

    int early_stop_round_w;

    int count;

    int windows_visit;

    int increase;

    int num_batches;

    int* random_list;

    int* windows_continuos;

    float** W;

    float** joint;

    float** w_hist;

    float** error_ep_f_hist;

    float** error_ep_f;

    float** train;

    float** ytrain;

    float** val;

    float** yval;

    float** test;

    float** ytest;

    float start_time;

} WOMC;

 

WOMC* create_WOMC(int nlayer, int wlen, int train_size, int val_size, int test_size, char* error_type, int neighbors_sample, int epoch_f, int epoch_w, int batch, char* path_results, char* name_save, int seed, int parallel, int early_stop_round_f, int early_stop_round_w) {

    WOMC* womc = (WOMC*) malloc(sizeof(WOMC));

    womc->nlayer = nlayer;

    womc->wlen = wlen;

    womc->wsize = wlen * wlen;

    womc->train_size = train_size;

    womc->val_size = val_size;

    womc->test_size = test_size;

    womc->error_type = error_type;

    womc->neighbors_sample = neighbors_sample;

    womc->epoch_f = epoch_f;

    womc->epoch_w = epoch_w;

    womc->batch = batch;

    womc->num_batches = train_size / batch;

    womc->windows_continuos = (int*) malloc(sizeof(int) * womc->wsize);

    womc->increase = (int)(wlen / 2 - 0.1);

    womc->count = 0;

    womc->random_list = (int*) malloc(100000*sizeof(int));

    womc->W = (float**) malloc(sizeof(float*) * nlayer);

    womc->joint = (float**) malloc(sizeof(float*) * nlayer);

    womc->w_hist = (float**) malloc(sizeof(float*) * 4);

    womc->error_ep_f_hist = (float**) malloc(sizeof(float*) * 3);

    womc->error_ep_f = (float**) malloc(sizeof(float*) * (nlayer + 4));

    womc->train = (float**) malloc(sizeof(float*) * train_size);

    womc->ytrain = (float**) malloc(sizeof(float*) * train_size);

    womc->val = (float**) malloc(sizeof(float*) * val_size);

    womc->yval = (float**) malloc(sizeof(float*) * val_size);

    womc->test = (float**) malloc(sizeof(float*) * test_size);

    womc->ytest = (float**) malloc(sizeof(float*) * test_size);

    womc->start_time = 0;

    return womc;

}