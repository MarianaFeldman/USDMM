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

    womc->windows_continuos = (int*) malloc(sizeof(int) * wsize);

    womc->increase = (int) round(wlen / 2 - 0.1);

    womc->count = 0;

    womc->random_list = (int*) malloc(sizeof(int) * (epoch_f * epoch_w) * 200 + len(windows_continuos) * 2);

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

 

void randon_seed_number(WOMC* womc, int s) {

    srand(s);

    int n_ep = (womc->epoch_f * womc->epoch_w) * 200 + len(womc->windows_continuos) * 2;

    for (int i = 0; i < n_ep; i++) {

        womc->random_list[i] = rand() % 1000000000;

    }

}

 

float* create_joint(float* W) {

    int ni = 0;

    for (int i = 0; i < womc->nlayer; i++) {

        if (!isnan(W[i])) {

            ni += W[i];

        }

    }

    char** Ji = (char**) malloc(sizeof(char*) * pow(2, ni));

    for (int i = 0; i < pow(2, ni); i++) {

        Ji[i] = (char*) malloc(sizeof(char) * (ni + 1));

    }

    for (int i = 0; i < pow(2, ni); i++) {

        for (int j = 0; j < ni; j++) {

            Ji[i][j] = '0';

        }

        Ji[i][ni] = '\0';

    }

    for (int i = 0; i < pow(2, ni); i++) {

        for (int j = 0; j < ni; j++) {

            if (i & (1 << j)) {

                Ji[i][j] = '1';

            }

        }

    }

    int* random_numbers = (int*) malloc(sizeof(int) * pow(2, ni));

    for (int i = 0; i < pow(2, ni); i++) {

        random_numbers[i] = rand() % 2;

    }

    float** joint = (float**) malloc(sizeof(float*) * pow(2, ni));

    for (int i = 0; i < pow(2, ni); i++) {

        joint[i] = (float*) malloc(sizeof(float) * 2);

    }

    for (int i = 0; i < pow(2, ni); i++) {

        joint[i][0] = Ji[i];

        joint[i][1] = random_numbers[i];

    }

    return joint;

}

 

float* convert_binary(float* img) {

    float* img_bin = (float*) malloc(sizeof(float) * img_size);

    for (int i = 0; i < img_size; i++) {

        if (img[i] == 0) {

            img_bin[i] = 1;

        } else if (img[i] == 255) {

            img_bin[i] = -1;

        } else {

            img_bin[i] = img[i];

        }

    }

    return img_bin;

}

 

void get_images(WOMC* womc, float* img_size, char* img_type) {

    float** ximg = (float**) malloc(sizeof(float*) * img_size);

    float** yimg = (float**) malloc(sizeof(float*) * img_size);

    for (int img = 0; img < img_size; img++) {

        float* x = (float*) malloc(sizeof(float) * img_size);

        float* y = (float*) malloc(sizeof(float) * img_size);

        for (int i = 0; i < img_size; i++) {

            x[i] = img[i];

            y[i] = img[i];

        }

        ximg[img] = convert_binary(x);

        yimg[img] = convert_binary(y);

    }

    womc->ximg = ximg;

    womc->yimg = yimg;

}

 

float* create_w_matrices(float* W, float* joint) {

    float** matrices = (float**) malloc(sizeof(float*) * womc->nlayer);

    for (int k = 0; k < womc->nlayer; k++) {

        float** matrix_k = (float**) malloc(sizeof(float*) * len(reduced_minterms));

        float* reduced_minterms = find_minterms(joint[k]);

        for (int minterm = 0; minterm < len(reduced_minterms); minterm++) {

            float* matrix = (float*) malloc(sizeof(float) * 9);

            for (int i = 0; i < 9; i++) {

                matrix[i] = W[k][i];

            }

            int c = 0;

            for (int i = 0; i < 9; i++) {

                if (W[k][i] == 1) {

                    matrix[i] = reduced_minterms[minterm][c];

                    c++;

                }

            }

            for (int i = 0; i < 9; i++) {

                if (matrix[i] == 0) {

                    matrix[i] = -1;

                }

                if (isnan(matrix[i])) {

                    matrix[i] = 0;

                }

            }

            matrix_k[minterm] = matrix;

        }

        matrices[k] = matrix_k;

    }

    return matrices;

}

 

float* find_minterms(float* truth_table) {

    float** inputs = (float**) malloc(sizeof(float*) * len(truth_table));

    float** results = (float**) malloc(sizeof(float*) * len(truth_table));

    for (int i = 0; i < len(truth_table); i++) {

        inputs[i] = truth_table[i][0];

        results[i] = truth_table[i][1];

    }

    float** minterms = (float**) malloc(sizeof(float*) * len(results));

    for (int i = 0; i < len(results); i++) {

        if (results[i] == '1') {

            minterms.append(inputs[i]);

        }

    }

    return minterms;

}

 

float* window_error_generate_c(float* W_matrices, float* sample, float* sample_size, float* y, char* error_type, float* Wlast, float* layer, float* bias) {

    float** W_hood = run_window_convolve(sample, sample_size, W_matrices, Wlast, layer, bias);

    float* error_hood = calculate_error(y, W_hood, error_type);

    float** result = (float**) malloc(sizeof(float*) * 2);

    result[0] = W_hood;

    result[1] = error_hood;

    return result;

}

 

float* calculate_error(float* y, float* h, char* et) {

    float* error = (float*) malloc(sizeof(float) * 2);

    error[0] = 0;

    int n_samples = len(y);

    for (int k = 0; k < n_samples; k++) {

        float* h_z = copy.deepcopy(h[k][-1]);

        for (int i = 0; i < len(h_z); i++) {

            if (h_z[i] == -1) {

                h_z[i] = 0;

            }

        }

        float* y_z = copy.deepcopy(y[k]);

        for (int i = 0; i < len(y_z); i++) {

            if (y_z[i] == -1) {

                y_z[i] = 0;

            }

        }

        if (et == 'mae') {

            float sample_error = 0;

            for (int i = 0; i < len(h_z); i++) {

                sample_error += abs(h_z[i] - y_z[i]);

            }

            error[0] += sample_error / len(y_z);

        } else if (et == 'iou') {

            float union = 0;

            float interc = 0;

            for (int i = 0; i < len(h_z); i++) {

                if (h_z[i] + y_z[i] == 2) {

                    interc++;

                }

                if (max(h_z[i], y_z[i]) == 1) {

                    union++;

                }

            }

            error[1] += (1 - interc / union);

        }

    }

    error[0] /= n_samples;

    error[1] /= n_samples;

    return error;

}

 

float* get_batches(float* imgX, float* imgY, int batch_size, int img_size) {

    float** imgX_batches = (float**) malloc(sizeof(float*) * (img_size / batch_size));

    float** imgY_batches = (float**) malloc(sizeof(float*) * (img_size / batch_size));

    for (int i = 0; i < (img_size / batch_size); i++) {

        imgX_batches[i] = (float*) malloc(sizeof(float) * batch_size);

        imgY_batches[i] = (float*) malloc(sizeof(float) * batch_size);

    }

    for (int i = 0; i < img_size; i++) {

        imgX_batches[i / batch_size][i % batch_size] = imgX[i];

        imgY_batches[i / batch_size][i % batch_size] = imgY[i];

    }

    float** result = (float**) malloc(sizeof(float*) * 2);

    result[0] = imgX_batches;

    result[1] = imgY_batches;

    return result;

}

 

float* run_window_convolve(float* sample, float* sample_size, float* W_matrices, float* Wlast, float* layer, float* bias) {

    float** Wsample = (float**) malloc(sizeof(float*) * sample_size);

    for (int k = 0; k < sample_size; k++) {

        float** Wsample_k = (float**) malloc(sizeof(float*) * womc->nlayer);

        for (int i = 0; i < womc->nlayer; i++) {

            if (layer > i) {

                Wsample_k[i] = Wlast[k][i];

            } else if (i == 0) {

                Wsample_k[i] = apply_convolve(sample[k], W_matrices[i], bias[i]);

            } else {

                Wsample_k[i] = apply_convolve(Wsample_k[i - 1], W_matrices[i], bias[i]);

            }

        }

        Wsample[k] = Wsample_k;

    }

    return Wsample;

}

 

float* apply_convolve(float* img, float* W_matrices, float* bias) {

    float* img_c = (float*) malloc(sizeof(float) * len(img));

    for (int i = 0; i < len(img); i++) {

        img_c[i] = 0;

    }

    for (int kernel = 0; kernel < len(W_matrices); kernel++) {

        float* img_b = (float*) malloc(sizeof(float) * (len(img) + 2 * womc->increase));

        for (int i = 0; i < len(img_b); i++) {

            if (i < womc->increase || i >= len(img_b) - womc->increase) {

                img_b[i] = -1;

            } else {

                img_b[i] = img[i - womc->increase];

            }

        }

        float* img_r = (float*) malloc(sizeof(float) * len(img_b));

        for (int i = 0; i < len(img_b); i++) {

            img_r[i] = img_b[i];

        }

        for (int i = 0; i < len(img_r); i++) {

            img_r[i] -= bias;

            if (img_r[i] > 0) {

                img_r[i] = 1;

            } else {

                img_r[i] = 0;

            }

        }

        for (int i = 0; i < len(img_c); i++) {

            img_c[i] += img_r[i + womc->increase];

        }

    }

    for (int i = 0; i < len(img_c); i++) {

        if (img_c[i] == 0) {

            img_c[i] = -1;

        }

    }

    return img_c;

}

 

float* calculate_neighbors(float* W, float* joint, int k, int i, float* Wlast, float* img, float* yimg, int ep, float* bias, float* error_ep_f_hist) {

    float** joint_temp = copy.deepcopy(joint);

    if (joint[k][i][1] == '1') {

        joint_temp[k][i][1] = '0';

    } else {

        joint_temp[k][i][1] = '1';

    }

    float* W_matrices = create_w_matrices(W, joint_temp);

    float** result = window_error_generate_c(W_matrices, img, womc->batch, yimg, womc->error_type, Wlast, k, bias);

    float* W_hood = result[0];

    float* error_hood = result[1];

    error_ep_f_hist["error"].append(error_hood);

    error_ep_f_hist["joint"].append(joint_temp);

    error_ep_f_hist["ix"].append(str(k) + str(i));

    return error_ep_f_hist;

}

 

float* get_error_window(float* W, float* joint, int ep_w) {

    float* W_matrices = create_w_matrices(W, joint);

    float* bias = np.nansum(W, axis=1) - 1;

    if (womc->batch >= womc->train_size) {

        float** train_b = (float**) malloc(sizeof(float*) * 1);

        float** ytrain_b = (float**) malloc(sizeof(float*) * 1);

        train_b[0] = copy.deepcopy(womc->train);

        ytrain_b[0] = copy.deepcopy(womc->ytrain);

    }

    int flg = 0;

    int epoch_min = 0;

    float* W_size = (float*) malloc(sizeof(float) * womc->nlayer);

    for (int i = 0; i < womc->nlayer; i++) {

        W_size[i] = np.sum((W[i] == 1));

    }

    float** Wtrain, w_error = window_error_generate_c(W_matrices, womc->train, womc->train_size, womc->ytrain, womc->error_type, 0, 0, bias);

    for (int ep = 0; ep < womc->epoch_f; ep++) {

        if (womc->batch < womc->train_size) {

            float** train_b, ytrain_b = get_batches(womc->train, womc->ytrain, womc->batch, womc->train_size);

        }

        for (int b = 0; b < womc->num_batches; b++) {

            float* error_ep_f_hist = {"error": [], "joint": [], "ix": []};

            float** Wtrain = run_window_convolve(train_b[b], womc->batch, W_matrices, 0, 0, bias);

            for (int k = 0; k < womc->nlayer; k++) {

                int neighbors_to_visit;

                if (!womc->neighbors_sample || womc->neighbors_sample >= len(joint[k])) {

                    neighbors_to_visit = range(len(joint[k]));

                } else {

                    neighbors_to_visit = random.sample(range(len(joint[k])), womc->neighbors_sample);

                }

                for (int i = 0; i < len(neighbors_to_visit); i++) {

                    error_ep_f_hist = calculate_neighbors(W, joint, k, i, Wtrain, train_b[b], ytrain_b[b], ep, bias, error_ep_f_hist);

                }

                float error_min_ep = min(error_ep_f_hist['error']);

                int ix_min = [i for i, e in enumerate(error_ep_f_hist['error']) if e == error_min_ep];

                int runs = [v for i, v in enumerate(error_ep_f_hist['ix']) if i in(ix_min)];

                int ix_run = error_ep_f_hist['ix'].index(min(runs));

                joint = error_ep_f_hist['joint'][ix_run];

            }

            float** W_matrices = create_w_matrices(W, joint);

            float* Wtrain_min, w_error_min = window_error_generate_c(W_matrices, womc->train, womc->train_size, womc->ytrain, womc->error_type, 0, 0, bias);

            womc->error_ep_f["W_key"].append(womc->windows_visit);

            womc->error_ep_f["epoch_w"].append(ep_w);

            womc->error_ep_f["epoch_f"].append(ep);

            womc->error_ep_f["error"].append(w_error_min);

            womc->error_ep_f["time"].append((time() - womc->start_time));

            for (int i = 0; i < womc->nlayer; i++) {

                womc->error_ep_f[f"window_size_{i}"].append(W_size[i]);

            }

            if (w_error_min < w_error) {

                w_error = w_error_min;

                joint_min = copy.deepcopy(joint);

                flg = 1;

                epoch_min = ep;

            }

            if (ep - epoch_min) > womc->early_stop_round_f {

                break;

            }

        }

        if (flg == 1) {

            joint = copy.deepcopy(joint_min);

        }

        float** W_matrices = create_w_matrices(W, joint);

        float* error_val = window_error_generate_c(W_matrices, womc->val, womc->val_size, womc->yval, womc->error_type, womc->val, 0, bias);

        float* error = np.array([w_error, error_val]);

        float** result = (float**) malloc(sizeof(float*) * 3);

        result[0] = joint;

        result[1] = error;

        result[2] = epoch_min;

        return result;

    }

}

 

float* check_neighboors(float* W, float* joint, int ep_w, float* error_ep) {

    for (int k = 0; k < womc->nlayer; k++) {

        int* nan_idx = np.where(np.isnan(W[k]))[0];

        float* w_line_temp_base = copy.deepcopy(W[k]);

        for (int i = 0; i < len(nan_idx); i++) {

            error_ep = neighboors_func(w_line_temp_base, i, W, joint, k, ep_w, error_ep, 1);

        }

    }

    for (int k = 0; k < womc->nlayer; k++) {

        int* nan_idx = np.where(W[k] == 1)[0];

        float* w_line_temp_base = copy.deepcopy(W[k]);

        for (int i = 0; i < len(nan_idx); i++) {

            error_ep = neighboors_func(w_line_temp_base, i, W, joint, k, ep_w, error_ep, 0);

        }

    }

    return error_ep;

}

 

float* neighboors_func(float* w_line_temp_base, int i, float* W, float* joint, int k, int ep_w, float* error_ep, int type) {

    float* W_line_temp = copy.deepcopy(w_line_temp_base);

    if (type == 1) {

        W_line_temp[i] = 1;

    } else {

        W_line_temp[i] = np.nan;

    }

    float* W_line_temp_NN = copy.deepcopy(W_line_temp);

    for (int i = 0; i < len(W_line_temp_NN); i++) {

        if (isnan(W_line_temp_NN[i])) {

            W_line_temp_NN[i] = 0;

        }

    }

    int* random_list = (int*) malloc(sizeof(int) * len(W_line_temp_NN));

    for (int i = 0; i < len(W_line_temp_NN); i++) {

        random_list[i] = rand() % 2;

    }

    float** joint_temp = copy.deepcopy(joint);

    joint_temp[k] = create_joint(W_line_temp);

    float** result = get_error_window(W, joint_temp, ep_w);

    float* w_error = result[1];

    error_ep['error_train'].append(w_error[0]);

    error_ep['error_val'].append(w_error[1]);

    error_ep['W'].append(W);

    error_ep['joint'].append(joint_temp);

    return error_ep;

}

 

void fit(WOMC* womc) {

    womc->start_time = time();

    float** joint, error, f_epoch_min = get_error_window(womc->W, womc->joint, 0);

    womc->w_hist["W_key"].append(womc->windows_visit);

    womc->w_hist["W"].append(window_history(womc->W, womc->nlayer, womc->wsize));

    womc->w_hist["error"].append(error);

    womc->w_hist["f_epoch_min"].append(f_epoch_min);

    int ep_min = 0;

    float** W = copy.deepcopy(womc->W);

    float* error_min = copy.deepcopy(error);

    float** W_min = copy.deepcopy(W);

    float** joint_min = copy.deepcopy(joint);

    float* error_ep = {"epoch": [], "error_train": [], "error_val": []};

    error_ep['epoch'].append(0);

    error_ep['error_train'].append(error[0]);

    error_ep['error_val'].append(error[1]);

    float time_min = (time() - womc->start_time) / 60;

    printf("Time: %.2f | Epoch 0 / %d - start Validation error: %f\n", time_min, womc->epoch_w, error[1]);

    for (int ep = 1; ep <= womc->epoch_w; ep++) {

        float* error_ep_ = {"error_train": [], "error_val": [], "W": [], "joint": []};

        error_ep_ = check_neighboors(W, joint, ep, error_ep_);

        if (error_ep_["error_train"]) {

            int ix = error_ep_['error_val'].index(min(error_ep_['error_val']));

            float* error = np.array([error_ep_['error_train'][ix], error_ep_['error_val'][ix]]);

            float** W = error_ep_['W'][ix];

            float** joint = error_ep_['joint'][ix];

        }

        if (error[1] < error_min[1]) {

            int ep_min = ep;

            save_window(joint, W);

            float* error_min = copy.deepcopy(error);

            float** W_min = copy.deepcopy(W);

            float** joint_min = copy.deepcopy(joint);

            float* W_matrices = create_w_matrices(W_min, joint_min);

            float* bias = np.nansum(W, axis=1) - 1;

            float** Wtrain = run_window_convolve(womc->train, womc->train_size, W_matrices, 0, 0, bias);

            float** Wval = run_window_convolve(womc->val, womc->val_size, W_matrices, 0, 0, bias);

            float** Wtest = run_window_convolve(womc->test, womc->test_size, W_matrices, 0, 0, bias);

            save_results_complet_in(Wtrain, Wval, Wtest, f'_epoch{ep}');

        }

        error_ep['epoch'].append(ep);

        error_ep['error_train'].append(error[0]);

        error_ep['error_val'].append(error[1]);

        save_to_csv(error_ep, womc->path_results + '/run/error_ep_w' + womc->name_save + '_epoch' + str(ep));

        save_to_csv(womc->error_ep_f, womc->path_results + '/run/error_ep_f' + womc->name_save + '_epoch' + str(ep));

        float time_min = (time() - womc->start_time) / 60;

        printf("Time: %.2f | Epoch %d / %d - Validation error: %f\n", time_min, ep, womc->epoch_w, error[1]);

        if (ep - ep_min > womc->early_stop_round_w) {

            printf("End by Early Stop Round\n");

            break;

        }

    }

    printf("----------------------------------------------------------------------\n");

    float** Wtest, error_test = window_error_generate_c(W_matrices, womc->test, womc->test_size, womc->ytest, womc->error_type, 0, 0, bias);

    float** Wtrain, error_train = window_error_generate_c(W_matrices, womc->train, womc->train_size, womc->ytrain, womc->error_type, 0, 0, bias);

    float** Wval, error_val = window_error_generate_c(W_matrices, womc->val, womc->val_size, womc->yval, womc->error_type, 0, 0, bias);

    printf("End of testing\n");

    float end_time = time();

    float time_min = (end_time - womc->start_time) / 60;

    printf("Time: %.2f | Min-Epoch %d / %d - Train error: %f / Validation error: %f / Test error: %f\n", time_min, ep_min, womc->epoch_w, error_train, error_val, error_test);

    save_results_complet(Wtrain, Wval, Wtest);

    pickle.dump(womc->w_hist, open(f'{womc->path_results}/W_hist{womc->name_save}.txt', 'wb'));

    pickle.dump(error_ep, open(f'{womc->path_results}/error_ep_w{womc->name_save}.txt', 'wb'));

    pickle.dump(womc->error_ep_f, open(f'{womc->path_results}/error_ep_f{womc->name_save}.txt', 'wb'));

    printf("Janela Final Aprendida: %f\n", W_min);

    printf("Joint Final Aprendida: %f\n", joint_min);

}

 

void save_window(float* joint, float* W) {

    FILE* file_joint = fopen(f'{womc->path_results}/joint{womc->name_save}.txt', 'wb');

    fwrite(joint, sizeof(float), len(joint), file_joint);

    fclose(file_joint);

    FILE* file_W = fopen(f'{womc->path_results}/W{womc->name_save}.txt', 'wb');

    fwrite(W, sizeof(float), len(W), file_W);

    fclose(file_W);

}

 

void save_results_complet(float* Wtrain, float* Wval, float* Wtest) {

    for (int k = 0; k < womc->nlayer; k++) {

        save_results(Wtrain, 'train', k);

        save_results(Wval, 'val', k);

        save_results(Wtest, 'test', k);

    }

}

 

void save_results(float* W, char* img_type, int k) {

    for (int img = 0; img < len(W); img++) {

        float* x = copy.deepcopy(W[img - 1][k]);

        for (int i = 0; i < len(x); i++) {

            if (x[i] == -1) {

                x[i] = 255;

            } else if (x[i] == 1) {

                x[i] = 0;

            }

        }

        FILE* file = fopen(f'{womc->path_results}/{img_type}_op{k + 1}_{img:02d}{womc->name_save}.jpg', 'wb');

        fwrite(x, sizeof(float), len(x), file);

        fclose(file);

    }

}

 

void save_results_complet_in(float* Wtrain, float* Wval, float* Wtest, int ep) {

    for (int k = 0; k < womc->nlayer; k++) {

        save_results_in(Wtrain, 'train', k, ep);

        save_results_in(Wval, 'val', k, ep);

        save_results_in(Wtest, 'test', k, ep);

    }

}

 

void save_results_in(float* W, char* img_type, int k, int ep) {

    for (int img = 0; img < len(W); img++) {

        float* x = copy.deepcopy(W[img - 1][k]);

        for (int i = 0; i < len(x); i++) {

            if (x[i] == -1) {

                x[i] = 255;

            } else if (x[i] == 1) {

                x[i] = 0;

            }

        }

        FILE* file = fopen(f'{womc->path_results}/run/{img_type}_op{k + 1}_{img:02d}{womc->name_save}{ep}.jpg', 'wb');

        fwrite(x, sizeof(float), len(x), file);

        fclose(file);

    }

}

 

float* window_history(float* W, int nlayer, int wsize) {

    char* window_hist = (char*) malloc(sizeof(char) * (nlayer * wsize));

    for (int k = 0; k < nlayer; k++) {

        if (k == 0) {

            window_hist = ''.join([''.join(item) for item in np.reshape(W[k], (wsize,)).astype(str)]);

        } else {

            window_hist = window_hist + ''.join([''.join(item) for item in np.reshape(W[k], (wsize,)).astype(str)]);

        }

    }

    return window_hist;

}

 

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <time.h>

#include "numpy_helper.h" // Assuming a custom helper header for NumPy-like operations in C

 

// Assuming the existence of a struct representing the class containing these methods

typedef struct MyClass {

    int *random_list;

    char **windows_continuos;

    int test_size;

    int *test;

    int *ytest;

    int train_size;

    int *train;

    int *ytrain;

    int val_size;

    int *val;

    int *yval;

    // ... other members ...

} MyClass;

 

void create_window(MyClass *self, double *W) {

    srand(self->random_list[0]);

    int wind = rand() % strlen(self->windows_continuos[0]);

    for (int i = 0; i < strlen(self->windows_continuos[wind]); i++) {

        W[i] = (self->windows_continuos[wind][i] == '1') ? 1.0 : NAN;

    }

}

 

void save_to_csv(double *data, int data_length, const char *name) {

    FILE *fp;

    char filename[256];

    sprintf(filename, "%s.csv", name);

    fp = fopen(filename, "w");

    if (fp == NULL) {

        fprintf(stderr, "Failed to open file for writing\n");

        return;

    }

    for (int i = 0; i < data_length; i++) {

        if (!isnan(data[i])) {

            fprintf(fp, "%f\n", data[i]);

        }

    }

    fclose(fp);

}

//-----------------------------------------------------------------------

void results_after_fit(MyClass *self, const char *path_results, const char *name_save) {

    char joint_path[256], W_path[256];

    sprintf(joint_path, "%s/joint%s.txt", path_results, name_save);

    sprintf(W_path, "%s/W%s.txt", path_results, name_save);

    printf("Reading from: %s\n", path_results);

    // Assuming create_w_matrices, window_error_generate_c, and save_results_complet are implemented elsewhere

    // double *W_matrices = create_w_matrices(W, joint);

    // double bias = nansum(W, axis_length) - 1;

    // double Wtest, error_test = window_error_generate_c(W_matrices, self->test, self->test_size, self->ytest, error_type, 0, 0, bias);

    // ... similar for Wtrain and Wval ...

    printf("Train error: %f / Validation error: %f / Test error: %f\n", error_train, error_val, error_test);

    // save_results_complet(Wtrain, Wval, Wtest);

}

 

void test_neightbors(MyClass *self) {

    clock_t start_time = clock();

    // Assuming error_ep is a struct or similar data structure

    // error_ep = check_neighboors(self->W, self->joint, 0, error_ep);

    if (/* condition to check if error_ep["error_train"] is not empty */) {

        // int ix = index_of_min(error_ep['error_val']);

       // double error[2] = {error_ep['error_train'][ix], error_ep['error_val'][ix]};

        // double *W = error_ep['W'][ix];

        // double *joint = error_ep['joint'][ix];

    }

    printf("error_train: %f\n", error[0]);

    printf("error_val: %f\n", error[1]);

    printf("W: ");

    // print_array(W, W_length); // Assuming print_array is a function to print an array

    clock_t end_time = clock();

    double time_min = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Time: %.2f\n", time_min);

}

 

void test_window(MyClass *self) {

    clock_t start = clock();

    // Assuming get_error_window is implemented elsewhere

    // double error, f_epoch_min;

    // get_error_window(self->W, self->joint, 0, &error, &f_epoch_min);

    clock_t end = clock();

    printf("tempo de execução: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

    printf("época-min: %f - com erro: %f\n", f_epoch_min, error);

}