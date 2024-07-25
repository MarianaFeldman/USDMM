import jax
import jax.numpy as jnp

import USDMM_class
import importlib
import copy

import pandas as pd
import numpy as np

import os
import argparse
import USDMM_data
from sklearn.metrics import confusion_matrix



work_directory = 'app'



USDMM_class.WOMC_DATA(
    nlayer = 7, # INT -> Number of operators in each layer
    wlen = 5,  # INT -> Size of the operator (wlen*wlen)
    train_size = 3000, # INT -> Number of images to train
    val_size = 1000, # INT -> Number of images to validate
    test_size = 5, # INT -> Number of images to test
    img_type = 'mnist1', #STR ->'img_n' / 'gl' / 'classification' / 'GoL' / 'mnist{i}'
    error_type = 'mae', # 'mae' / 'iou' -> type of error
    neighbors_sample_f = 5, # INT/False -> Number of neighbors to sort
    neighbors_sample_w = 3, # INT/False -> Number of neighbors to sort
    epoch_f = 5000, # INT -> Number of epochs for the boolean function lattice (fixed windows)
    epoch_w = 2, # INT -> Number of epochs for the windows lattice
    batch = 250, # INT -> Batch size
    path_results = 'results_V01_MNIST', # STR -> file where we want to save the results
    name_save='_V01', # STR -> pos fixed name for the results saved
    seed = 1, #INT -> seed for reproducibilit
    early_stop_round_f = 500, #20, #INT -> max number of epochs without changes in the boolean function lattice
    early_stop_round_w = 10, #10 #INT -> max number of epochs without changes in the windows lattice
    w_ini = [0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0],#[0, 1, 0, 1, 1, 1, 0, 1, 0], #inicial operator
)

import USDMM_JAX_C as USDMM

WOMC_IMG_test = USDMM_data.WOMC_load_images(
            train_size = 60000,
            val_size = 0,
            test_size = 10000,
            img_type = 'complet_mnist1'
        )

version = '01'
WOMC_WINDOW_fixed = USDMM_data.load_window(
            path_window = f'./data/W_V{version}_fixed.txt',
            path_joint = f'./data/joint_V{version}_fixed.txt',
            path_joint_shape = f'./data/joint_shape_V{version}_fixed.txt'
        )

def main():
    print('Iniciando Complet data set')
    max_w = jnp.max(jnp.sum(WOMC_WINDOW_fixed.W, axis=1))
    USDMM.WOMC.joint_max_size = 2**max_w.item()
    W_matrices_all = USDMM.create_w_matrices(WOMC_WINDOW_fixed.W)
    W_matrices = USDMM.mult_w_matrices(W_matrices_all, WOMC_WINDOW_fixed.joint)
    bias = (jnp.sum(WOMC_WINDOW_fixed.W, axis=1) - 1).astype(jnp.int8)
    #W_hood,w_error =  USDMM.run_window_convolve_jit(WOMC_IMG.jax_train, WOMC_IMG.jax_ytrain, W_matrices, bias)
    W_hood,w_error =  USDMM.run_window_convolve_jit(WOMC_IMG_test.jax_test, WOMC_IMG_test.jax_ytest, W_matrices, bias)
    #W_hood,w_error =  USDMM.window_error_generate_train(W_matrices,WOMC_IMG.jax_train, WOMC_IMG.jax_ytrain, 0, bias)

    print('*-*-*-*-*-*-')
    print(f'Erro Test Cmplet Digito 1 = {w_error}')
    print('*-*-*-*-*-*-')
    y_pred = jnp.where(W_hood == -1, 0, 1).astype(jnp.int8)
    conf_matrix = confusion_matrix(WOMC_IMG_test.jax_ytest, y_pred)
    print('Matriz de Confusão:')
    print(conf_matrix)

    W_hood_train,w_error_train =  USDMM.run_window_convolve_jit(WOMC_IMG_test.jax_train, WOMC_IMG_test.jax_ytrain, W_matrices, bias)

    print('*-*-*-*-*-*-')
    print(f'Erro Train Cmplet Digito 1 = {w_error_train}')
    print('*-*-*-*-*-*-')
    y_pred = jnp.where(W_hood_train == -1, 0, 1).astype(jnp.int8)
    conf_matrix = confusion_matrix(WOMC_IMG_test.jax_ytrain, y_pred)
    print('Matriz de Confusão:')
    print(conf_matrix)

    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--nlayer', type=int, required=True, help='number of layers')
    parser.add_argument('--wlen', type=int, required=True, help='Size of W operator')
    parser.add_argument('--train_size', type=int, required=True, help='Training size')
    parser.add_argument('--neighbors_sample_f', type=int, required=True, help='Neighbors sample factor - function')
    parser.add_argument('--neighbors_sample_w', type=int, required=True, help='Neighbors sample factor - window')
    parser.add_argument('--epoch_f', type=int, required=True, help='number of epochs - function')
    parser.add_argument('--epoch_w', type=int, required=True, help='number of epochs - window')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--w_ini', type=str, required=True, help='Training size')
    parser.add_argument('--run', type=int, required=True, help='Run number')
    

    #--nlayer 2 wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini [0, 1, 0, 1, 1, 1, 0, 1, 0] --run 1"
    args = parser.parse_args()
        
'''
docker build -t jax-app .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" jax-app
python3 main.py

docker ps
docker cp Documents/GitHub/USDMM/. e52a8931887b:/workspace/

'''

if __name__ == "__main__":
    main()