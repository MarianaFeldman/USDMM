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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



work_directory = 'app'

USDMM_class.WOMC_DATA(
    nlayer = 7, # INT -> Number of operators in each layer
    wlen = 5,  # INT -> Size of the operator (wlen*wlen)
    train_size = 1000, # INT -> Number of images to train
    val_size = 1000, # INT -> Number of images to validate
    test_size = 1000, # INT -> Number of images to test
    img_type = 'mnist0', #STR ->'img_n' / 'gl' / 'classification' / 'GoL' / 'mnist{i}'
    error_type = 'mae', # 'mae' / 'iou' -> type of error
    neighbors_sample_f = 16, # INT/False -> Number of neighbors to sort
    neighbors_sample_w = 16, # INT/False -> Number of neighbors to sort
    epoch_f = 1000, # INT -> Number of epochs for the boolean function lattice (fixed windows)
    epoch_w = 10, # INT -> Number of epochs for the windows lattice
    batch = 200, # INT -> Batch size
    path_results = 'results_V01_MNIST', # STR -> file where we want to save the results
    name_save='_V1', # STR -> pos fixed name for the results saved
    seed = 1, #INT -> seed for reproducibilit
    early_stop_round_f = 100, #20, #INT -> max number of epochs without changes in the boolean function lattice
    early_stop_round_w = 10, #10 #INT -> max number of epochs without changes in the windows lattice
    w_ini = [0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0],#[0, 1, 0, 1, 1, 1, 0, 1, 0], #inicial operator
)

import USDMM_JAX_C as USDMM

def run_training(config, W):
    model = USDMM_class.WOMC_DATA(**config)
    importlib.reload(USDMM)
    return USDMM.fit(W)

def run_experiments(base_config, results_dir, run):
    

    summary_results = []


    config = base_config.copy()


    for digito in [1]:#range(10):  # Rodar 10 vezes com diferentes seeds
        path_old_run = 'results_run63_d1'
        WOMC_WINDOW = USDMM_data.load_window(
            path_window = f'/{work_directory}/output/{path_old_run}/run/W_V1_ep8.txt',
            path_joint = f'/{work_directory}/output/{path_old_run}/run/joint_V1_ep8.txt',
            path_joint_shape = f'/{work_directory}/output/{path_old_run}/run/joint_shape_V1_ep8.txt'
        )
        path_result = f'results_run{run}_d{digito}'
        config['img_type'] = f'mnist{digito}'
        config['path_results'] = path_result
        result, total_time, min_train_error, min_val_error, min_test_error = run_training(config, WOMC_WINDOW.W)
        result_df = pd.DataFrame(result)
        result_df['digito'] = digito
        result_df['path_results'] =  path_result
        result_df['time'] = total_time
        
        
        version = '1'
        WOMC_WINDOW = USDMM_data.load_window(
            path_window = f'/{work_directory}/output/{path_result}/trained/W_V{version}.txt',
            path_joint = f'/{work_directory}/output/{path_result}/trained/joint_V{version}.txt',
            path_joint_shape = f'/{work_directory}/output/{path_result}/trained/joint_shape_V{version}.txt'
        )
        WOMC_IMG = USDMM_data.WOMC_load_images(
            train_size = 1000,
            val_size = 0,
            test_size = 10000,
            img_type = f'complet_mnist{digito}'
        )
        max_w = jnp.max(jnp.sum(WOMC_WINDOW.W, axis=1))
        USDMM.WOMC.joint_max_size = 2**max_w.item()
        W_matrices_all = USDMM.create_w_matrices(WOMC_WINDOW.W)
        W_matrices = USDMM.mult_w_matrices(W_matrices_all, WOMC_WINDOW.joint)
        bias = (jnp.sum(WOMC_WINDOW.W, axis=1) - 1).astype(jnp.int8)
        key = jax.random.PRNGKey(0)

        W_hood_total = None
        total_error = 0

        # Executando a função em 4 partes
        for i in range(4):
            start_idx = i * 2500
            end_idx = (i + 1) * 2500

            # Executa a função na parte atual
            W_hood_part, w_error_fixed_part = USDMM.run_window_convolve_jit(
                WOMC_IMG.jax_test[start_idx:end_idx],
                WOMC_IMG.jax_ytest[start_idx:end_idx],
                W_matrices,
                bias
            )

            # Agregando os resultados
            if W_hood_total is None:
                W_hood_total = W_hood_part
            else:
                W_hood_total = jnp.concatenate((W_hood_total, W_hood_part), axis=0)
            
            total_error += w_error_fixed_part * (end_idx - start_idx)  # Somando o erro ponderado pelo número de elementos

        # Calculando o erro médio total (MAE)
        total_error /= WOMC_IMG.jax_test.shape[0]
        
       
        print('*-*-*-*-*-*-')
        print(f'Erro digito {digito} | Test Complet = {total_error}')
        print('*-*-*-*-*-*-')
        y_pred = jnp.where(W_hood_total == -1, 0, 1).astype(jnp.int8)
        conf_matrix = confusion_matrix(WOMC_IMG.jax_ytest, y_pred, normalize='true')
        print('Matriz de Confusão:')
        print(conf_matrix)
        print("\nPredicted Labels: ", np.unique(y_pred))
        print("True Labels: ", np.unique(WOMC_IMG.jax_ytest))

        result_df['Test_complet_error'] = total_error
        result_df.to_csv(f"{results_dir}/{path_result}.csv", index=False)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(WOMC_IMG.jax_ytest[:3000]))
        disp.plot(cmap=plt.cm.Blues)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # Salvando o gráfico como uma imagem
        plt.savefig(f'/{results_dir}/confusion_matrix_run{run}_d{digito}.png')

        # Exibindo o gráfico
        plt.show()

        print('Learning with Fixed Window')
        USDMM_class.WOMC_DATA(
            nlayer = 7, # INT -> Number of operators in each layer
            wlen = 5,  # INT -> Size of the operator (wlen*wlen)
            train_size = 600, # INT -> Number of images to train
            val_size = 10, # INT -> Number of images to validate
            test_size = 10, # INT -> Number of images to test
            img_type = 'mnist1',#'GoL_final', #STR ->'img_n' / 'gl' / 'classification' / 'GoL'
            error_type = 'mae', # 'mae' / 'iou' -> type of error
            neighbors_sample_f = False, # INT/False -> Number of neighbors to sort
            neighbors_sample_w = False, # INT/False -> Number of neighbors to sort
            epoch_f = 3000, # INT -> Number of epochs for the boolean function lattice (fixed windows)
            epoch_w = 1, # INT -> Number of epochs for the windows lattice
            batch = 50, # INT -> Batch size
            path_results = path_result,#'results_V1_GoL', # STR -> file where we want to save the results
            name_save='_V2', # STR -> pos fixed name for the results saved
            seed = 10, #INT -> seed for reproducibilit
            early_stop_round_f = 500, #20, #INT -> max number of epochs without changes in the boolean function lattice
            early_stop_round_w = 20, #10 #INT -> max number of epochs without changes in the windows lattice
            w_ini = [0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0],#[0, 1, 0, 1, 1, 1, 0, 1, 0], #inicial operator
        )
        importlib.reload(USDMM)
        #joint,joint_shape = USDMM.create_joint(WOMC_WINDOW.W)
        error_ep_f, error, joint_new, total_time, epoch_min, ep = USDMM.get_error_fixed_window(WOMC_WINDOW.W, WOMC_WINDOW.joint, WOMC_WINDOW.joint_shape, False)
        #diff = jnp.where(joint_new != joit_ideal)[1].shape[0]
        USDMM.save_window_fixed(joint_new,WOMC_WINDOW.joint_shape, WOMC_WINDOW.W)
        result_df_fixed = pd.DataFrame(error_ep_f)

        result_df_fixed['time'] = total_time
        result_df_fixed['epoch_min'] = epoch_min
        result_df_fixed['total_epoch'] = ep
        result_df_fixed['error_train'] = error[0]
        result_df_fixed['error_val'] = error[1]
        result_df_fixed['error_test'] = error[2]
        #result_df_fixed.to_csv(f"{results_dir}/{path_result}_fixed.csv", index=False)
        W_matrices = USDMM.mult_w_matrices(W_matrices_all,joint_new)
        bias = (jnp.sum(WOMC_WINDOW.W, axis=1) - 1).astype(jnp.int8)
        
        W_hood_total = None
        total_error = 0

        # Executando a função em 4 partes
        for i in range(4):
            start_idx = i * 2500
            end_idx = (i + 1) * 2500

            # Executa a função na parte atual
            W_hood_part, w_error_fixed_part = USDMM.run_window_convolve_jit(
                WOMC_IMG.jax_test[start_idx:end_idx],
                WOMC_IMG.jax_ytest[start_idx:end_idx],
                W_matrices,
                bias
            )

            # Agregando os resultados
            if W_hood_total is None:
                W_hood_total = W_hood_part
            else:
                W_hood_total = jnp.concatenate((W_hood_total, W_hood_part), axis=0)
            
            total_error += w_error_fixed_part * (end_idx - start_idx)  # Somando o erro ponderado pelo número de elementos

        # Calculando o erro médio total (MAE)
        total_error /= WOMC_IMG.jax_test.shape[0]

        # Resultados finais
        print('*-*-*-*-*-*-')
        print(f'Erro digito {digito} | Test Complet = {total_error}')
        print('*-*-*-*-*-*-')
        y_pred = jnp.where(W_hood_total == -1, 0, 1).astype(jnp.int8)
        

        result_df_fixed['Test_complet_error'] = total_error
        result_df_fixed.to_csv(f"{results_dir}/{path_result}_fixed.csv", index=False)
        #all_results.append(result_df)

        #total_times.append(total_time)
        #min_train_errors.append(min_train_error)
        #min_val_errors.append(min_val_error)
        #min_test_errors.append(min_test_error)
        #result_df.to_csv(f"{results_dir}/{path_result}.csv", index=False)
        conf_matrix = confusion_matrix(WOMC_IMG.jax_ytest, y_pred, normalize='true')
        print('Matriz de Confusão:')
        print(conf_matrix)
        print("\nPredicted Labels: ", np.unique(y_pred))
        print("True Labels: ", np.unique(WOMC_IMG.jax_ytest))
        
        
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(WOMC_IMG.jax_ytest[:3000]))
        disp.plot(cmap=plt.cm.Blues)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        # Salvando o gráfico como uma imagem
        plt.savefig(f'/{results_dir}/confusion_matrix_run{run}_d{digito}_fixed.png')

        # Exibindo o gráfico
        plt.show()

    #combined_results = pd.concat(all_results)
    #combined_results.to_csv(f"/{results_dir}/results_run{run}.csv", index=False)

    


def str_to_jax_array(s):
    # Remove aspas simples adicionais e espaços, depois converte a string em uma lista de inteiros
    s = s.replace("'", "").strip('[]')
    lst = [int(item) for item in s.split(',')]
    return jnp.array(lst).astype(jnp.int8)

def main():

    results_dir = f"/{work_directory}/output/results"
    isExist = os.path.exists(results_dir)
    if not isExist:
        os.makedirs(results_dir)

    # Configuração base
    base_config = {
        'nlayer': 7,
        'wlen': 5,
        'train_size': 100,
        'val_size': 1000,
        'test_size': 10,
        'img_type': 'mnist1',
        'error_type': 'mae',
        'neighbors_sample_f': 16,
        'neighbors_sample_w': 16,
        'epoch_f': 500,#100,
        'epoch_w': 100,#20,
        'batch': 200,
        'path_results': 'results_V1',
        'name_save': '_V1',
        'seed': 10,
        'early_stop_round_f': 50,
        'early_stop_round_w': 10,
        'w_ini': [0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0]
    }

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
    print(args.w_ini)
    print(str_to_jax_array(args.w_ini))
    print('******************************************************')
    print(f"Running with nlayer={args.nlayer}, wlen={args.wlen}, train_size={args.train_size}, neighbors_sample_f={args.neighbors_sample_f}, neighbors_sample_w={args.neighbors_sample_w}, batch={args.batch}, w_ini={args.w_ini}")
    base_config['nlayer'] = args.nlayer
    base_config['wlen'] = args.wlen
    base_config['train_size'] = args.train_size
    base_config['neighbors_sample_f'] = args.neighbors_sample_f
    base_config['neighbors_sample_w'] = args.neighbors_sample_w
    base_config['epoch_f'] = args.epoch_f
    base_config['epoch_w'] = args.epoch_w
    base_config['batch'] = args.batch
    base_config['w_ini'] = str_to_jax_array(args.w_ini)
    run_experiments(base_config, results_dir, args.run)

    # Parâmetros a serem testados
    #param_configs = {
    #    'train_size': [10, 20],
    #    'neighbors_sample_f': [4, 10],
    #    'neighbors_sample_w': [3, 5],
    #    'epoch_f': [50, 100],
    #    'epoch_w': [20],
    #    'batch': [5, 10]
    #}

    
    

    #for param_name, param_values in param_configs.items():
    #    print('******************************************************')
    #    print(f'Iniciando teste {param_name} = {param_values}')
    #    run_experiments(base_config, param_name, param_values, results_dir)
    #    print('******************************************************')
        
'''
docker build -t jax-app .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" jax-app
python3 main.py

docker ps
docker cp Documents/GitHub/USDMM/. e52a8931887b:/workspace/

'''

if __name__ == "__main__":
    main()