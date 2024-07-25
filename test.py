import jax
import jax.numpy as jnp

import USDMM_class
import importlib
import copy

import pandas as pd
import numpy as np

import os
import argparse



work_directory = 'app'

USDMM_class.WOMC_DATA(
    nlayer = 2, # INT -> Number of operators in each layer
    wlen = 3,  # INT -> Size of the operator (wlen*wlen)
    train_size = 10, # INT -> Number of images to train
    val_size = 10, # INT -> Number of images to validate
    test_size = 10, # INT -> Number of images to test
    error_type = 'iou', # 'mae' / 'iou' -> type of error
    neighbors_sample_f = 8, # INT/False -> Number of neighbors to sort
    neighbors_sample_w = 5, # INT/False -> Number of neighbors to sort
    epoch_f = 100, # INT -> Number of epochs for the boolean function lattice (fixed windows)
    epoch_w = 20, # INT -> Number of epochs for the windows lattice
    batch = 10, # INT -> Batch size
    path_results = 'test_V1', # STR -> file where we want to save the results
    name_save='_V1', # STR -> pos fixed name for the results saved
    seed = 0, #INT -> seed for reproducibilit
    early_stop_round_f = 1e6, #20, #INT -> max number of epochs without changes in the boolean function lattice
    early_stop_round_w = 1e6, #10 #INT -> max number of epochs without changes in the windows lattice
    w_ini = [0, 1, 0, 1, 1, 1, 0, 1, 0] #inicial operator
)
import USDMM_JAX as USDMM

def run_training(config, seed):
    np.random.seed(seed)
    
    model = USDMM_class.WOMC_DATA(**config)
    importlib.reload(USDMM)
    return USDMM.fit()

def run_experiments(base_config, results_dir, run):
    

    summary_results = []


    config = base_config.copy()

    total_times = []
    min_train_errors = []
    min_val_errors = []
    min_test_errors = []

    all_results = []

    for seed in range(7):  # Rodar 10 vezes com diferentes seeds
        config['seed'] = seed
        config['path_results'] = f'results_run{run}_V{seed}'
        result, total_time, min_train_error, min_val_error, min_test_error = run_training(config, seed)
        result_df = pd.DataFrame(result)
        result_df['seed'] = seed
        result_df['path_results'] =  f'results_run{run}_V{seed}'
        result_df['time'] = total_time
        all_results.append(result_df)

        total_times.append(total_time)
        min_train_errors.append(min_train_error)
        min_val_errors.append(min_val_error)
        min_test_errors.append(min_test_error)
        result_df.to_csv(f"{results_dir}/results_run{run}_V{seed}.csv", index=False)

    combined_results = pd.concat(all_results)
    combined_results.to_csv(f"/{results_dir}/results_run{run}.csv", index=False)

    # Calcular estatísticas
    summary_results.append({
        'run': run,
        'total_time_mean': np.mean(total_times),
        'total_time_median': np.median(total_times),
        'total_time_std': np.std(total_times),
        'min_train_error_mean': np.mean(min_train_errors),
        'min_train_error_median': np.median(min_train_errors),
        'min_train_error_std': np.std(min_train_errors),
        'min_val_error_mean': np.mean(min_val_errors),
        'min_val_error_median': np.median(min_val_errors),
        'min_val_error_std': np.std(min_val_errors),
        'min_test_error_mean': np.mean(min_test_errors),
        'min_test_error_median': np.median(min_test_errors),
        'min_test_error_std': np.std(min_test_errors)
    })

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(f"/{results_dir}/summary_results_run{run}.csv", index=False)


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
        'nlayer': 2,
        'wlen': 3,
        'train_size': 10,
        'val_size': 10,
        'test_size': 10,
        'error_type': 'iou',
        'neighbors_sample_f': 8,
        'neighbors_sample_w': 5,
        'epoch_f': 30,#100,
        'epoch_w': 5,#20,
        'batch': 10,
        'path_results': 'results_V1',
        'name_save': '_V5',
        'seed': 0,
        'early_stop_round_f': 30,
        'early_stop_round_w': 20,
        'w_ini': [0, 1, 0, 1, 1, 1, 0, 1, 0]
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