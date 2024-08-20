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
    nlayer = 1, # INT -> Number of operators in each layer
    wlen = 3,  # INT -> Size of the operator (wlen*wlen)
    train_size = 300, # INT -> Number of images to train
    val_size = 300, # INT -> Number of images to validate
    test_size = 300, # INT -> Number of images to test
    img_type = 'GoL_final', #STR ->'img_n' / 'gl' / 'classification' / 'GoL'
    error_type = 'iou', # 'mae' / 'iou' -> type of error
    neighbors_sample_f = 16, # INT/False -> Number of neighbors to sort
    neighbors_sample_w = 16, # INT/False -> Number of neighbors to sort
    epoch_f = 2000, # INT -> Number of epochs for the boolean function lattice (fixed windows)
    epoch_w = 200, # INT -> Number of epochs for the windows lattice
    batch = 50, # INT -> Batch size
    path_results = 'results_V1_GoL', # STR -> file where we want to save the results
    name_save='_V1', # STR -> pos fixed name for the results saved
    seed = 1, #INT -> seed for reproducibilit
    early_stop_round_f = 50, #20, #INT -> max number of epochs without changes in the boolean function lattice
    early_stop_round_w = 10, #10 #INT -> max number of epochs without changes in the windows lattice
    w_ini = [0, 1, 0, 1, 1, 1, 0, 1, 0] #[1, 1, 1, 1, 1, 1, 1, 1, 1], #[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0],#[0, 1, 0, 1, 1, 1, 0, 1, 0], #inicial operator
)

import USDMM_JAX_mae as USDMM

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
    all_results_fixed =[]

    for img_type in ['GoL_final', 'GoL_final_sp', 'GoL_final_sp1','GoL_final_sp2','GoL_final_sp3', 'GoL_final_sp4']:
        config['img_type'] = img_type
        print(f'Running with images {img_type}')
        for seed in range(10):  # Rodar 10 vezes com diferentes seeds
            config['seed'] = seed
            if img_type =='GoL_final_sp':
                seed_path = seed+10
            elif img_type == 'GoL_final_sp1':
                seed_path = seed+20
            elif img_type == 'GoL_final_sp2':
                seed_path = seed+30
            elif img_type == 'GoL_final_sp3':
                seed_path = seed+40
            elif img_type == 'GoL_final_sp4':
                seed_path = seed+50
            else:
                seed_path = seed
            path_results = f'results_run{run}_V{seed_path}'
            config['path_results'] = path_results
            
            result, total_time, min_train_error, min_val_error, min_test_error,W, joint, joint_shape = run_training(config, seed)
            result_df = pd.DataFrame(result)
            result_df['seed'] = seed
            result_df['img_type'] = img_type
            result_df['path_results'] =  path_results
            result_df['time'] = total_time
            
            
            version = '1'
            WOMC_WINDOW = USDMM_data.load_window(
                path_window = f'/{work_directory}/output/{path_results}/trained/W_V{version}.txt',
                path_joint = f'/{work_directory}/output/{path_results}/trained/joint_V{version}.txt',
                path_joint_shape = f'/{work_directory}/output/{path_results}/trained/joint_shape_V{version}.txt'
            )
            W_ideal = jnp.array([[1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(USDMM.WOMC.nlayer)]).astype(jnp.int8)
            diff_w = jnp.where(W != W_ideal)[1].shape[0]
            diif_p_w = 100*diff_w/len(W[0])

            filename_joint = f'./data/joint_ideal_GoL.txt'
            joint_ideal = jnp.load(filename_joint, allow_pickle=True)
            try:
                diff_joint = jnp.where(joint != joint_ideal)[1].shape[0]
                diif_p_joint = 100*diff_joint/joint.shape[1]
                
            except:
                diff_joint=512
                diif_p_joint=100
            print(f'Diff Learned from Ideal - W: {diff_w}/{diif_p_w:.2f}% | Joint: {diff_joint}/{diif_p_joint}%')
            result_df['diff_w'] = diff_w
            result_df['diff_w_p'] = diif_p_w
            result_df['diff_joint'] = diff_joint
            result_df['diif_p_joint'] = diif_p_joint

            WOMC_IMG = USDMM_data.WOMC_load_images(
                train_size = 1000,
                val_size = 0,
                test_size = 0,
                img_type = 'GoL'
            )
            WOMC_IMG_final = USDMM_data.WOMC_load_images(
                train_size = 5000,
                val_size = 0,
                test_size = 0,
                img_type = 'GoL_final'
            )
            max_w = jnp.max(jnp.sum(W, axis=1))
            USDMM.WOMC.joint_max_size = 2**max_w.item()
            W_matrices_all = USDMM.create_w_matrices(W)
            W_matrices = USDMM.mult_w_matrices(W_matrices_all, joint)
            bias = (jnp.sum(W, axis=1) - 1).astype(jnp.int8)
            W_hood,w_error =  USDMM.run_window_convolve_jit(WOMC_IMG.jax_train, WOMC_IMG.jax_ytrain, W_matrices, bias)
            W_hood2,w_error2 =  USDMM.run_window_convolve_jit(WOMC_IMG_final.jax_train, WOMC_IMG_final.jax_ytrain, W_matrices, bias)

            print('*-*-*-*-*-*-')
            print(f'Run Learning Window | Erro GoL_final = {w_error2}/ Erro Gosper glider: {w_error}')
            print('*-*-*-*-*-*-')
            result_df['total_error_gol_final'] = w_error2
            result_df['total_error_gosper_glider'] = w_error
            
            print('Learning with Fixed Window')
            USDMM_class.WOMC_DATA(
                nlayer = 1, # INT -> Number of operators in each layer
                wlen = 3,  # INT -> Size of the operator (wlen*wlen)
                train_size = base_config['train_size'], # INT -> Number of images to train
                val_size = base_config['val_size'], # INT -> Number of images to validate
                test_size = 100, # INT -> Number of images to test
                img_type = img_type,#'GoL_final', #STR ->'img_n' / 'gl' / 'classification' / 'GoL'
                error_type = 'mae', # 'mae' / 'iou' -> type of error
                neighbors_sample_f = False, # INT/False -> Number of neighbors to sort
                neighbors_sample_w = False, # INT/False -> Number of neighbors to sort
                epoch_f = 2000, # INT -> Number of epochs for the boolean function lattice (fixed windows)
                epoch_w = 200, # INT -> Number of epochs for the windows lattice
                batch = 50, # INT -> Batch size
                path_results = path_results,#'results_V1_GoL', # STR -> file where we want to save the results
                name_save='_V1', # STR -> pos fixed name for the results saved
                seed = seed, #INT -> seed for reproducibilit
                early_stop_round_f = 100, #20, #INT -> max number of epochs without changes in the boolean function lattice
                early_stop_round_w = 20, #10 #INT -> max number of epochs without changes in the windows lattice
                w_ini = [1, 1, 1, 1, 1, 1, 1, 1, 1], #[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0],#[0, 1, 0, 1, 1, 1, 0, 1, 0], #inicial operator
            )
            importlib.reload(USDMM)
            
            joint,joint_shape = USDMM.create_joint(W_ideal)
            error_ep_f, error, joint_new, total_time, epoch_min, ep = USDMM.get_error_fixed_window(W_ideal, joint, joint_shape, False)
            #diff = jnp.where(joint_new != joit_ideal)[1].shape[0]
            USDMM.save_window_fixed(joint_new,joint_shape, W_ideal)
            result_df_fixed = pd.DataFrame(error_ep_f)
            result_df_fixed['seed'] = seed
            result_df_fixed['img_type'] = img_type
            result_df_fixed['time'] = total_time
            result_df_fixed['epoch_min'] = epoch_min
            result_df_fixed['total_epoch'] = ep
            result_df_fixed['error_train'] = error[0]
            result_df_fixed['error_val'] = error[1]
            result_df_fixed['error_test'] = error[2]


            max_w = jnp.max(jnp.sum(W_ideal, axis=1))
            USDMM.WOMC.joint_max_size = 2**max_w.item()
            W_matrices_all = USDMM.create_w_matrices(W_ideal)
            W_matrices = USDMM.mult_w_matrices(W_matrices_all, joint_new)
            bias = (jnp.sum(W_ideal, axis=1) - 1).astype(jnp.int8)
            W_hood,w_error =  USDMM.run_window_convolve_jit(WOMC_IMG.jax_train, WOMC_IMG.jax_ytrain, W_matrices, bias)
            W_hood2,w_error2 =  USDMM.run_window_convolve_jit(WOMC_IMG_final.jax_train, WOMC_IMG_final.jax_ytrain, W_matrices, bias)

            print('*-*-*-*-*-*-')
            print(f'Run Fixed Window | Erro GoL_final = {w_error2}/ Erro Gosper glider: {w_error}')
            print('*-*-*-*-*-*-')
            result_df_fixed['total_error_gol_final'] = w_error2
            result_df_fixed['total_error_gosper_glider'] = w_error
            try:
                diff_joint_fixed = jnp.where(joint_new!= joint_ideal)[1].shape[0]
                diif_p_joint_fixed = 100*diff_joint_fixed/joint_new.shape[1]
            except:
                diff_joint_fixed=512
                diif_p_joint_fixed=100
            print(f'Diff Learned from Ideal - Joint: {diff_joint_fixed}/{diif_p_joint_fixed}%')

            result_df_fixed['diff_joint'] = diff_joint_fixed
            result_df_fixed['diif_p_joint'] = diif_p_joint_fixed

            all_results.append(result_df)
            all_results_fixed.append(result_df_fixed)

            total_times.append(total_time)
            min_train_errors.append(min_train_error)
            min_val_errors.append(min_val_error)
            min_test_errors.append(min_test_error)
            result_df.to_csv(f"{results_dir}/{path_results}.csv", index=False)
            result_df_fixed.to_csv(f"{results_dir}/{path_results}_fixed.csv", index=False)



    combined_results = pd.concat(all_results)
    combined_results.to_csv(f"/{results_dir}/results_run{run}.csv", index=False)

    combined_results_fixed = pd.concat(all_results_fixed)
    combined_results_fixed.to_csv(f"/{results_dir}/results_run{run}_fixed.csv", index=False)

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
        'min_test_error_std': np.std(min_test_errors),
        'qtd_diff_w_mean': np.mean(diff_w),
        'qtd_diff_joint_mean': np.mean(diff_joint),
        'qtd_diff_joint_fixed_mean': np.mean(diff_joint_fixed)
    })

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(f"/{results_dir}/summary_results_run{run}_3.csv", index=False)


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
        'nlayer': 1,
        'wlen': 3,
        'train_size': 100,
        'val_size': 100,
        'test_size': 100,
        'img_type': 'GoL_final',
        'error_type': 'mae',
        'neighbors_sample_f': 16,
        'neighbors_sample_w': 16,
        'epoch_f': 2000,#100,
        'epoch_w': 200,#20,
        'batch': 50,
        'path_results': 'results_V1',
        'name_save': '_V1',
        'seed': 1,
        'early_stop_round_f': 50,
        'early_stop_round_w': 10,
        'w_ini': [0, 1, 0, 1, 1, 1, 0, 1, 0]
    }

    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--nlayer', type=int, required=True, help='number of layers')
    parser.add_argument('--wlen', type=int, required=True, help='Size of W operator')
    parser.add_argument('--train_size', type=int, required=True, help='Training size')
    parser.add_argument('--val_size', type=int, required=True, help='Validation size')
    parser.add_argument('--neighbors_sample_f', type=int, required=True, help='Neighbors sample factor - function')
    parser.add_argument('--neighbors_sample_w', type=int, required=True, help='Neighbors sample factor - window')
    parser.add_argument('--epoch_f', type=int, required=True, help='number of epochs - function')
    parser.add_argument('--epoch_w', type=int, required=True, help='number of epochs - window')
    parser.add_argument('--es_f', type=int, required=True, help='early stop - function')
    parser.add_argument('--es_w', type=int, required=True, help='early stop - window')
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
    base_config['val_size'] = args.val_size
    base_config['neighbors_sample_f'] = args.neighbors_sample_f
    base_config['neighbors_sample_w'] = args.neighbors_sample_w
    base_config['epoch_f'] = args.epoch_f
    base_config['epoch_w'] = args.epoch_w
    base_config['early_stop_round_f'] = args.es_f
    base_config['early_stop_round_w'] = args.es_w
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