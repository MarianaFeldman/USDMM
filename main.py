 #main.py
import USDMM_class 
import jax
import jax.numpy as jnp
import numpy as np
import cv2
import pandas as pd
import argparse

work_directory = 'app'
results_dir = f"/{work_directory}/output/results"

def run_training(config):
    
    model = USDMM_class.WOMC_DATA(**config)
    import USDMM_JAX as USDMM
    return USDMM.fit()

def str_to_jax_array(s):
    # Remove aspas simples adicionais e espaços, depois converte a string em uma lista de inteiros
    s = s.replace("'", "").strip('[]')
    lst = [int(item) for item in s.split(',')]
    return jnp.array(lst).astype(jnp.int8)

def main():
    # Configuração base
    base_config = {
        'nlayer': 2,
        'wlen': 3,
        'train_size': 10,
        'val_size': 10,
        'test_size': 10,
        'img_type': 'img_n',
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

    
    

    result, total_time, min_train_error, min_val_error, min_test_error = run_training(base_config)
    result_df = pd.DataFrame(result)
    result_df['time'] = total_time
    result_df['min_train_error'] = min_train_error
    result_df['min_val_error'] = min_val_error
    result_df['min_test_error'] = min_test_error
    result_df.to_csv(f"{results_dir}/results.csv", index=False)
    print('---------------------------------------------')
    
if __name__ == "__main__":
    main()

