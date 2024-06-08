import jax
import jax.numpy as jnp
from itertools import product
from time import time
import pickle

def create_w_matrices_for_dict(W, joint_function, wlen):
    indices = jnp.where(W == 1)[0]
    def apply_minterm(minterm):
        matrix = W.at[indices].set(minterm)
        return matrix.reshape((wlen, wlen))
    return jax.vmap(apply_minterm)(joint_function)

def create_w_matrices_dict(windows_continuos, wlen):
    start_time = time()
    dict_matrices = {"W": [], "W_matrices": []}
    total_windows = len(windows_continuos)
    progress_step = total_windows // 500
    for i, W in enumerate(windows_continuos):
        dict_matrices["W"].append(W)
        ni = jnp.sum(W)
        binary_combinations = jnp.array(list(product([-1, 1], repeat=int(ni))), dtype=W.dtype)
        W_matrices = create_w_matrices_for_dict(W, binary_combinations, wlen)
        dict_matrices["W_matrices"].append(W_matrices)
        if (i + 1) % progress_step == 0 or (i + 1) == total_windows:
          progress_percentage = (i + 1) / total_windows * 100
          time_min = (time() -  start_time) / 60
          print(f"Progress: {progress_percentage:.2f}% complete. | Time: {time_min}m")
    return dict_matrices

def main():
    wlen = 5
    work_directory = 'app'
    filename = f'/{work_directory}/data/window_continuous_wlen{wlen}.txt'
    windows_continuos = jnp.load(filename, allow_pickle=True).astype(jnp.int8)
    dict_matrices = create_w_matrices_dict(windows_continuos, wlen)
    filename = f'/{work_directory}/output/dict_matrices_wlen{wlen}.txt'
    pickle.dump(dict_matrices, open(filename, 'wb'))

if __name__ == "__main__":
    main()

'''
docker build -t jax-app .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" jax-app
python3 main.py

docker ps
docker cp Documents/GitHub/USDMM/. e52a8931887b:/workspace/

'''