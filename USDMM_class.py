import jax.numpy as jnp
from jax.scipy.signal import convolve2d as convolve2d_jax
import jax
import numpy as np
import random
import copy
from time import time
import pickle
import os
import cv2
from itertools import product
import USDMM_data

#directory for colab
#work_directory = '.'
#work_output = 'output_colab' 

#directory for linux
work_directory = '/app/'
work_output = f'{work_directory}output' 

class WOMC_JAX:

    def __init__(self, nlayer, wlen, train_size, val_size, test_size, img_type,
                 error_type, neighbors_sample_f, neighbors_sample_w, epoch_f, epoch_w, 
                 batch, path_results, name_save, seed,
                 early_stop_round_f , early_stop_round_w, w_ini):
                 #train_img, train_yimg, val_img, val_yimg, test_img, test_yimg):

        WOMC_IMG = USDMM_data.WOMC_load_images(
            train_size = train_size,
            val_size = val_size,
            test_size = test_size,
            img_type = img_type
        )

        self.nlayer = nlayer
        self.wlen = wlen
        self.wsize = wlen ** 2
        self.joint_max_size = 2**self.wsize
        k_grid, i_grid = jnp.meshgrid(jnp.arange(self.nlayer), jnp.arange(self.wsize), indexing='ij')
        self.k_ix = k_grid.flatten().astype(jnp.int8)
        self.i_ix = i_grid.flatten().astype(jnp.int8)
        self.identity_matrix = jnp.eye(self.wsize, dtype=jnp.int8)
        self.train_size = WOMC_IMG.train_size
        self.val_size = WOMC_IMG.val_size
        self.test_size = WOMC_IMG.test_size
        self.error_type = error_type
        self.neighbors_sample_f = neighbors_sample_f
        self.neighbors_sample_w = neighbors_sample_w
        self.epoch_f = epoch_f
        self.epoch_w = epoch_w
        self.batch = batch
        self.num_batches = WOMC_IMG.train_size // batch
        self.windows_continuos = self.load_or_generate_matrices(self.wlen) #jnp.load('./data/window_continuous_array.txt', allow_pickle=True).astype(jnp.int8)
        #self.dict_matrices = self.create_w_matrices_dict(self.windows_continuos, self.wlen)
        self.increase = int(round(wlen/2-0.1,0))
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        random.seed(seed)
        np.random.seed(seed)


        self.w_hist = {"W_key": [],"W":[],"error":[], "f_epoch_min":[]}
        self.windows_visit = 1
        self.error_ep_f_hist = {}
        self.error_ep_f = {"W_key": [], "epoch_w": [], "epoch_f":[], "error":[], "time":[] }
        wind_size_dict = {f'window_size_{i}': [] for i in range(self.nlayer)}
        self.error_ep_f = {key: value for d in [self.error_ep_f, wind_size_dict] for key, value in d.items()}

        self.jax_train = jnp.array(WOMC_IMG.jax_train).astype(jnp.int8)
        self.jax_ytrain = jnp.array(WOMC_IMG.jax_ytrain).astype(jnp.int8)

        self.jax_val = jnp.array(WOMC_IMG.jax_val).astype(jnp.int8)
        self.jax_yval = jnp.array(WOMC_IMG.jax_yval).astype(jnp.int8)

        self.jax_test = jnp.array(WOMC_IMG.jax_test).astype(jnp.int8)
        self.jax_ytest = jnp.array(WOMC_IMG.jax_ytest).astype(jnp.int8)

        img_h, img_w = self.jax_train[0].shape

        # Calcular o índice do elemento central
        self.ci_h = img_h // 2
        self.ci_w = img_w // 2

        self.path_results = path_results
        path_file_name = f'{work_output}'
        isExist = os.path.exists(path_file_name)
        if not isExist:
            os.makedirs(path_file_name)
        path_file_name = f'{work_output}/{path_results}'
        isExist = os.path.exists(path_file_name)
        if not isExist:
            os.makedirs(path_file_name)
        path_file_name = f'{work_output}/{path_results}/run'
        isExist = os.path.exists(path_file_name)
        if not isExist:
            os.makedirs(path_file_name)
        path_file_name = f'{work_output}/{path_results}/trained'
        isExist = os.path.exists(path_file_name)
        if not isExist:
            os.makedirs(path_file_name)
        print(f"Resultados serão salvos em {work_output}/{path_results}")
        print(f'Dataset shape: Train = {self.jax_train.shape} / Val = {self.jax_val.shape} / Test = {self.jax_test.shape}')
        self.name_save = name_save

        self.joint_hist = []

        self.early_stop_round_f = int(early_stop_round_f)
        self.early_stop_round_w = int(early_stop_round_w)

        self.error_ep_f_hist = np.array([])

        self.w_ini = jnp.array(w_ini).astype(jnp.int8)

        # Inicializando o timer
        self.start_time = 0
        print('------------------------------------------------------------------')
    
    # Create continuous matrices file
    def is_connected(self, matrix):
        wlen = len(matrix)
        visited = np.zeros((wlen, wlen), dtype=bool)
        
        def dfs(x, y):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if not (0 <= cx < wlen and 0 <= cy < wlen):
                    continue
                if matrix[cx][cy] == 1 and not visited[cx][cy]:
                    visited[cx][cy] = True
                    # Adjacências (cima, baixo, esquerda, direita)
                    neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                    for nx, ny in neighbors:
                        stack.append((nx, ny))

        # Encontre o primeiro 1 na matriz
        for i in range(wlen):
            for j in range(wlen):
                if matrix[i][j] == 1:
                    dfs(i, j)
                    break
            else:
                continue
            break

        # Verifique se todos os 1's foram visitados
        for i in range(wlen):
            for j in range(wlen):
                if matrix[i][j] == 1 and not visited[i][j]:
                    return False
        return True

    def generate_continuous_matrices(self, wlen):
        matrices = []
        # Gera todas as combinações possíveis de matrizes wlen x wlen com 0 e 1
        for bits in product([0, 1], repeat=wlen*wlen):
            if sum(bits) == 0:
                continue
            matrix = np.array(bits).reshape((wlen, wlen))
            if self.is_connected(matrix):
                matrices.append(bits)  # Armazena o reshape (wlen*wlen,)
        return jnp.array(matrices).astype(jnp.int8)


    def load_or_generate_matrices(self, wlen):
        filename = f'./data/window_continuous_wlen{wlen}.txt'
        
        if os.path.exists(filename):
            continuous_matrices = jnp.load(filename, allow_pickle=True).astype(jnp.int8)
        else:
            continuous_matrices = self.generate_continuous_matrices(wlen)
            pickle.dump(continuous_matrices, open(filename, 'wb'))
        return continuous_matrices
    
    def create_w_matrices_for_dict(self, W, joint_function, wlen):
        matrix_k = []
        for minterm in joint_function:
            matrix = jnp.array(W)
            matrix = matrix.at[jnp.where(matrix == 1)].set(minterm)
            matrix = jnp.nan_to_num(matrix)
            matrix_k.append(matrix.reshape((wlen, wlen)))
        return matrix_k
    
    def create_w_matrices_dict(self, windows_continuos, wlen):
        filename = f'./data/dict_matrices_wlen{wlen}.txt'
        if os.path.exists(filename):
          dict_matrices = jnp.load(filename, allow_pickle=True)
        else:
          dict_matrices = {"W": [], "W_matrices": []}
          for W in windows_continuos:
            dict_matrices["W"].append(W)
            ni = jnp.sum(W)
            binary_combinations = jnp.array(list(product([-1, 1], repeat=int(ni))))
            dict_matrices["W_matrices"].append(self.create_w_matrices_for_dict(W, binary_combinations, wlen))
          pickle.dump(dict_matrices, open(filename, 'wb'))
        return dict_matrices

class WOMC_DATA:

    def __init__(self, **kwargs):

        self.data = kwargs
        with open('./data/parameters.pkl', 'wb') as file:
            pickle.dump(self.data, file)
    
def load_from_file():
    with open('./data/parameters.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


