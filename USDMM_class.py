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

class WOMC_JAX:

    def __init__(self, nlayer, wlen, train_size, val_size, test_size, error_type,
                 neighbors_sample, epoch_f, epoch_w, batch, path_results, name_save, seed,
                 early_stop_round_f , early_stop_round_w):

        self.nlayer = nlayer
        self.wlen = wlen
        self.wsize = wlen ** 2
        self.joint_max_size = 2**self.wsize
        k_grid, i_grid = jnp.meshgrid(jnp.arange(self.nlayer), jnp.arange(self.wsize), indexing='ij')
        self.k_ix = k_grid.flatten().astype(jnp.int8)
        self.i_ix = i_grid.flatten().astype(jnp.int8)
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.error_type = error_type
        self.neighbors_sample = neighbors_sample
        self.epoch_f = epoch_f
        self.epoch_w = epoch_w
        self.batch = batch
        self.num_batches = train_size // batch
        self.windows_continuos = jnp.load('./data/window_continuous_array.txt', allow_pickle=True).astype(jnp.int8)
        self.dict_matrices = np.load('./data/dict_matrices.txt', allow_pickle=True)
        self.dict_matrices['W'] = [arr.astype(jnp.int8) for arr in self.dict_matrices['W']]
        self.dict_matrices['W_matrices'] = [[arr.astype(jnp.int8) for arr in sublist] for sublist in self.dict_matrices['W_matrices']]
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

        self.train, self.ytrain = np.array(self.get_images(train_size, 'train'))
        self.val, self.yval = np.array(self.get_images(val_size, 'val'))
        self.test, self.ytest = np.array(self.get_images(test_size, 'test'))

        self.jax_train = jnp.array(self.train).astype(jnp.int8)
        self.jax_ytrain = jnp.array(self.ytrain).astype(jnp.int8)

        self.jax_val = jnp.array(self.val).astype(jnp.int8)
        self.jax_yval = jnp.array(self.yval).astype(jnp.int8)

        self.jax_test = jnp.array(self.test).astype(jnp.int8)
        self.jax_ytest = jnp.array(self.ytest).astype(jnp.int8)


        self.path_results = path_results
        isExist = os.path.exists(path_results)
        if not isExist:
            os.makedirs(path_results)
        isExist = os.path.exists(f'{path_results}/run')
        if not isExist:
            os.makedirs(f'{path_results}/run')
        isExist = os.path.exists(f'{path_results}/trained')
        if not isExist:
            os.makedirs(f'{path_results}/trained')
        print("Resultados serão salvos em ",path_results)
        self.name_save = name_save

        self.joint_hist = []

        self.early_stop_round_f = int(early_stop_round_f)
        self.early_stop_round_w = int(early_stop_round_w)

        self.error_ep_f_hist = np.array([])

        # Inicializando o timer
        self.start_time = 0
        print('------------------------------------------------------------------')
    
    def _convert_binary(self, img):
        (_, img_bin) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img_bin = img_bin.astype('float64')
        img_bin[(img_bin==0)]=1
        img_bin[(img_bin==255)]=-1
        return img_bin

    def get_images(self, img_size, img_type):
        ximg = []
        yimg = []
        for img in range(1,img_size+1):
            x = cv2.imread(f'./data/x/{img_type}{img:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            y = cv2.imread(f'./data/y/{img_type}{img:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            ximg.append(self._convert_binary(x))
            yimg.append(self._convert_binary(y))
        return ximg, yimg

class WOMC_DATA:

    def __init__(self, **kwargs):

        self.data = kwargs
        with open('./data/parameters.pkl', 'wb') as file:
            pickle.dump(self.data, file)
    
def load_from_file():
    with open('./data/parameters.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


