# V2 com convolução

import numpy as np
import itertools
import cv2
import random
import copy
from time import time
import pickle
import os
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

from itertools import product
#from libc.stdio cimport printf
from scipy.signal import convolve2d


#import cupy as cp
#from cupyx.scipy.signal import convolve2d as convolve2d_gpu


class WOMC:
    
    def __init__(self, new, nlayer, wlen, train_size, val_size, test_size, error_type, 
                 neighbors_sample, epoch_f, epoch_w, batch, path_results, name_save, seed, parallel, 
                 early_stop_round_f , early_stop_round_w):
        
        self.nlayer = nlayer
        self.wlen = wlen
        self.wsize = wlen ** 2
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.error_type = error_type
        self.neighbors_sample = neighbors_sample
        self.epoch_f = epoch_f
        self.epoch_w = epoch_w
        self.batch = batch
        self.num_batches = train_size // batch
        self.windows_continuos = np.load('./data/windows_continuos.txt', allow_pickle=True)
        self.increase = int(round(wlen/2-0.1,0))
        random.seed(seed)
        np.random.seed(seed)
        
        if new:
            Wini = np.array([np.nan, 1., np.nan, 1., 1., 1., np.nan, 1., np.nan])
            self.W = [Wini.copy() for _ in range(self.nlayer)]
            self.joint = [self.create_joint(self.W[i]) for i in range(self.nlayer)]


        else:
            self.joint = np.load('./data/joint'+new+'.txt', allow_pickle=True)
            self.W = np.load('./data/W'+new+'.txt', allow_pickle=True)

        print(f"Janela Inicializada: {self.W}")
        #print(f"Joint Inicializada: {self.joint}")


        self.w_hist = {"W_key": [],"W":[],"error":[], "f_epoch_min":[]}
        self.windows_visit = 1
        self.error_ep_f_hist = {}
        self.error_ep_f = {"W_key": [], "epoch_w": [], "epoch_f":[], "error":[], "time":[] }
        wind_size_dict = {f'window_size_{i}': [] for i in range(self.nlayer)}
        self.error_ep_f = {key: value for d in [self.error_ep_f, wind_size_dict] for key, value in d.items()}

        
        self.train, self.ytrain = np.array(self.get_images(train_size, 'train'))
        self.val, self.yval = np.array(self.get_images(val_size, 'val'))
        self.test, self.ytest = np.array(self.get_images(test_size, 'test'))

        #self.cp_train = cp.array(self.train)
        #self.cp_val = cp.array(self.train)
        #self.cp_test = cp.array(self.train)

        self.path_results = path_results
        isExist = os.path.exists(path_results)
        if not isExist:
            os.makedirs(path_results)
        isExist = os.path.exists(f'{path_results}/run')
        if not isExist:
            os.makedirs(f'{path_results}/run')
        print("Resultados serão salvos em ",path_results)
        self.name_save = name_save
        self.parallel = parallel
        if self.parallel not in ['no-parallel','parallel-window','parallel-func','parallel-func-layer']:
            self.parallel = 'no-parallel'
        self.joint_hist = []

        self.early_stop_round_f = early_stop_round_f
        self.early_stop_round_w = early_stop_round_w

        self.error_ep_f_hist = np.array([])

        self.start_time = 0
        print('------------------------------------------------------------------')

    def create_joint2(self,W):
        ni = np.sum(~np.isnan(W))
        binary_combinations = np.array(list(itertools.product([0, 1], repeat=int(ni))))
        random_values = np.random.randint(2, size=len(binary_combinations))
        return np.column_stack((binary_combinations, random_values))

    def create_w_matrices2(self, W, joint):
        matrices = []
        for k in range(self.nlayer):
            matrix_k = []
            reduced_minterms = self.find_minterms(joint[k])
            for minterm in reduced_minterms:
                matrix = W[k].copy()
                matrix[np.where(W[k] == 1)] = minterm
                matrix = np.where(matrix == 0, -1, matrix)
                matrix = np.nan_to_num(matrix)
                matrix_k.append(matrix.reshape((3, 3)))
            matrices.append(matrix_k)
        return matrices

    def find_minterms2(self, truth_table):
        return truth_table[truth_table[:, -1] == 1][:, :-1]

    def create_joint(self,W):
        ni = int(np.sum(~np.isnan(W)))
        binary_combinations = np.array(list(itertools.product([0, 1], repeat=ni)))
        binary_strings = [''.join(map(str, row)) for row in binary_combinations]
        random_values = np.random.randint(2, size=len(binary_strings))
        return np.column_stack((binary_strings, random_values))
    
    def _convert_binary(self,img):
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
    
    def create_w_matrices(self, W, joint):
        matrices = []
        for k in range(self.nlayer):
            matrix_k = []
            reduced_minterms = self.find_minterms(joint[k])
            for minterm in range(len(reduced_minterms)):
                matrix = np.copy(W[k])
                c=0
                for i in np.where(W[k]==1)[0]:
                    matrix[i] = reduced_minterms[minterm][c]
                    c+=1
                matrix = np.where(matrix==0,-1,matrix)
                matrix = np.nan_to_num(matrix)
                matrix_k.append(matrix.reshape((3, 3)))
            matrices.append(matrix_k)
        return matrices
    
    def find_minterms(self, truth_table):
        inputs = truth_table[:, 0]
        results = truth_table[:, 1]
        minterms = []
        for i in np.where(results =='1')[0]:
            minterms.append(inputs[i])
        minterms
        return minterms

    def window_error_generate_c(self, W_matrices, sample, sample_size, y, error_type, Wlast, layer, bias):
        W_hood = self.run_window_convolve(sample, sample_size, W_matrices, Wlast, layer, bias)
        error_hood = self.calculate_error(y, W_hood, error_type)
        return W_hood,error_hood
    
    def calculate_error(self, y, h, et = 'mae'):
        error = 0
        n_samples = len(y)
        for k in range(n_samples):
            #h_z = copy.deepcopy(h[k][-1])
            h_z = copy.deepcopy(h[-1][k])
            h_z[h_z==-1]=0
            y_z = copy.deepcopy(y[k])
            y_z[y_z==-1]=0
            if et == 'mae':
                sample_error = np.abs(h_z - y_z).sum()
                error += sample_error / (y_z.size)
            elif et== 'iou':
                union = np.sum(np.maximum(h_z,y_z)==1)
                interc = np.sum(h_z +y_z == 2)
                error += (1 - interc/union)
        return (error/n_samples) 
    
    def get_batches(self, imgX,imgY, batch_size, img_size):
        np.random.shuffle(imgX)
        np.random.shuffle(imgY)
        num_batches = img_size // batch_size

        imgX_batches = np.array_split(imgX, num_batches)
        imgY_batches = np.array_split(imgY, num_batches)
        return imgX_batches, imgY_batches
    
    def run_window_convolve(self, sample, sample_size, W_matrices, Wlast, layer, bias):
        Wsample = []
        sample_b = np.pad(sample, ((0, 0), (self.increase, self.increase), (self.increase, self.increase)), mode='constant', constant_values=-1)
        for i in range(self.nlayer):
        #for k in range(sample_size):
            Wsample_k = [] 
            #for i in range(self.nlayer):
           #cupy_kernel = cp.array(W_matrices[i])
            for k in range(sample_size):
                if layer > i:
                    Wsample_k.append(Wlast[i][k])
                elif i==0:
                   Wsample_k.append(self.apply_convolve(sample_b[k], W_matrices[i], bias[i]))
                    #Wsample_k.append(self.apply_convolve_sc2(sample_b[k], cupy_kernel, bias[i]))
                else:
                    Wsample_k.append(self.apply_convolve(Wsample_b[k], W_matrices[i], bias[i]))
                    #Wsample_k.append(self.apply_convolve_sc2(Wsample_b[k], cupy_kernel, bias[i]))

            Wsample_b = np.pad(Wsample_k, ((0, 0), (self.increase, self.increase), (self.increase, self.increase)), mode='constant', constant_values=-1)
            Wsample.append(Wsample_k)
        return Wsample
    
    def run_window_convolve2(self, sample, sample_size, W_matrices, Wlast, layer, bias):
        Wsample = []
        for k in range(sample_size):
            Wsample_k = [] 
            for i in range(self.nlayer):
                if layer > i:
                    Wsample_k.append(Wlast[k][i])
                elif i==0:
                    Wsample_k.append(self.apply_convolve(sample[k], W_matrices[i], bias[i]))
                else:
                    Wsample_k.append(self.apply_convolve(Wsample_k[i-1], W_matrices[i], bias[i]))
            Wsample.append(Wsample_k)
        return Wsample
    
    def apply_convolve(self, img, W_matrices, bias):
        #img_c = np.zeros([img.shape[0]-2*self.increase, img.shape[1]-2*self.increase], dtype=float)
        img_c = np.zeros_like(img, dtype=float)
        for kernel in W_matrices:
            #img_b = cv2.copyMakeBorder(img, self.increase, self.increase, self.increase, self.increase, cv2.BORDER_CONSTANT, None, value = -1) 
            #img_b = np.pad(img, ((self.increase, self.increase), (self.increase, self.increase)), mode='constant', constant_values=-1)
            img_r = cv2.filter2D(img, -1, kernel)-bias
            np.maximum(img_r, 0, out=img_r)
            img_c += img_r
        np.place(img_c, img_c == 0, -1)
        return img_c[self.increase:img_r.shape[0]-self.increase, self.increase:img_r.shape[1]-self.increase]
    
    def apply_convolve_sc(self, img, W_matrices, bias):
        img_c = np.zeros_like(img, dtype=float)
        for kernel in W_matrices:
        #img_c = np.zeros([img.shape[0]-2*self.increase, img.shape[1]-2*self.increase], dtype=float)

            #img_b = cv2.copyMakeBorder(img, self.increase, self.increase, self.increase, self.increase, cv2.BORDER_CONSTANT, None, value = -1) 
            #img_b = np.pad(img, ((self.increase, self.increase), (self.increase, self.increase)), mode='constant', constant_values=-1)
            img_r = convolve2d(img, kernel, mode='same')- bias 
            np.maximum(img_r, 0, out=img_r)
            img_c += img_r
        np.place(img_c, img_c == 0, -1)
        return img_c[self.increase:img_r.shape[0]-self.increase, self.increase:img_r.shape[1]-self.increase]




    def apply_convolve2(self, img, W_matrices, bias):
        #img_c = np.zeros([img.shape[0]-2*self.increase, img.shape[1]-2*self.increase], dtype=float)
        img_c = np.zeros_like(img, dtype=float)
        for kernel in W_matrices:
            #img_b = cv2.copyMakeBorder(img, self.increase, self.increase, self.increase, self.increase, cv2.BORDER_CONSTANT, None, value = -1) 
            #img_b = np.pad(img, ((self.increase, self.increase), (self.increase, self.increase)), mode='constant', constant_values=-1)
            img_r = cv2.filter2D(img, -1, kernel)
            img_r = img_r-bias
            img_r = (img_r > 0).astype(float)
            img_c += img_r[self.increase:img_r.shape[0]-self.increase, self.increase:img_r.shape[1]-self.increase]
        img_c[img_c == 0] = -1
        return img_c

    

    
    def calculate_neighbors(self,W,  joint, k, Wlast, img,yimg, bias, i):
        '''
            Calculate the function window neighbor
        '''
        start_time = time()
        joint_temp = copy.deepcopy(joint)
        if joint[k][i][1] == '1':
            joint_temp[k][i][1] = '0'
        else:
            joint_temp[k][i][1] = '1'
        W_matrices = self.create_w_matrices(W, joint_temp)
        _,error_hood = self.window_error_generate_c(W_matrices, img, self.batch, yimg, self.error_type, Wlast, k, bias)
        end_time = time()
        self.time_test_neightbor.append((end_time - start_time))
        return [error_hood, joint_temp, str(k)+str(i)]

    def calculate_neighbors2(self,W,  joint, k, Wlast, img,yimg, bias, i):
        '''
            Calculate the function window neighbor
        '''
        joint_temp = copy.deepcopy(joint)
        if joint[k][i][1] == '1':
            joint_temp[k][i][1] = '0'
        else:
            joint_temp[k][i][1] = '1'
        W_matrices = self.create_w_matrices(W, joint_temp)
        _,error_hood = self.window_error_generate_c(W_matrices, img, self.batch, yimg, self.error_type, Wlast, k, bias)
        self.error_ep_f_hist.append([error_hood, joint_temp, str(k)+str(i)])
        #print('inside: ',len(self.error_ep_f_hist))
        #return [error_hood, joint_temp, str(k)+str(i)]  
        return 0    

    def calculate_neighbors_parallel(self,W,  joint, Wlast, img,yimg, bias, neighbors_to_visit, ix_layer,neighbor_it):
        '''
            Calculate the function window neighbor
            for the parallel case with parallel - layer
        '''
        k = ix_layer[neighbor_it]
        i = neighbors_to_visit[neighbor_it]
        joint_temp = copy.deepcopy(joint)
        if joint[k][i][1] == '1':
            joint_temp[k][i][1] = '0'
        else:
            joint_temp[k][i][1] = '1'

        W_matrices = self.create_w_matrices(W, joint_temp)
        _,error_hood = self.window_error_generate_c(W_matrices, img, self.batch, yimg, self.error_type, Wlast, k, bias)
        return [error_hood, joint_temp, str(k)+str(i)] 

    def get_error_window(self,W, joint, ep_w):
        '''
            Find the function with the smallest error
            Sequential mode
        '''
        W_matrices = self.create_w_matrices(W, joint)
        bias = np.nansum(W, axis=1) - 1

        if self.batch>=self.train_size:
            train_b = [copy.deepcopy(self.train)]
            ytrain_b = [copy.deepcopy(self.ytrain)]
        flg = 0
        epoch_min = 0
        W_size = [np.count_nonzero(layer == 1) for layer in W]
        Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
        self.time_test = {"ep":[],"time":[]}

        self.time_test_neightbor = []
        for ep in range(self.epoch_f):
            start_time = time()
            if self.batch<self.train_size:
                train_b, ytrain_b = self.get_batches(self.train,self.ytrain, self.batch, self.train_size)
            for b in range(self.num_batches):
                error_ep_f_hist = []
                Wtrain = self.run_window_convolve(train_b[b], self.batch, W_matrices, 0, 0, bias)
                for k in range(self.nlayer):
                    if (not self.neighbors_sample) | (self.neighbors_sample>=len(joint[k])):
                        neighbors_to_visit = range(len(joint[k]))
                    else:
                        neighbors_to_visit = random.sample(range(len(joint[k])), self.neighbors_sample)
                        
                    for nv in neighbors_to_visit: 
                        error_ep_f_hist.append(self.calculate_neighbors(W,  joint, k, Wtrain, train_b[b],ytrain_b[b],bias, nv))

                error_ep_f_hist_np = np.array(error_ep_f_hist, dtype=object)
            
                ix = np.lexsort((error_ep_f_hist_np[:, 2], error_ep_f_hist_np[:, 0]))[0]
                _,joint, _ = error_ep_f_hist_np[ix]               

            W_matrices = self.create_w_matrices(W, joint)
            Wtrain_min,w_error_min =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(w_error_min) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i])

            if w_error_min < w_error:
                w_error = w_error_min
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep

            if (ep-epoch_min)>self.early_stop_round_f :
                break
            self.time_test['ep'].append(ep)
            self.time_test['time'].append((time() - start_time))  
        if flg==1:
            joint = copy.deepcopy(joint_min)
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)
    
    def get_error_window_gpu(self,W, joint, ep_w):
        '''
            Find the function with the smallest error
            Sequential mode
        '''
        W_matrices = self.create_w_matrices(W, joint)
        bias = np.nansum(W, axis=1) - 1

        if self.batch>=self.train_size:
            train_b = [copy.deepcopy(self.cp_train)]
            ytrain_b = [copy.deepcopy(self.ytrain)]
        flg = 0
        epoch_min = 0
        W_size = [np.count_nonzero(layer == 1) for layer in W]
        Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
        self.time_test = {"ep":[],"time":[]}

        self.time_test_neightbor = []
        for ep in range(self.epoch_f):
            start_time = time()
            if self.batch<self.train_size:
                train_b, ytrain_b = self.get_batches(self.cp_train,self.ytrain, self.batch, self.train_size)
            for b in range(self.num_batches):
                error_ep_f_hist = []
                Wtrain = self.run_window_convolve(train_b[b], self.batch, W_matrices, 0, 0, bias)
                for k in range(self.nlayer):
                    if (not self.neighbors_sample) | (self.neighbors_sample>=len(joint[k])):
                        neighbors_to_visit = range(len(joint[k]))
                    else:
                        neighbors_to_visit = random.sample(range(len(joint[k])), self.neighbors_sample)
                        
                    for nv in neighbors_to_visit: 
                        error_ep_f_hist.append(self.calculate_neighbors(W,  joint, k, Wtrain, train_b[b],ytrain_b[b],bias, nv))

                error_ep_f_hist_np = np.array(error_ep_f_hist, dtype=object)
            
                ix = np.lexsort((error_ep_f_hist_np[:, 2], error_ep_f_hist_np[:, 0]))[0]
                _,joint, _ = error_ep_f_hist_np[ix]               

            W_matrices = self.create_w_matrices(W, joint)
            Wtrain_min,w_error_min =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(w_error_min) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i])

            if w_error_min < w_error:
                w_error = w_error_min
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep

            if (ep-epoch_min)>self.early_stop_round_f :
                break
            self.time_test['ep'].append(ep)
            self.time_test['time'].append((time() - start_time))  
        if flg==1:
            joint = copy.deepcopy(joint_min)
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)
    
    def get_error_window_parallel(self,W, joint, ep_w):
        '''
            Find the function with the smallest error
            Parallel mode - only neighbors
        '''
        
        W_matrices = self.create_w_matrices(W, joint)
        bias = np.nansum(W, axis=1) - 1

        if self.batch>=self.train_size:
            train_b = [copy.deepcopy(self.train)]
            ytrain_b = [copy.deepcopy(self.ytrain)]
        flg = 0
        epoch_min = 0
        W_size = [np.count_nonzero(layer == 1) for layer in W]    
        Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
        self.time_test = {"ep":[],"time":[]}
        self.time_test_neightbor = []
        for ep in range(self.epoch_f):
            start_time = time()
            if self.batch<self.train_size:
                train_b, ytrain_b = self.get_batches(self.train,self.ytrain, self.batch, self.train_size)
            for b in range(self.num_batches):
                error_ep_f_hist = [] #np.array([])
                Wtrain = self.run_window_convolve(train_b[b], self.batch, W_matrices, 0, 0, bias)
                for k in range(self.nlayer):
                    if (not self.neighbors_sample) | (self.neighbors_sample>=len(joint[k])):
                        neighbors_to_visit = range(len(joint[k]))
                    else:
                        neighbors_to_visit = random.sample(range(len(joint[k])), self.neighbors_sample)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        fixed_params_func = lambda neighbor: self.calculate_neighbors(W,  joint, k, Wtrain, train_b[b],ytrain_b[b],bias, neighbor)
                        error_ep_f_hist_ = list(executor.map(fixed_params_func, neighbors_to_visit))
                    
                    error_ep_f_hist.extend(error_ep_f_hist_)
                error_ep_f_hist_np = np.array(error_ep_f_hist, dtype=object)
            
                ix = np.lexsort((error_ep_f_hist_np[:, 2], error_ep_f_hist_np[:, 0]))[0]
                _,joint, _ = error_ep_f_hist_np[ix]               
              
            W_matrices = self.create_w_matrices(W, joint)
            Wtrain_min,w_error_min =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(w_error_min) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i])

            if w_error_min < w_error:
                w_error = w_error_min
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep

            if (ep-epoch_min)>self.early_stop_round_f :
                break

            self.time_test['ep'].append(ep)
            self.time_test['time'].append((time() - start_time))  
        if flg==1:
            joint = copy.deepcopy(joint_min)
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)
    
    def get_error_window_parallel2(self,W, joint, ep_w):
        '''
            Find the function with the smallest error
            Parallel mode - only neighbors
        '''
        W_matrices = self.create_w_matrices(W, joint)
        bias = np.nansum(W, axis=1) - 1

        if self.batch>=self.train_size:
            train_b = [copy.deepcopy(self.train)]
            ytrain_b = [copy.deepcopy(self.ytrain)]
        flg = 0
        epoch_min = 0
        W_size = [np.count_nonzero(layer == 1) for layer in W]    
        Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)

        for ep in range(self.epoch_f):
            if self.batch<self.train_size:
                train_b, ytrain_b = self.get_batches(self.train,self.ytrain, self.batch, self.train_size)
            for b in range(self.num_batches):
                self.error_ep_f_hist = [] #np.array([],dtype=object)
                Wtrain = self.run_window_convolve(train_b[b], self.batch, W_matrices, 0, 0, bias)
                for k in range(self.nlayer):
                    if (not self.neighbors_sample) | (self.neighbors_sample>=len(joint[k])):
                        neighbors_to_visit = range(len(joint[k]))
                    else:
                        neighbors_to_visit = random.sample(range(len(joint[k])), self.neighbors_sample)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        fixed_params_func = lambda neighbor: self.calculate_neighbors2(W,  joint, k, Wtrain, train_b[b],ytrain_b[b],bias, neighbor)
                        ll = list(executor.map(fixed_params_func, neighbors_to_visit))
                error_ep_f_hist_np = np.array(self.error_ep_f_hist, dtype=object)
                ix = np.lexsort((error_ep_f_hist_np[:, 2], error_ep_f_hist_np[:, 0]))[0]
                _,joint, _ = self.error_ep_f_hist[ix]               
                
            W_matrices = self.create_w_matrices(W, joint)
            Wtrain_min,w_error_min =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(w_error_min) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i])

            if w_error_min < w_error:
                w_error = w_error_min
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep

            if (ep-epoch_min)>self.early_stop_round_f :
                break
        if flg==1:
            joint = copy.deepcopy(joint_min)
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)
    
    def get_random_neighbors(self, joint):
        neighbors_to_visit = []
        ix = []
        for k, sublist in enumerate(joint):
            sublist_length = len(sublist)
            if not self.neighbors_sample or self.neighbors_sample >= sublist_length:
                neighbors_to_visit.extend(range(sublist_length))
                ix.extend([k] * sublist_length)
            else:
                sample_indices = random.sample(range(sublist_length), self.neighbors_sample)
                neighbors_to_visit.extend(sample_indices)
                ix.extend([k] * len(sample_indices))
        return neighbors_to_visit, ix

    def get_error_window_parallel_layer(self,W, joint, ep_w):
        '''
            Find the function with the smallest error
            Parallel mode - with layers
        '''
        W_matrices = self.create_w_matrices(W, joint)
        bias = np.nansum(W, axis=1) - 1

        if self.batch>=self.train_size:
            train_b = [copy.deepcopy(self.train)]
            ytrain_b = [copy.deepcopy(self.ytrain)]
        flg = 0
        epoch_min = 0
        W_size = [np.count_nonzero(layer == 1) for layer in W]
        Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)

        for ep in range(self.epoch_f):
            if self.batch<self.train_size:
                train_b, ytrain_b = self.get_batches(self.train,self.ytrain, self.batch, self.train_size)
            for b in range(self.num_batches):
                error_ep_f_hist = [] 
                Wtrain = self.run_window_convolve(train_b[b], self.batch, W_matrices, 0, 0, bias)
                neighbors_to_visit, ix_layer = self.get_random_neighbors(joint)
             
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    fixed_params_func = lambda neighbor_it: self.calculate_neighbors_parallel(W,  joint, Wtrain, train_b[b],ytrain_b[b],bias, neighbors_to_visit, ix_layer,neighbor_it)
                    error_ep_f_hist = list(executor.map(fixed_params_func, range(len(neighbors_to_visit))))

                error_ep_f_hist_np = np.array(error_ep_f_hist, dtype=object)
            
                ix = np.lexsort((error_ep_f_hist_np[:, 2], error_ep_f_hist_np[:, 0]))[0]
                _,joint, _ = error_ep_f_hist_np[ix]               

            W_matrices = self.create_w_matrices(W, joint)
            Wtrain_min,w_error_min =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias)
            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(w_error_min) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i])

            if w_error_min < w_error:
                w_error = w_error_min
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep

            if (ep-epoch_min)>self.early_stop_round_f :
                break
        if flg==1:
            joint = copy.deepcopy(joint_min)
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)

    
    def check_neighboors(self, W,joint, ep_w,error_ep):
        '''
            Find the smallest function of all Window-neighbor
            Sequential mode
        '''
        for k,i in product(range(self.nlayer), range(len(W[0]))):
            error_ep.append(self.neighboors_func(W, joint,ep_w,k, i))
        return error_ep
    
    def check_neighboors_func_parallel(self, W,joint, ep_w,error_ep):
        '''
            Find the smallest function of all Window-neighbor
            Window function - parallel
        '''
        for k,i in product(range(self.nlayer), range(len(W[0]))):
            error_ep.append(self.neighboors_func_parallel(W, joint,ep_w,k, i))
        return error_ep
    
    def check_neighboors_func_parallel2(self, W,joint, ep_w,error_ep):
        '''
            Find the smallest function of all Window-neighbor
            Window function - parallel
        '''
        for k,i in product(range(self.nlayer), range(len(W[0]))):
            error_ep.append(self.neighboors_func_parallel2(W, joint,ep_w,k, i))
        return error_ep
    
    def check_neighboors_func_parallel_layer(self, W,joint, ep_w,error_ep):
        '''
            Find the smallest function of all Window-neighbor
            Window function - parallel with layer
        '''
        for k,i in product(range(self.nlayer), range(len(W[0]))):
            error_ep.append(self.neighboors_func_parallel_layer(W, joint,ep_w,k, i))
        return error_ep
    
    def check_neighboors_parallel(self, W,joint, ep_w,error_ep):
        '''
            Find the smallest function of all Window-neighbor
            Parallel mode
        '''
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fixed_params_func = lambda neighbor: self.neighboors_func(W, joint,ep_w,neighbor[0],neighbor[1])
            error_ep = list(executor.map(fixed_params_func, product(range(self.nlayer), range(len(W[0])))))
        return error_ep
        
    def neighboors_func(self, W, joint, ep_w, k, i):#, type):
        W_line_temp = copy.deepcopy(W[k])
        if np.isnan(W_line_temp[i]):
            W_line_temp[i] = 1
        else:
            W_line_temp[i] = np.nan
        W_line_temp_NN = copy.deepcopy(W_line_temp)
        W_line_temp_NN[np.isnan(W_line_temp_NN)] = 0
        W_line_temp_NN = W_line_temp_NN.astype(int)
            
        if ''.join(W_line_temp_NN.astype(str)) in self.windows_continuos:
            W_temp = copy.deepcopy(W)
            W_temp[k] = W_line_temp
            
            Wh_in_w_hist = any(
                all(np.allclose(w1, w2, equal_nan=True) for w1, w2 in zip(sublist, W_temp))
                for sublist in self.w_hist["W"]
            )
            if not Wh_in_w_hist:
                
                self.windows_visit+=1
                joint_temp = copy.deepcopy(joint)
                joint_temp[k] = self.create_joint(W_temp[k])

                joint_temp, w_error, f_epoch_min = self.get_error_window(W_temp, joint_temp, ep_w)

                self.w_hist["W_key"].append(self.windows_visit)
                self.w_hist["W"].append(W_temp)
                self.w_hist["error"].append(w_error)
                self.w_hist["f_epoch_min"].append(f_epoch_min)
                return [w_error[0], w_error[1], W_temp, joint_temp, str(k)+str(i)]
        return [np.inf, np.inf, np.nan, np.nan, str(k)+str(i)]
    
    def neighboors_func_parallel(self, W, joint, ep_w, k, i):#, type):
        W_line_temp = copy.deepcopy(W[k])
        if np.isnan(W_line_temp[i]):
            W_line_temp[i] = 1
        else:
            W_line_temp[i] = np.nan
        W_line_temp_NN = copy.deepcopy(W_line_temp)
        W_line_temp_NN[np.isnan(W_line_temp_NN)] = 0
        W_line_temp_NN = W_line_temp_NN.astype(int)
            
        if ''.join(W_line_temp_NN.astype(str)) in self.windows_continuos:
            W_temp = copy.deepcopy(W)
            W_temp[k] = W_line_temp
            
            Wh_in_w_hist = any(
                all(np.allclose(w1, w2, equal_nan=True) for w1, w2 in zip(sublist, W_temp))
                for sublist in self.w_hist["W"]
            )
            if not Wh_in_w_hist:
                
                self.windows_visit+=1
                joint_temp = copy.deepcopy(joint)
                joint_temp[k] = self.create_joint(W_temp[k])

                joint_temp, w_error, f_epoch_min = self.get_error_window_parallel(W_temp, joint_temp, ep_w)

                self.w_hist["W_key"].append(self.windows_visit)
                self.w_hist["W"].append(W_temp)
                self.w_hist["error"].append(w_error)
                self.w_hist["f_epoch_min"].append(f_epoch_min)
                return [w_error[0], w_error[1], W_temp, joint_temp, str(k)+str(i)]
        return [np.inf, np.inf, np.nan, np.nan, str(k)+str(i)]
    
    def neighboors_func_parallel2(self, W, joint, ep_w, k, i):#, type):
        W_line_temp = copy.deepcopy(W[k])
        if np.isnan(W_line_temp[i]):
            W_line_temp[i] = 1
        else:
            W_line_temp[i] = np.nan
        W_line_temp_NN = copy.deepcopy(W_line_temp)
        W_line_temp_NN[np.isnan(W_line_temp_NN)] = 0
        W_line_temp_NN = W_line_temp_NN.astype(int)
            
        if ''.join(W_line_temp_NN.astype(str)) in self.windows_continuos:
            W_temp = copy.deepcopy(W)
            W_temp[k] = W_line_temp
            
            Wh_in_w_hist = any(
                all(np.allclose(w1, w2, equal_nan=True) for w1, w2 in zip(sublist, W_temp))
                for sublist in self.w_hist["W"]
            )
            if not Wh_in_w_hist:
                
                self.windows_visit+=1
                joint_temp = copy.deepcopy(joint)
                joint_temp[k] = self.create_joint(W_temp[k])

                joint_temp, w_error, f_epoch_min = self.get_error_window_parallel2(W_temp, joint_temp, ep_w)

                self.w_hist["W_key"].append(self.windows_visit)
                self.w_hist["W"].append(W_temp)
                self.w_hist["error"].append(w_error)
                self.w_hist["f_epoch_min"].append(f_epoch_min)
                return [w_error[0], w_error[1], W_temp, joint_temp, str(k)+str(i)]
        return [np.inf, np.inf, np.nan, np.nan, str(k)+str(i)]
    
    def neighboors_func_parallel_layer(self, W, joint, ep_w, k, i):#, type):
        W_line_temp = copy.deepcopy(W[k])
        if np.isnan(W_line_temp[i]):
            W_line_temp[i] = 1
        else:
            W_line_temp[i] = np.nan
        W_line_temp_NN = copy.deepcopy(W_line_temp)
        W_line_temp_NN[np.isnan(W_line_temp_NN)] = 0
        W_line_temp_NN = W_line_temp_NN.astype(int)
            
        if ''.join(W_line_temp_NN.astype(str)) in self.windows_continuos:
            W_temp = copy.deepcopy(W)
            W_temp[k] = W_line_temp
            
            Wh_in_w_hist = any(
                all(np.allclose(w1, w2, equal_nan=True) for w1, w2 in zip(sublist, W_temp))
                for sublist in self.w_hist["W"]
            )
            if not Wh_in_w_hist:
                
                self.windows_visit+=1
                joint_temp = copy.deepcopy(joint)
                joint_temp[k] = self.create_joint(W_temp[k])

                joint_temp, w_error, f_epoch_min = self.get_error_window_parallel_layer(W_temp, joint_temp, ep_w)

                self.w_hist["W_key"].append(self.windows_visit)
                self.w_hist["W"].append(W_temp)
                self.w_hist["error"].append(w_error)
                self.w_hist["f_epoch_min"].append(f_epoch_min)
                return [w_error[0], w_error[1], W_temp, joint_temp, str(k)+str(i)]
        return [np.inf, np.inf, np.nan, np.nan, str(k)+str(i)]


    def fit(self):
        self.start_time = time()
        
        if self.parallel == 'parallel-func':
                joint, error,f_epoch_min = self.get_error_window_parallel(self.W,self.joint,0)
        elif self.parallel == 'parallel-func-layer':
            joint, error,f_epoch_min = self.get_error_window_parallel_layer(self.W,self.joint,0)
        else:
            joint, error,f_epoch_min = self.get_error_window(self.W,self.joint,0)

        self.w_hist["W_key"].append(self.windows_visit)
        #self.w_hist["W"].append(self.window_history(self.W, self.nlayer, self.wsize))
        self.w_hist["W"].append(self.W)
        self.w_hist["error"].append(error)
        self.w_hist["f_epoch_min"].append(f_epoch_min)
            
        
        ep_min=0

        W = copy.deepcopy(self.W)
        error_min = copy.deepcopy(error)
        W_min = copy.deepcopy(W)
        joint_min = copy.deepcopy(joint)
        error_ep = {"epoch":[],"error_train":[], "error_val":[]}
        error_ep['epoch'].append(0)
        error_ep['error_train'].append(error[0])
        error_ep['error_val'].append(error[1])
        
        time_min = (time() -  self.start_time) / 60
        print(f'Time: {time_min:.2f} | Epoch 0 / {self.epoch_w} - start Validation error: ', error[1])

        for ep in range(1,self.epoch_w+1):

            error_ep_ = []
            if self.parallel == 'no-parallel':
                error_ep_ = np.array(self.check_neighboors(W,joint,ep, error_ep_),dtype=object)
            elif self.parallel == 'parallel-window':
                error_ep_ = np.array(self.check_neighboors_parallel(W,joint,ep, error_ep_),dtype=object)
            elif self.parallel == 'parallel-func':
                error_ep_ = np.array(self.check_neighboors_func_parallel(W,joint,ep, error_ep_),dtype=object)
            elif self.parallel == 'parallel-func-layer':
                error_ep_ = np.array(self.check_neighboors_func_parallel_layer(W,joint,ep, error_ep_),dtype=object)
            else:
                print('Error: Parallel mode not found')
                break
            
            if error_ep_[0][0]:
                ix = np.lexsort((error_ep_[:, 4], error_ep_[:, 1]))[0]
                error_train_min, error_val_min, W, joint, _ = error_ep_[ix]               
                error = np.array([error_train_min,error_val_min])

            if error[1]<error_min[1]:
                ep_min = ep
                self.save_window(joint, W)
                error_min = copy.deepcopy(error)
                W_min = copy.deepcopy(W)
                joint_min = copy.deepcopy(joint)
                W_matrices = self.create_w_matrices(W_min,joint_min)
                bias = np.nansum(W, axis=1) - 1
                Wtrain = self.run_window_convolve(self.train, self.train_size,W_matrices, 0, 0, bias)
                Wval = self.run_window_convolve(self.val, self.val_size,W_matrices, 0, 0, bias)
                Wtest = self.run_window_convolve(self.test, self.test_size,W_matrices, 0, 0, bias)
                self.save_results_complet_in(Wtrain, Wval, Wtest, f'_epoch{ep}')
            
            error_ep['epoch'].append(ep)
            error_ep['error_train'].append(error[0])
            error_ep['error_val'].append(error[1])
            
            self.save_to_csv(error_ep, self.path_results+'/run/error_ep_w'+self.name_save+'_epoch'+str(ep))
            self.save_to_csv(self.error_ep_f, self.path_results+'/run/error_ep_f'+self.name_save+'_epoch'+str(ep))

            time_min = (time() -  self.start_time) / 60
            print(f'Time: {time_min:.2f} | Epoch {ep} / {self.epoch_w} - Validation error: {error[1]}')
            print(f'Janela: {W}, época minina = {ep_min}')
            if (ep-ep_min)>self.early_stop_round_w :
                print('End by Early Stop Round')
                break
                
        print('----------------------------------------------------------------------') 
        Wtest,error_test =  self.window_error_generate_c(W_matrices, self.test, self.test_size, self.ytest, self.error_type, 0, 0, bias)
        Wtrain,error_train =  self.window_error_generate_c(W_matrices, self.train, self.train_size, self.ytrain, self.error_type, 0, 0,bias)
        Wval,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, 0, 0, bias)

        print('End of testing')
        end_time = time()
        time_min = (end_time -  self.start_time) / 60
        print(f'Time: {time_min:.2f} | Min-Epoch {ep_min} / {self.epoch_w} - Train error: {error_train} / Validation error: {error_val} / Test error: {error_test}')

        self.save_results_complet(Wtrain, Wval, Wtest)
        pickle.dump(self.w_hist, open(f'{self.path_results}/W_hist{self.name_save}.txt', 'wb'))
        pickle.dump(error_ep, open(f'{self.path_results}/error_ep_w{self.name_save}.txt', 'wb'))
        pickle.dump(self.error_ep_f, open(f'{self.path_results}/error_ep_f{self.name_save}.txt', 'wb'))
        print(f"Janela Final Aprendida: {W_min}")
        #print(f"Joint Final Aprendida: {joint_min}")  

    def save_window(self, joint, W):
        filename_joint = f'{self.path_results}/joint{self.name_save}.txt'
        pickle.dump(joint, open(filename_joint, 'wb'))
        filename_W =f'{self.path_results}/W{self.name_save}.txt'
        pickle.dump(W, open(filename_W, 'wb'))  

    def save_results_complet(self, Wtrain, Wval, Wtest):
        for k in range(self.nlayer):
            self._save_results(Wtrain, 'train',k)
            self._save_results(Wval, 'val',k)
            self._save_results(Wtest, 'test',k)

    def _save_results(self, W, img_type, k):
        for img in range(1,len(W)+1):
            x = copy.deepcopy(W[img-1][k])
            x[(x==-1)]=255
            x[(x==1)]=0
            cv2.imwrite(f'{self.path_results}/{img_type}_op{k+1}_{img:02d}{self.name_save}.jpg', x)

    def save_results_complet_in(self, Wtrain, Wval, Wtest, ep = None):
        for k in range(self.nlayer):
            self._save_results_in(Wtrain, 'train',k, ep)
            self._save_results_in(Wval, 'val',k, ep)
            self._save_results_in(Wtest, 'test',k, ep)

    def _save_results_in(self, W, img_type, k, ep = None):
        for img in range(1,len(W)+1):
            x = copy.deepcopy(W[img-1][k])
            x[(x==-1)]=255
            x[(x==1)]=0
            cv2.imwrite(f'{self.path_results}/run/{img_type}_op{k+1}_{img:02d}{self.name_save}{ep}.jpg', x)


    def window_history(self, W, nlayer, wsize):
        for k in range(nlayer):
            #print('---')
            #print('W -k = ', W[k])
            #print('k = ', k)
            #print('---')
            if k==0:
                try:

                    window_hist = ''.join([''.join(item) for item in np.reshape(W[k], (wsize,)).astype(str)])
                    #window_hist = ''.join([np.array_str(item) for item in np.reshape(W[k], (wsize,))])
                except Exception as e:
                    print("Erro:", e)
                    print("Valor de k:", k)
                    print("Valor de W[k]:", W[k])
            else:
                #window_hist = window_hist + ''.join([np.array_str(item) for item in np.reshape(W[k], (wsize,))])
                try:
                    window_hist = window_hist+''.join([''.join(item) for item in np.reshape(W[k], (wsize,)).astype(str)])
                except Exception as e:
                    print("Erro:", e)
                    print("Valor de k:", k)
                    print("Valor de W[k]:", W[k])
        return window_hist 

    #-----------------------------------------

    

    def create_window(self):
        wind =  random.randint(0, len(self.windows_continuos)-1)
        W = np.array([1. if i=='1' else np.nan for i in self.windows_continuos[wind]])
        return W
    

    def save_to_csv(self, data, name):
        df = pd.DataFrame(data)
        df.to_csv(f'{name}.csv', index=False)
    

    
    def results_after_fit(self, path_results, name_save):
        joint = np.load(f'{path_results}/joint{name_save}.txt', allow_pickle=True)
        W = np.load(f'{path_results}/W{name_save}.txt', allow_pickle=True)

        print(f'Reading from: {path_results}')

        W_matrices = self.create_w_matrices(W,joint)
        bias = np.nansum(W, axis=1) - 1

        Wtest,error_test =  self.window_error_generate_c(W_matrices, self.test, self.test_size, self.ytest, self.error_type, 0, 0, bias)
        Wtrain,error_train =  self.window_error_generate_c(W_matrices, self.train, self.train_size, self.ytrain, self.error_type, 0, 0,bias)
        Wval,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, 0, 0, bias)

        print(f'Train error: {error_train} / Validation error: {error_val} / Test error: {error_test}')

        self.save_results_complet(Wtrain, Wval, Wtest)


    def test_neightbors(self):
        start_time = time()
        #error_ep = {"error_train":[],"error_val":[],"W":[],"joint":[]}
        error_ep = []
        #error_ep = np.array(self.check_neighboors(self.W,self.joint,0, error_ep),dtype=object)
        #error_ep = np.array(self.check_neighboors_parallel(self.W,self.joint,0, error_ep),dtype=object)
        #error_ep = np.array(self.check_neighboors_func_parallel(self.W,self.joint,0, error_ep),dtype=object)
        error_ep = np.array(self.check_neighboors_func_parallel2(self.W,self.joint,0, error_ep),dtype=object)
        #error_ep = np.array(self.check_neighboors_func_parallel_layer(self.W,self.joint,0, error_ep),dtype=object)
        
        print('error list: ', error_ep[:, 1])
        print('indices: ', error_ep[:, 4])

        if error_ep[0][0]:
            ix = np.lexsort((error_ep[:, 4], error_ep[:, 1]))[0]
            #print('ix: ', ix)
            error_train_min, error_val_min, W, joint, _ = error_ep[ix]
            #print('error_val_min: ', error_val_min)
            
            #print(error_ep[ix])
            error = np.array([error_train_min,error_val_min])

        print(f'error_train: ', error[0])
        print(f'error_val: ', error[1])
        print(f'W: ', W)
        end_time = time()
        time_min = (end_time -  start_time)
        print(f'Time: {time_min:.2f}')
    
    def test_window(self):
        start = time()

        if self.parallel == 'no-parallel':
             _, error,f_epoch_min = self.get_error_window(self.W,self.joint,0)
        elif self.parallel == 'parallel-window':
             _, error,f_epoch_min = self.get_error_window(self.W,self.joint,0)
        elif self.parallel == 'parallel-func':
            _, error,f_epoch_min = self.get_error_window_parallel(self.W,self.joint,0)
        elif self.parallel == 'parallel-func-layer':
            _, error,f_epoch_min = self.get_error_window_parallel_layer(self.W,self.joint,0)
        elif self.parallel == 'gpu':
            _, error,f_epoch_min = self.get_error_window_gpu(self.W,self.joint,0)
        else:
            print('Error: Parallel mode not found')


        end = time()
        print('tempo de execução: {}'.format(end - start))
        print(f'época-min: {f_epoch_min} - com erro: {error}')
        print(pd.DataFrame(self.time_test))
        print('Tempo Médio por época: ', np.mean(self.time_test['time']))
        print('Tempo média por vizinho: ', np.mean(self.time_test_neightbor))
    
    def compare_times_window(self):
        start = time()
        _, error,f_epoch_min = self.get_error_window_parallel(self.W,self.joint,0)
        
        end = time()
        print('tempo de execução paralelo 1: {}'.format(end - start))
        print(f'época-min: {f_epoch_min} - com erro: {error}')
        
        start = time()
        _, error,f_epoch_min = self.get_error_window_parallel_layer(self.W,self.joint,0)
        
        end = time()
        print('tempo de execução paralelo 2: {}'.format(end - start))
        print(f'época-min: {f_epoch_min} - com erro: {error}')

        start = time()
        _, error,f_epoch_min = self.get_error_window(self.W,self.joint,0)

        end = time()
        print('tempo de execução sequencial: {}'.format(end - start))
        print(f'época-min: {f_epoch_min} - com erro: {error}')
    
    def compare_times_neighbor(self):
        # sequential mode:
        start_time = time()
        error_ep = []
        error_ep = np.array(self.check_neighboors(self.W,self.joint,0, error_ep),dtype=object)

        if error_ep[0][0]:
            ix = np.lexsort((error_ep[:, 4], error_ep[:, 1]))[0]
            error_train_min, error_val_min, W, joint, _ = error_ep[ix]
            error = np.array([error_train_min,error_val_min])

        print(f'error_train: {error[0]} - error_val: { error[1]}')
        print(f'W: ', W)
        end_time = time()
        time_min = (end_time -  start_time)
        print(f'Time sequential: {time_min:.2f}')
        print('-----------------------------')
        #--------------
        # Window parallel mode:
        start_time = time()
        error_ep = []
        error_ep = np.array(self.check_neighboors_parallel(self.W,self.joint,0, error_ep),dtype=object)

        if error_ep[0][0]:
            ix = np.lexsort((error_ep[:, 4], error_ep[:, 1]))[0]
            error_train_min, error_val_min, W, joint, _ = error_ep[ix]
            error = np.array([error_train_min,error_val_min])

        print(f'error_train: {error[0]} - error_val: { error[1]}')
        print(f'W: ', W)
        end_time = time()
        time_min = (end_time -  start_time)
        print(f'Time Window-parallel: {time_min:.2f}')
        print('-----------------------------')
        #--------------
        # Func parallel mode:
        start_time = time()
        error_ep = []
        error_ep = np.array(self.check_neighboors_func_parallel(self.W,self.joint,0, error_ep),dtype=object)

        if error_ep[0][0]:
            ix = np.lexsort((error_ep[:, 4], error_ep[:, 1]))[0]
            error_train_min, error_val_min, W, joint, _ = error_ep[ix]
            error = np.array([error_train_min,error_val_min])

        print(f'error_train: {error[0]} - error_val: { error[1]}')
        print(f'W: ', W)
        end_time = time()
        time_min = (end_time -  start_time)
        print(f'Time Func-parallel: {time_min:.2f}')
        print('-----------------------------')
        #--------------
        # Func parallel-layer mode:
        start_time = time()
        error_ep = []
        error_ep = np.array(self.check_neighboors_func_parallel_layer(self.W,self.joint,0, error_ep),dtype=object)

        if error_ep[0][0]:
            ix = np.lexsort((error_ep[:, 4], error_ep[:, 1]))[0]
            error_train_min, error_val_min, W, joint, _ = error_ep[ix]
            error = np.array([error_train_min,error_val_min])

        print(f'error_train: {error[0]} - error_val: { error[1]}')
        print(f'W: ', W)
        end_time = time()
        time_min = (end_time -  start_time)
        print(f'Time Func-parallel-layer: {time_min:.2f}')
        print('-----------------------------')