import numpy as np
import itertools
import cv2
import random
import copy
from time import sleep, time
import pickle
import concurrent.futures
import os
import pandas as pd

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
        self.windows_continuos = np.load('./data/windows_continuos.txt', allow_pickle=True)
        self.increase = int(round(wlen/2-0.1,0))
        self.count = 0
        self.random_list = self.randon_seed_number(seed)

        
        
        if new:
            Wini = np.array([np.nan, 1., np.nan, 1., 1., 1., np.nan, 1., np.nan])
            self.W = [Wini.copy() for _ in range(self.nlayer)]
            #self.joint = [self.create_joint(self.W[i]) for i in range(self.nlayer)]

        else:
            self.W = np.load('./data/W'+new+'.txt', allow_pickle=True)
        print(f"Janela Inicializada: {self.W}")
        #self.parameter_nn = self.initialize_parameters_nn(self.W)

        self.w_hist = {"W_key": [],"W":[],"error":[], "f_epoch_min":[]}
        self.windows_visit = 1
        self.error_ep_f_hist = {}
        self.error_ep_f = {"W_key": [], "epoch_w": [], "epoch_f":[], "error":[], "time":[] }
        wind_size_dict = {f'window_size_{i}': [] for i in range(self.nlayer)}
        self.error_ep_f = {key: value for d in [self.error_ep_f, wind_size_dict] for key, value in d.items()}

        
        self.train, self.ytrain, self.img_shape = self.get_images_2(train_size, 'train')
        self.val, self.yval, _ = self.get_images_2(val_size, 'val')
        self.test, self.ytest, _ = self.get_images_2(test_size, 'test')

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
        self.joint_hist = []

        self.early_stop_round_f = early_stop_round_f
        self.early_stop_round_w = early_stop_round_w

        self.lr = 0.01

        self.start_time = 0
        print('------------------------------------------------------------------')
        
        

    def randon_seed_number(self, s):
        random.seed(s)
        n_ep = (self.epoch_f*self.epoch_w)*200+len(self.windows_continuos)*2
        random_numbers = random.sample(range(0, 1000000000), n_ep)

        return random_numbers

    def create_window(self):
        random.seed(self.random_list[0])
        wind =  random.randint(0, len(self.windows_continuos)-1)
        W = np.array([1. if i=='1' else np.nan for i in self.windows_continuos[wind]])
        return W

    def create_joint(self,W):
        Ji=[]
        ni = int(W[~np.isnan(W)].sum())
        for i in itertools.product([0, 1], repeat=ni):
            Ji.append(''.join(np.array(i).astype(str)))
        return Ji#, np.random.randint(2, size=len(Ji))
    
    
    def find_minterms(self, truth_table):
        inputs = truth_table[:, 0]
        results = truth_table[:, 1]
        minterms = []
        for i in np.where(results =='1')[0]:
            minterms.append(inputs[i])
        minterms

        return minterms
    
    def create_w_matrices(self, W):
        matrix_k = []
        reduced_minterms = self.create_joint(W)
        for minterm in range(len(reduced_minterms)):
            matrix = np.copy(W)
            c=0
            for i in np.where(W==1)[0]:
                matrix[i] = reduced_minterms[minterm][c]
                c+=1
            matrix = np.where(matrix==0,-1,matrix)
            matrix = np.nan_to_num(matrix)
            matrix_k.append(matrix)
        return matrix_k
    
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
    
    def get_images_2(self, img_size, img_type):
        ximg = []
        yimg = []
        for img in range(1,img_size+1):
            x = cv2.imread(f'./data/x/{img_type}{img:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            y = cv2.imread(f'./data/y/{img_type}{img:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            xbin = self._convert_binary(x)
            xbin_inc = self.increase_zero(xbin)
            ybin = self._convert_binary(y)
            for i in range(xbin.shape[0]):
                for j in range(xbin.shape[1]):
                    window = np.array(xbin_inc[i:i + self.wlen, j:j + self.wlen]).flatten()
                    #window = (xbin_inc[i:i + self.wlen, j:j + self.wlen]).flatten()
                    ximg.append(window)
                    yimg.append(ybin[i, j])
        ximg_np = np.vstack(ximg)
        ximg_np = np.transpose(ximg_np)
        yimg_np = np.array(yimg)
        yimg_np = yimg_np.reshape(1,-1)
        return ximg_np , yimg_np, xbin.shape[0]

    def transform_image_forlayer(self, img, img_size):
        img[img == 0] = -1
        imagem_shape = (self.img_shape, self.img_shape)
        img_reshape = np.reshape(img, (img_size, *imagem_shape))
        img_t = []
        for l in range(img_size):
            img_inc = self.increase_zero(img_reshape[l])
            for i in range(self.img_shape):
                for j in range(self.img_shape):
                    window = np.array(img_inc[i:i + self.wlen, j:j + self.wlen]).flatten()
                    #window = (img_inc[i:i + self.wlen, j:j + self.wlen]).flatten()
                    img_t.append(window)
        img_np = np.vstack(img_t)
        img_np = np.transpose(img_np)
        return img_np

    def increase_zero(self, img):
        img_inc = -np.ones((img.shape[0]+self.increase*2, img.shape[1]+self.increase*2), dtype=img.dtype)
        img_inc[self.increase:img_inc.shape[0]-self.increase, self.increase:img_inc.shape[1]-self.increase] = img
        return img_inc

    
    def _save_results_in(self, W, img_type, k, ep = None):
        for img in range(1,len(W)+1):
            x = copy.deepcopy(W[img-1][k])
            x[(x==-1)]=255
            x[(x==1)]=0
            cv2.imwrite(f'{self.path_results}/run/{img_type}_op{k+1}_{img:02d}{self.name_save}{ep}.jpg', x)
    
    def _save_results(self, W, img_type, k):
        for img in range(1,len(W)+1):
            x = copy.deepcopy(W[img-1][k])
            x[(x==-1)]=255
            x[(x==1)]=0
            cv2.imwrite(f'{self.path_results}/{img_type}_op{k+1}_{img:02d}{self.name_save}.jpg', x)

    def save_results(self, Wtrain, Wval, Wtest):
        self._save_results_in(Wtrain, 'train',1)
        self._save_results_in(Wval, 'val',1)
        self._save_results_in(Wtest, 'test',1)
        
    
    def save_results_complet_in(self, Wtrain, Wval, Wtest, ep = None):
        for k in range(self.nlayer):
            self._save_results_in(Wtrain, 'train',k, ep)
            self._save_results_in(Wval, 'val',k, ep)
            self._save_results_in(Wtest, 'test',k, ep)
    
    def save_results_complet(self, Wtrain, Wval, Wtest):
        for k in range(self.nlayer):
            self._save_results(Wtrain, 'train',k)
            self._save_results(Wval, 'val',k)
            self._save_results(Wtest, 'test',k)
    
    def apply_window(self, x, W_n, j_n):
        Xl = np.c_[np.zeros([x.shape[0], self.increase], dtype=int), x, np.zeros([x.shape[0], self.increase], dtype=int)]
        Xl = np.r_[np.zeros([self.increase, Xl.shape[1]], dtype=int), Xl, np.zeros([self.increase, Xl.shape[1]], dtype=int)]

        z = np.zeros([x.shape[0], x.shape[0]], dtype=int)

        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                p = Xl[i:i+self.wlen, j:j+self.wlen].flatten()
                p = p * W_n
                p = (p[~np.isnan(p)].astype(int))
                p = ''.join(p.astype(str))

                indices = np.where(j_n[:, 0] == p)
                if indices[0].size > 0 and j_n[indices[0], 1] == '1':
                    z[i, j] = 1
        return z

    def run_window_hood(self, sample, sample_size, W_current, joint_current, Wlast, layer):
        Wsample = []
        for k in range(sample_size):
            Wsample_k = [] 
            for i in range(self.nlayer):
                if layer > i:
                    Wsample_k.append(Wlast[k][i])
                elif i==0:
                    Wsample_k.append(self.apply_window(sample[k], W_current[i], joint_current[i]))
                else:
                    Wsample_k.append(self.apply_window(Wsample_k[i-1], W_current[i], joint_current[i]))
            Wsample.append(Wsample_k)
        return Wsample
    
    def apply_convolve(self, img, W_matrices, bias, weight):
        img_c = np.zeros([img.shape[0], img.shape[1]], dtype=float)
        #_W_matrices = np.array(W_matrices) * weight[:, np.newaxis, np.newaxis]
        for i, kernel in enumerate(W_matrices):
            img_b = cv2.copyMakeBorder(img, self.increase, self.increase, self.increase, self.increase, cv2.BORDER_CONSTANT, None, value = -1) 
            img_r = cv2.filter2D(img_b, -1, kernel)
            img_r = img_r-bias
            img_r = (img_r > 0).astype(float)
            img_c += img_r[self.increase:img_r.shape[0]-self.increase, self.increase:img_r.shape[1]-self.increase]*weight[i]
        img_c = (img_c > 0).astype(float)
        img_c[img_c == 0] = -1
        return img_c
    
    def run_window_convolve(self, sample, sample_size, W_matrices, Wlast, layer, bias, weights):
        Wsample = []
        for k in range(sample_size):
            Wsample_k = [] 
            for i in range(self.nlayer):
                if layer > i:
                    Wsample_k.append(Wlast[k][i])
                elif i==0:
                    Wsample_k.append(self.apply_convolve(sample[k], W_matrices[i], bias[i], weights[i]))
                else:
                    Wsample_k.append(self.apply_convolve(Wsample_k[i-1], W_matrices[i], bias[i], weights[i]))
            Wsample.append(Wsample_k)
        return np.array(Wsample)

    def window_error_generate(self, W_current, joint_current, sample, sample_size, y, error_type, Wlast, layer):
        W_hood = self.run_window_hood(sample, sample_size, W_current, joint_current, Wlast, layer)
        error_hood = self.calculate_error(y, W_hood, error_type)
        return W_hood,error_hood, joint_current
    
    def window_error_generate_c(self, W_matrices, sample, sample_size, y, error_type, Wlast, layer, bias, weights):
        W_hood = self.run_window_convolve(sample, sample_size, W_matrices, Wlast, layer, bias, weights)
        error_hood = self.calculate_error(y, W_hood, error_type)
        return W_hood,error_hood

    def calculate_error(self, y, h, et = 'mae'):
        error = 0
        n_samples = len(y)
        for k in range(n_samples):
            h_z = h[k][-1]
            h_z[h_z==-1]=0
            y_z = y[k]
            y_z[y_z==-1]=0
            if et == 'mae':
                sample_error = np.abs(h_z - y_z).sum()
                error += sample_error / (y_z.size)
            elif et== 'iou':
                union = np.sum(np.maximum(h_z,y_z)==1)
                interc = np.sum(h_z +y_z == 2)
                error += (1 - interc/union)
        return (error/n_samples)     

    def joint_history(self, joint, nlayer):
        for k in range(nlayer):
            if k==0:
                joint_hist = ''.join(joint[k][:,1])
            else:
                joint_hist = joint_hist+''.join(joint[k][:,1])
        return joint_hist

    def window_history(self, W, nlayer, wsize):
        for k in range(nlayer):
            if k==0:
                window_hist = ''.join([''.join(item) for item in np.reshape(W[k], (wsize,)).astype(str)])
            else:
                window_hist = window_hist+''.join([''.join(item) for item in np.reshape(W[k], (wsize,)).astype(str)])
        return window_hist

    def sort_neighbor(self, v, n):
        if n<len(v):
            random.seed(self.random_list[self.count])
            ix = random.sample(range(len(v)), n)
            self.count+=1
        else:
            ix = range(len(v))
        return ix

    def sort_images(self, imgX, imgY, b, img_size):
        random.seed(self.random_list[self.count])
        ix =  random.sample(range(img_size), b)
        self.count+=1
        return [imgX[i] for i in ix], [imgY[i] for i in ix]

    def save_window(self, joint, W):
        filename_joint = f'{self.path_results}/joint{self.name_save}.txt'
        pickle.dump(joint, open(filename_joint, 'wb'))
        filename_W =f'{self.path_results}/W{self.name_save}.txt'
        pickle.dump(W, open(filename_W, 'wb'))

    def get_error_window(self,W, W_matrices, weights, ep_w):

        bias = np.nansum(W, axis=1) - 1

        if self.batch>=self.train_size:
            train_b = copy.deepcopy(self.train)
            ytrain_b = copy.deepcopy(self.ytrain)
            
        self.joint_hist = []
        flg = 0
        epoch_min = 0
        W_size = []
        for i in range(self.nlayer):
            W_size.append(np.sum((W[i] == 1)))

        for ep in range(self.epoch_f):
            self.error_ep_f_hist={"error":[], "joint":[], "ix":[]}
            if self.batch<self.train_size:
                train_b, ytrain_b = self.sort_images(self.train,self.ytrain, self.batch, self.train_size)
                
            Wtrain,w_error =  self.window_error_generate_c(W_matrices, train_b, self.batch,ytrain_b, self.error_type, 0, 0, bias, weights)
            error_last = w_error.copy()
            for k in reversed(range(self.nlayer)):
                
                gradient = -2 * w_error * (1 - Wtrain[k]**2)  # Derivative of tanh
                weights[k] -= self.lr * convolution(activated_result1.sum(axis=2), gradient2)

                # Backward pass for the first layer
                error1 = np.zeros_like(conv_result1)
                for channel in range(filters_layer1.shape[2]):
                    error1[:, :, channel] = convolution(gradient2, filters_layer2[:, :, 0])
                
                gradient1 = -2 * error1 * (1 - activated_result1**2)  # Derivative of tanh
                for channel in range(filters_layer1.shape[2]):
                    filters_layer1[:, :, channel] -= learning_rate * convolution(image[:, :, channel], gradient1[:, :, channel])

                if not self.neighbors_sample:
                    neighbors_to_visit = range(len(joint[k]))
                else:
                    neighbors_to_visit = self.sort_neighbor(joint[k], self.neighbors_sample)
                for i in neighbors_to_visit:
                     self.calculate_neighbors(W,  joint, k, i, Wtrain, train_b,ytrain_b,ep, bias)
                            
            error_min_ep = min(self.error_ep_f_hist['error'])
            ix_min = [i for i,e in enumerate(self.error_ep_f_hist['error']) if e==error_min_ep]
            runs = [v for i, v in enumerate(self.error_ep_f_hist['ix']) if i in(ix_min)]
            ix_run = self.error_ep_f_hist['ix'].index(min(runs))
            joint = self.error_ep_f_hist['joint'][ix_run]

            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(error_min_ep) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i])

            if error_min_ep < w_error:
                w_error = error_min_ep
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep

            if (ep-epoch_min)>self.early_stop_round_f :
                break
        if flg==1:
            joint = copy.deepcopy(joint_min)
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias, weights)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)

    def calculate_neighbors(self,W,  joint, k, i, Wlast, img,yimg, ep, bias):
        joint_temp = copy.deepcopy(joint)
        if joint[k][i][1] == '1':
            joint_temp[k][i][1] = '0'
        else:
            joint_temp[k][i][1] = '1'
        j_temp = self.joint_history(joint_temp, self.nlayer)
        if j_temp not in self.joint_hist:
            self.joint_hist.append(j_temp)
            W_matrices = self.create_w_matrices(W, joint_temp)
            _,error_hood = self.window_error_generate_c(W_matrices, img, self.batch, yimg, self.error_type, Wlast, k, bias,weights)
            self.error_ep_f_hist["error"].append(error_hood)
            self.error_ep_f_hist["joint"].append(joint_temp)
            self.error_ep_f_hist["ix"].append(str(k)+str(i))         

    def get_error_window_parallel(self,W, joint, ep_w, weights):

        nn_parameters = self.initialize_parameters_nn(self.W)

        #W_matrices = self.create_w_matrices(W)
        #bias = np.nansum(W, axis=1) - 1
        #wv=str(self.windows_visit)
        if self.batch>=self.train_size:
            train_b = copy.deepcopy(self.train)
            ytrain_b = copy.deepcopy(self.ytrain)
            #Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias,weights)
        self.joint_hist = []
        flg = 0
        epoch_min = 0
        W_size = []
        for i in range(self.nlayer):
            W_size.append(np.sum((W[i] == 1)))
        for ep in range(1,self.epoch_f+1):
            #self.error_ep_f_hist={"error":[], "joint":[], "ix":[]}
            if self.batch<self.train_size:
                train_b, ytrain_b = self.sort_images(self.train,self.ytrain, self.batch, self.train_size)
                #Wtrain,w_error_b,_ =  self.window_error_generate(W, joint, train_b, self.batch, ytrain_b, self.error_type, self.train, 0)
                #Wtrain = self.run_window_convolve(train_b, self.batch, W_matrices, 0, 0, bias)
                #if ep==1:
                    #w_error = self.calculate_error(ytrain_b, Wtrain, self.error_type)
            #self.joint_hist.append(self.joint_history(joint, self.nlayer))
            for k in range(self.nlayer):

                self.L_model_forward(self, X, parameters, layer)



                if not self.neighbors_sample:
                    neighbors_to_visit = range(len(joint[k]))
                else:
                    neighbors_to_visit = self.sort_neighbor(joint[k], self.neighbors_sample)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    [executor.submit(self.calculate_neighbors,W,  joint, k, i, Wtrain, train_b,ytrain_b,ep, bias) for i in neighbors_to_visit]
            error_min_ep = min(self.error_ep_f_hist['error'])
            ix_min = [i for i,e in enumerate(self.error_ep_f_hist['error']) if e==error_min_ep]
            runs = [v for i, v in enumerate(self.error_ep_f_hist['ix']) if i in(ix_min)]
            ix_run = self.error_ep_f_hist['ix'].index(min(runs))
            joint = self.error_ep_f_hist['joint'][ix_run]

            self.error_ep_f["W_key"].append(self.windows_visit) 
            self.error_ep_f["epoch_w"].append(ep_w) 
            self.error_ep_f["epoch_f"].append(ep) 
            self.error_ep_f["error"].append(error_min_ep) 
            self.error_ep_f["time"].append((time() -  self.start_time)) 
            for i in range(self.nlayer):
                self.error_ep_f[f"window_size_{i}"].append(W_size[i]) 

            if error_min_ep < w_error:
                w_error = copy.deepcopy(error_min_ep)
                joint_min = copy.deepcopy(joint)
                flg=1
                epoch_min = ep 
            
            if (ep-epoch_min)>self.early_stop_round_f :
                break

        if flg==1:
            joint = copy.deepcopy(joint_min)
            
        W_matrices = self.create_w_matrices(W, joint)
        _,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, self.val, 0, bias, weights)
        
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)


    def check_great_neighboors(self, W,joint, ep_w):
        flg = 0
        error_ep = {"error_train":[],"error_val":[],"W":[],"joint":[]}
        
        for k in range(self.nlayer):
            nan_idx = np.where(np.isnan(W[k]))[0]
            w_line_temp_base = W[k].copy()
            for i in nan_idx:
                W_line_temp = copy.deepcopy(w_line_temp_base)
                W_line_temp[i] = 1
                W_line_temp_NN = copy.deepcopy(W_line_temp)
                W_line_temp_NN[np.isnan(W_line_temp_NN)] = 0
                W_line_temp_NN = W_line_temp_NN.astype(int)
                    
                if ''.join(W_line_temp_NN.astype(str)) in self.windows_continuos:
                    W_temp = copy.deepcopy(W)
                    W_temp[k] = W_line_temp
                    W_h = self.window_history(W_temp, self.nlayer, self.wsize)

                    if W_h not in self.w_hist['W']:
                        self.windows_visit+=1
                        joint_temp = copy.deepcopy(joint)
                        joint_temp[k] = self.create_joint(W_temp[k])

                        if self.parallel:
                            joint_temp, w_error, f_epoch_min = self.get_error_window_parallel(W_temp, joint_temp, ep_w)
                        else:
                            joint_temp, w_error, f_epoch_min = self.get_error_window(W_temp, joint_temp, ep_w)
                        error_ep['error_train'].append(w_error[0])
                        error_ep['error_val'].append(w_error[1])
                        error_ep['W'].append(W_temp)
                        error_ep['joint'].append(joint_temp)

                        self.w_hist["W_key"].append(self.windows_visit)
                        self.w_hist["W"].append(W_h)
                        self.w_hist["error"].append(w_error)
                        self.w_hist["f_epoch_min"].append(f_epoch_min)
        if error_ep["error_train"]:
            ix = error_ep['error_val'].index(min(error_ep['error_val']))
            error_min_ep = np.array([error_ep['error_train'][ix],error_ep['error_val'][ix]])
            W = error_ep['W'][ix]
            joint = error_ep['joint'][ix]
           
            return W, joint, error_min_ep
        else:
            return W, joint, np.array([np.nan, np.nan])


    def check_lesser_neighboors(self, W,joint, ep_w):
        flg = 0
        error_ep = {"error_train":[],"error_val":[],"W":[],"joint":[]}
        
        for k in range(self.nlayer):
            nan_idx = np.where(W[k]==1)[0]
            w_line_temp_base = W[k].copy()
            for i in nan_idx:
                W_line_temp = copy.deepcopy(w_line_temp_base)
                W_line_temp[i] = np.nan
                W_line_temp_NN = copy.deepcopy(W_line_temp)
                
                W_line_temp_NN[np.isnan(W_line_temp_NN)] = 0
                W_line_temp_NN = W_line_temp_NN.astype(int)
                
                if ''.join(W_line_temp_NN.astype(str)) in self.windows_continuos:
                    W_temp = copy.deepcopy(W)
                    W_temp[k] = W_line_temp
                    W_h = self.window_history(W_temp, self.nlayer, self.wsize)
                    
                    if W_h not in self.w_hist['W']:
                        self.windows_visit+=1
                        joint_temp = copy.deepcopy(joint)
                        joint_temp[k] = self.create_joint(W_temp[k])
                        
                        if self.parallel:
                            joint_temp, w_error, f_epoch_min = self.get_error_window_parallel(W_temp, joint_temp, ep_w)
                        else:
                            joint_temp, w_error, f_epoch_min = self.get_error_window(W_temp, joint_temp, ep_w)
                        error_ep['error_train'].append(w_error[0])
                        error_ep['error_val'].append(w_error[1])
                        error_ep['W'].append(W_temp)
                        error_ep['joint'].append(joint_temp)

                        self.w_hist["W_key"].append(self.windows_visit)
                        self.w_hist["W"].append(W_h)
                        self.w_hist["error"].append(w_error)
                        self.w_hist["f_epoch_min"].append(f_epoch_min)
       
        if error_ep["error_train"]:
            ix = error_ep['error_val'].index(min(error_ep['error_val']))
            error_min_ep = np.array([error_ep['error_train'][ix],error_ep['error_val'][ix]])
            W = error_ep['W'][ix]
            joint = error_ep['joint'][ix]
            return W, joint, error_min_ep
        else:
            return W, joint, np.array([np.nan, np.nan])

    def fit(self):
        self.start_time = time()

        #W_matrices, weights = zip(*[self.create_w_matrices(self.W[i]) for i in range(self.nlayer)])
        
        # Primeira rodada da janela inicial
        if self.parallel:
            joint, error,f_epoch_min = self.get_error_window_parallel(self.W ,ep_w=0)
        else:
            joint, error,f_epoch_min = self.get_error_window(self.W,W_matrices, weights, ep_w=0)

        #self.save_window(joint, self.W)

        self.w_hist["W_key"].append(self.windows_visit)
        self.w_hist["W"].append(self.window_history(self.W, self.nlayer, self.wsize))
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
            W_l, joint_l, error_l = self.check_lesser_neighboors(W,joint,ep)
            W_g, joint_g, error_g = self.check_great_neighboors(W, joint,ep)
            
            if (error_l[1] <= error_g[1]) | (np.isnan(error_g[1])):
                W = copy.deepcopy(W_l)
                joint = copy.deepcopy(joint_l)
                error = copy.deepcopy(error_l)
                
            elif (error_g[1] < error_l[1]) | (np.isnan(error_g[1])):
                W = copy.deepcopy(W_g)
                joint = copy.deepcopy(joint_g)
                error = copy.deepcopy(error_g)

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

            if (ep-ep_min)>self.early_stop_round_w :
                print('End by Early Stop Round')
                break
                
        print('----------------------------------------------------------------------') 
        Wtest,error_test =  self.window_error_generate_c(W_matrices, self.test, self.test_size, self.ytest, self.error_type, 0, 0, bias, weights)
        Wtrain,error_train =  self.window_error_generate_c(W_matrices, self.train, self.train_size, self.ytrain, self.error_type, 0, 0,bias, weights)
        Wval,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, 0, 0, bias, weights)

        print('End of testing')
        end_time = time()
        time_min = (end_time -  self.start_time) / 60
        print(f'Time: {time_min:.2f} | Min-Epoch {ep_min} / {self.epoch_w} - Train error: {error_train} / Validation error: {error_val} / Test error: {error_test}')

        self.save_results_complet(Wtrain, Wval, Wtest)
        pickle.dump(self.w_hist, open(f'{self.path_results}/W_hist{self.name_save}.txt', 'wb'))
        pickle.dump(error_ep, open(f'{self.path_results}/error_ep_w{self.name_save}.txt', 'wb'))
        pickle.dump(self.error_ep_f, open(f'{self.path_results}/error_ep_f{self.name_save}.txt', 'wb'))
        #pickle.dump(self.error_ep_f_hist, open(f'{self.path_results}/error_ep_f_hist{self.name_save}.txt', 'wb'))
        print(f"Janela Final Aprendida: {W_min}")
        print(f"Joint Final Aprendida: {joint_min}")

    def save_to_csv(self, data, name):
        df = pd.DataFrame(data)
        df.to_csv(f'{name}.csv', index=False)
    
    def test(self):
        start = time()
        if self.parallel:
            _, error,f_epoch_min = self.get_error_window_parallel(self.W,self.joint)
        else:
            _, error,f_epoch_min = self.get_error_window(self.W,self.joint)
        end = time()
        print('tempo de execução: {}'.format(end - start))
        print(f'época-min: {f_epoch_min} - com erro: {error}')
    
    def results_after_fit(self, path_results, name_save):
        joint = np.load(f'{path_results}/joint{name_save}.txt', allow_pickle=True)
        W = np.load(f'{path_results}/W{name_save}.txt', allow_pickle=True)

        print(f'Reading from: {path_results}')

        W_matrices = self.create_w_matrices(W,joint)
        bias = np.nansum(W, axis=1) - 1

        Wtest,error_test =  self.window_error_generate_c(W_matrices, self.test, self.test_size, self.ytest, self.error_type, 0, 0, bias, weights)
        Wtrain,error_train =  self.window_error_generate_c(W_matrices, self.train, self.train_size, self.ytrain, self.error_type, 0, 0,bias, weights)
        Wval,error_val =  self.window_error_generate_c(W_matrices, self.val, self.val_size, self.yval, self.error_type, 0, 0, bias, weights)

        print(f'Train error: {error_train} / Validation error: {error_val} / Test error: {error_test}')

        self.save_results_complet(Wtrain, Wval, Wtest)

    
    def test2(self):
        W_matrices = self.create_w_matrices(self.W, self.joint)
        bias = np.nansum(self.W, axis=1) - 1
        Wtrain,w_error =  self.window_error_generate_c(W_matrices, self.train, self.train_size,self.ytrain, self.error_type, 0, 0, bias, weights)
        return Wtrain,w_error, self.W, self.joint
    ##############-------------------__###################
    
    def initialize_parameters_nn(self, W):
        #np.random.seed(self.random_list[self.count])
        np.random.seed(1)
        self.count+=1
        parameters = {}
        
        W_size = np.nansum(W, axis=1)
        for l in range(self.nlayer):
            parameters['l' + str(l)] = {}
            layer_dims = []
            layer_dims.append(int(W_size[l]))
            layer_dims.append(int(2**W_size[l]))
            layer_dims.append(1)

            for index in range(len(layer_dims)-1):
                if (index) % 3 == 0:
                    parameters['l' + str(l)]['W' + str(index)] = np.array(self.create_w_matrices(W[l]))
                    parameters['l' + str(l)]['b' + str(index)] = np.array([-layer_dims[index]+1]*layer_dims[index+1]).reshape(layer_dims[index+1],1)
                else:
                    parameters['l' + str(l)]['W' + str(index)] = np.random.randint(2, size = (layer_dims[index+1], layer_dims[index]))
                    parameters['l' + str(l)]['b' + str(index)] = np.zeros((layer_dims[index+1], 1))

        return parameters
    
    def L_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def L_activation_forward(self, A_prev, W, b):
        Z, linear_cache = self.L_forward(A_prev, W, b)
        #A = np.tanh(Z)
        A = np.where(Z <= 0, 0, 1)
        activation_cache = Z
        cache = (linear_cache, activation_cache)
        return A, cache

    def L_model_forward(self, X, parameters):
        A = X
        caches = []
        for k in range(self.nlayer):
            L = len(parameters['l'+str(k)]) // 2
            for l in range(L):
                A_prev = A
                A, cache = self.L_activation_forward(
                    A_prev, parameters['l'+str(k)]["W" + str(l)], parameters['l'+str(k)]["b" + str(l)])
                caches.append(cache)
            if k<self.nlayer-1:
                A = self.transform_image_forlayer(A, self.train_size)
        return A, caches
    
    def compute_cost(self, AL, Y):
        #v = Y * AL
        #cost = np.sum((np.maximum(0, 1 - v)) / AL.size)

        AL = np.where(AL <= 0, 0, 1)
        y_pred = self.transform_nn_for_image(AL, self.train_size)
        y_img = self.transform_nn_for_image(Y, self.train_size)
        n_samples = len(y_img)
        error = 0
        for k in range(n_samples):
            y_z = copy.deepcopy(y_img[k])
            y_z[y_z==-1]=0
            union = np.sum(np.maximum(y_pred,y_z)==1)
            interc = np.sum(y_pred +y_z == 2)
            error += (1 - interc/union)
        return error/n_samples 
    
    def L_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m # delta W = gradient * neurons_inputs
        db = np.sum(dZ, axis=1, keepdims=True) / m # delta bias = sigmoid derivative median
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def sigmoid_backward(self, dA, cache, flag = False):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s) # gradient = sigmoid derivative * logloss derivative
        if flag:
            middle_row_index = dZ.shape[0] // 2
            dZ = dZ[middle_row_index].reshape(1, -1)

        return dZ
        
    def tanh_backward(self, dA, cache, flag = False):
        Z = cache
        s = 1 - np.tanh(Z)**2
        dZ = dA * s  # gradient = tanh derivative * hingeloss derivative
        if flag:
            middle_row_index = dZ.shape[0] // 2
            dZ = dZ[middle_row_index].reshape(1, -1)
        return dZ
    
    def hinge_loss_derivative(self, y, fx):
        condition = 1 - y * fx > 0
        derivative = -y * condition
        return derivative
    
    def calculate_iou_derivative(self, G, P):
        intersection = np.sum(G * P)
        union = np.sum(G) + np.sum(P) - intersection

        # Handle division by zero
        if union == 0:
            return 0.0

        # Calculate the derivative
        derivative = (G * (np.sum(P) + np.sum(G) - 2 * intersection)) / (union ** 2)

        return derivative
    
    def L_activation_backward(self, dA, cache, flag = False):
        linear_cache, activation_cache = cache

        #dZ = self.tanh_backward(dA, activation_cache, flag)
        dZ = self.sigmoid_backward(dA, activation_cache, flag)
        dA_prev, dW, db = self.L_backward(dZ, linear_cache)

        return dA_prev, dW, db
    
    def L_model_backward(self, AL, Y, caches, parameters):
        grads = {}
        L = len(caches) 
        #Y = Y.reshape(AL.shape)

        #dAL = self.hinge_loss_derivative(Y, AL) # hinge derivative
        dAL = self.calculate_iou_derivative(Y, AL)

        LL = len(caches)-1
        grads = {}
        for k in reversed(range(self.nlayer)):
            L = (len(parameters['l'+str(k)]) // 2)
            for l in reversed(range(L)):
                if k == self.nlayer-1 and l == L-1:
                    dA_prev_temp, dW_temp, db_temp = self.L_activation_backward(dAL, caches[LL])
                elif k<self.nlayer-1 and l == L-1:
                    dA_prev_temp, dW_temp, db_temp = self.L_activation_backward(dA_prev_temp, caches[LL], flag = True)
                else:
                    dA_prev_temp, dW_temp, db_temp = self.L_activation_backward(dA_prev_temp, caches[LL])
                grads["l"+str(k)+"dA" + str(l)] = dA_prev_temp
                grads["l"+str(k)+"dW" + str(l)] = dW_temp
                grads["l"+str(k)+"db" + str(l)] = db_temp
                
                LL-=1

        return grads
    
    def update_parameters(self, params, grads, learning_rate):
        parameters = params.copy()

        for k in range(self.nlayer):
            L = len(parameters['l'+str(k)]) // 2
            for l in range(L):
                if l>0:
                    parameters['l'+str(k)]["W" + str(l)] = parameters['l'+str(k)]["W" + str(l)] - \
                        learning_rate * grads["l"+str(k)+"dW" + str(l)]
                    #parameters['l'+str(k)]["b" + str(l)] = parameters["b" + str(l)] - \
                    #    learning_rate * grads["l"+str(k)+"db" + str(l)]

        return parameters
    
    def nn_model_train(self, X, Y, parameters, learning_rate=0.1, num_iterations = 20000
                   , print_cost = True, ):
        np.random.seed(1)
        costs = [] # keep track of cost
        
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X, parameters) # Calculate the output of the network -- forward propagation
            cost = self.compute_cost(AL, Y) # Calculate the Logloss error
            grads = self.L_model_backward(AL, Y, caches, parameters) # Calculate the Gradient
            parameters = self.update_parameters(parameters, grads, learning_rate)

            if i % 1000 == 0:
                learning_rate = learning_rate / 2 # After 2.000 iterarions, divide learning_rate by 2
            if i % 2000 == 0:
                learning_rate = learning_rate / 2 # After 2.000 iterarions, divide learning_rate by 2

            if print_cost and i % 10 == 0 or i == num_iterations - 1:
                print(f"Cost after iteration {i}: {np.squeeze(cost)}")
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        for k in range(self.nlayer):
            L = len(parameters['l'+str(k)]) // 2
            for l in range(L):
                parameters['l'+str(k)]["W" + str(l)] = (parameters['l'+str(k)]["W" + str(l)]>0.3).astype(float)

        return parameters, costs
    
    def transform_nn_for_image(self, img, img_size):
        imagem_shape = (self.img_shape, self.img_shape)
        img_reshape = np.reshape(img, (img_size, *imagem_shape))
        return img_reshape
    
    def apply_nn(self, X, y, parameters):

        probas, _ = self.L_model_forward(X, parameters)
        probas = np.where(probas < 0, 0, 1)
        y_pred = self.transform_nn_for_image(probas, self.train_size)
        y_img = self.transform_nn_for_image(y, self.train_size)
        error = 0
        n_samples = len(y_img)
        for k in range(n_samples):
            y_z = copy.deepcopy(y_img[k])
            y_z[y_z==-1]=0
            union = np.sum(np.maximum(y_pred,y_z)==1)
            interc = np.sum(y_pred +y_z == 2)
            error += (1 - interc/union)
            x = copy.deepcopy(y_pred[k])
            x[(x==-1)]=255
            x[(x==1)]=0
            cv2.imwrite(f'{self.path_results}/nn_{k:02d}{self.name_save}.jpg', x)


        error = error/n_samples           

        return error
    