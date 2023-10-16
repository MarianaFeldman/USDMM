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
            #self.W = [self.create_window() for _ in range(self.nlayer)]
            self.W = [Wini.copy() for _ in range(self.nlayer)]
            self.joint = [self.create_joint(self.W[i]) for i in range(self.nlayer)]
            #self.w_hist = {"W":[],"error":[], "f_epoch_min":[]}

        else:
            self.joint = np.load('./data/joint'+new+'.txt', allow_pickle=True)
            self.W = np.load('./data/W'+new+'.txt', allow_pickle=True)
            #self.w_hist = np.load('W_hist'+name_save+'.txt', allow_pickle=True)
        print(f"Janela Inicializada: {self.W}")
        print(f"Joint Inicializada: {self.joint}")
        self.w_hist = {"W_key": [],"W":[],"error":[], "f_epoch_min":[]}
        self.windows_visit = 1
        self.error_ep_f_hist = {"W_key": [], "epoch": [], "error":[], "joint":[]}
        self.error_ep_f = {"W_key": [], "epoch_w": [], "epoch_f":[], "error":[], "time":[] }
        wind_size_dict = {f'window_size_{i}': [] for i in range(self.nlayer)}
        self.error_ep_f = {key: value for d in [self.error_ep_f, wind_size_dict] for key, value in d.items()}

        
        self.train, self.ytrain = self.get_images(train_size, 'train')
        self.val, self.yval = self.get_images(val_size, 'val')
        self.test, self.ytest = self.get_images(test_size, 'test')

        self.path_results = path_results
        isExist = os.path.exists(path_results)
        if not isExist:
            os.makedirs(path_results)
        print("Resultados serão salvos em ",path_results)
        self.name_save = name_save
        self.parallel = parallel
        self.joint_hist = []

        self.early_stop_round_f = early_stop_round_f
        self.early_stop_round_w = early_stop_round_w

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
        random.seed(self.random_list[self.count])
        self.count +=1
        return np.c_[Ji, np.random.randint(2, size=len(Ji))]

    def _convert_binary(self,img):
        (_, img_bin) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img_bin[(img_bin==0)]=1
        img_bin[(img_bin==255)]=0
        img_bin = img_bin.astype(int)
        return img_bin
    
    def get_images(self, img_size, img_type):
        ximg = []
        yimg = []
        for img in range(1,img_size+1):
            s = str(img)
            s = s.zfill(2)
            x = cv2.imread('./data/x/'+img_type+s+'.jpg', cv2.IMREAD_GRAYSCALE)
            y = cv2.imread('./data/y/'+img_type+s+'.jpg', cv2.IMREAD_GRAYSCALE)
            ximg.append(self._convert_binary(x))
            yimg.append(self._convert_binary(y))
        return ximg, yimg

    def save_results(self, Wtrain, Wval, Wtest):
        for img in range(len(Wtrain)):
            s = str(img+1)
            s = s.zfill(2)
            x = copy.deepcopy(Wtrain[img][1])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite(self.path_results+'train'+s+self.name_save+'.jpg', x)

        for img in range(len(Wval)):
            s = str(img+1)
            s = s.zfill(2)
            x = copy.deepcopy(Wval[img][1])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite(self.path_results+'val'+s+self.name_save+'.jpg', x)
        
        for img in range(len(Wtest)):
            s = str(img+1)
            s = s.zfill(2)
            x = copy.deepcopy(Wtest[img][1])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite(self.path_results+'test'+s+self.name_save+'.jpg', x)
    
    def save_results_complet(self, Wtrain, Wval, Wtest, ep = None):
        for img in range(len(Wtrain)):
            s = str(img+1)
            s = s.zfill(2)
            for k in range(self.nlayer):
                x = copy.deepcopy(Wtrain[img][k])
                x[(x==0)]=255
                x[(x==1)]=0
                cv2.imwrite(self.path_results+'/train_op'+str(k+1)+'_'+s+self.name_save+ep+'.jpg', x)

        for img in range(len(Wval)):
            s = str(img+1)
            s = s.zfill(2)
            for k in range(self.nlayer):
                x = copy.deepcopy(Wval[img][k])
                x[(x==0)]=255
                x[(x==1)]=0
                cv2.imwrite(self.path_results+'/val_op'+str(k+1)+'_'+s+self.name_save+ep+'.jpg', x)

        for img in range(len(Wtest)):
            s = str(img+1)
            s = s.zfill(2)
            for k in range(self.nlayer):
                x = copy.deepcopy(Wtest[img][k])
                x[(x==0)]=255
                x[(x==1)]=0
                cv2.imwrite(self.path_results+'/test_op'+str(k+1)+'_'+s+self.name_save+ep+'.jpg', x)
    
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

    def window_error_generate(self, W_current, joint_current, sample, sample_size, y, error_type, Wlast, layer):
        W_hood = self.run_window_hood(sample, sample_size, W_current, joint_current, Wlast, layer)
        error_hood = self.calculate_error(y, W_hood, error_type)
        return W_hood,error_hood, joint_current

    def calculate_error(self, y, h, et = 'mae'):
        error = 0
        n_samples = len(y)
        for k in range(n_samples):
            if et == 'mae':
                sample_error = np.abs(h[k][-1] - y[k]).sum()
                error += sample_error / (y[k].size)
            elif et== 'iou':
                union = np.sum(np.maximum(h[k][-1],y[k])==1)
                interc = np.sum(h[k][-1] +y[k] == 2)
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
        filename_joint = self.path_results+'/joint'+self.name_save+'.txt'
        pickle.dump(joint, open(filename_joint, 'wb'))
        filename_W =self.path_results+ '/W'+self.name_save+'.txt'
        pickle.dump(W, open(filename_W, 'wb'))

    def get_error_window(self,W, joint, ep_w):
        if self.batch>=self.train_size:
            train_b = copy.deepcopy(self.train)
            ytrain_b = copy.deepcopy(self.ytrain)
            Wtrain,w_error,_ =  self.window_error_generate(W, joint, self.train, self.train_size, self.ytrain, self.error_type, 0, 0)
        self.joint_hist = []
        flg = 0
        epoch_min = 0
        W_size = []
        for i in range(self.nlayer):
            W_size.append(np.sum((W[i] == 1)))

        for ep in range(self.epoch_f):
            if self.batch<self.train_size:
                train_b, ytrain_b = self.sort_images(self.train,self.ytrain, self.batch, self.train_size)
                #Wtrain,w_error_b,_ =  self.window_error_generate(W, joint, train_b, self.batch, ytrain_b, self.error_type, self.train, 0)
                Wtrain = self.run_window_hood(train_b, self.batch, W, joint, 0, 0)
                if ep==1:
                    w_error = self.calculate_error(ytrain_b, Wtrain, self.error_type)
            self.joint_hist.append(self.joint_history(joint, self.nlayer))
            for k in range(self.nlayer):
                if not self.neighbors_sample:
                    neighbors_to_visit = range(len(joint[k]))
                else:
                    neighbors_to_visit = self.sort_neighbor(joint[k], self.neighbors_sample)
                for i in neighbors_to_visit:
                     self.calculate_neighbors(W,  joint, k, i, Wtrain, train_b,ytrain_b,ep)
                            
            ix = [i for i, value in enumerate(self.error_ep_f_hist["W_key"]) if value == self.windows_visit]
            error_ix = [self.error_ep_f_hist["error"][i] for i in ix]
            joint_ix = [self.error_ep_f_hist["joint"][i] for i in ix]

            error_min_ep = min(error_ix)
            joint = joint_ix[error_ix.index(min(error_ix))]

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

        _,error_val,_ =  self.window_error_generate(W, joint, self.val, self.val_size, self.yval, self.error_type, self.val, 0)
        error = np.array([w_error, error_val])
        return (joint, error, epoch_min)

    def calculate_neighbors(self,W,  joint, k, i, Wlast, img,yimg, ep):
        joint_temp = copy.deepcopy(joint)
        if joint[k][i][1] == '1':
            joint_temp[k][i][1] = '0'
        else:
            joint_temp[k][i][1] = '1'
        j_temp = self.joint_history(joint_temp, self.nlayer)
        if j_temp not in self.joint_hist:
            self.joint_hist.append(j_temp)
            _,error_hood, joint_temp = self.window_error_generate(W, joint_temp, img, self.batch, yimg, self.error_type, Wlast, k)
            self.error_ep_f_hist["W_key"].append(self.windows_visit)
            self.error_ep_f_hist["epoch"].append(ep)
            self.error_ep_f_hist["error"].append(error_hood)
            self.error_ep_f_hist["joint"].append(joint_temp)

            

    def get_error_window_parallel(self,W, joint, ep_w):
        if self.batch>=self.train_size:
            train_b = copy.deepcopy(self.train)
            ytrain_b = copy.deepcopy(self.ytrain)
            Wtrain,w_error,_ =  self.window_error_generate(W, joint, self.train, self.train_size, self.ytrain, self.error_type, 0, 0)
        self.joint_hist = []
        flg = 0
        epoch_min = 0
        W_size = []
        for i in range(self.nlayer):
            W_size.append(np.sum((W[i] == 1)))
        for ep in range(1,self.epoch_f+1):
            if self.batch<self.train_size:
                train_b, ytrain_b = self.sort_images(self.train,self.ytrain, self.batch, self.train_size)
                #Wtrain,w_error_b,_ =  self.window_error_generate(W, joint, train_b, self.batch, ytrain_b, self.error_type, self.train, 0)
                Wtrain = self.run_window_hood(train_b, self.batch, W, joint, 0, 0)
                if ep==1:
                    w_error = self.calculate_error(ytrain_b, Wtrain, self.error_type)
            self.joint_hist.append(self.joint_history(joint, self.nlayer))
            for k in range(self.nlayer):
                if not self.neighbors_sample:
                    neighbors_to_visit = range(len(joint[k]))
                else:
                    neighbors_to_visit = self.sort_neighbor(joint[k], self.neighbors_sample)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    [executor.submit(self.calculate_neighbors,W,  joint, k, i, Wtrain, train_b,ytrain_b,ep) for i in neighbors_to_visit]

            ix = [i for i, value in enumerate(self.error_ep_f_hist["W_key"]) if value == self.windows_visit]
            error_ix = [self.error_ep_f_hist["error"][i] for i in ix]
            joint_ix = [self.error_ep_f_hist["joint"][i] for i in ix]
        
            error_min_ep = min(error_ix)
            joint = joint_ix[error_ix.index(min(error_ix))]

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
            
        _,error_val,_ =  self.window_error_generate(W, joint, self.val, self.val_size, self.yval, self.error_type, 0, 0)
        
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
       
        ix = error_ep['error_val'].index(min(error_ep['error_val']))
        error_min_ep = np.array([error_ep['error_train'][ix],error_ep['error_val'][ix]])
        W = error_ep['W'][ix]
        joint = error_ep['joint'][ix]
           
        return W, joint, error_min_ep


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
       
        ix = error_ep['error_val'].index(min(error_ep['error_val']))
        error_min_ep = np.array([error_ep['error_train'][ix],error_ep['error_val'][ix]])
        W = error_ep['W'][ix]
        joint = error_ep['joint'][ix]
           
        return W, joint, error_min_ep

    def fit(self):
        self.start_time = time()
        
        # Primeira rodada da janela inicial
        if self.parallel:
            joint, error,f_epoch_min = self.get_error_window_parallel(self.W,self.joint,0)
        else:
            joint, error,f_epoch_min = self.get_error_window(self.W,self.joint,0)

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
            
            if error_l[1] <= error_g[1]:
                W = copy.deepcopy(W_l)
                joint = copy.deepcopy(joint_l)
                error = copy.deepcopy(error_l)
                
            elif error_g[1] < error_l[1]:
                W = copy.deepcopy(W_g)
                joint = copy.deepcopy(joint_g)
                error = copy.deepcopy(error_g)

            if error[1]<error_min[1]:
                ep_min = ep
                self.save_window(joint, W)
                error_min = copy.deepcopy(error)
                W_min = copy.deepcopy(W)
                joint_min = copy.deepcopy(joint)

                Wtrain = self.run_window_hood(self.train, self.train_size, W, joint, 0, 0)
                Wval = self.run_window_hood(self.val, self.val_size, W, joint, 0, 0)
                Wtest = self.run_window_hood(self.test, self.test_size, W, joint, 0, 0)
                self.save_results_complet(Wtrain, Wval, Wtest, f'_epoch{ep}')
            
            error_ep['epoch'].append(ep)
            error_ep['error_train'].append(error[0])
            error_ep['error_val'].append(error[1])
            
            self.save_to_csv(error_ep, self.path_results+'/error_ep_w'+self.name_save+'_epoch'+str(ep))
            self.save_to_csv(self.error_ep_f, self.path_results+'/error_ep_f'+self.name_save+'_epoch'+str(ep))

            time_min = (time() -  self.start_time) / 60
            print(f'Time: {time_min:.2f} | Epoch {ep} / {self.epoch_w} - Validation error: {error[1]}')

            if (ep-ep_min)>self.early_stop_round_w :
                print('End by Early Stop Round')
                break
                
        print('----------------------------------------------------------------------') 
        Wtest,error_test,_ =  self.window_error_generate(W_min, joint_min, self.test, self.test_size, self.ytest, self.error_type, 0, 0)
        Wtrain,error_train,_ =  self.window_error_generate(W_min, joint_min, self.train, self.train_size, self.ytrain, self.error_type, 0, 0)
        Wval,error_val,_ =  self.window_error_generate(W_min, joint_min, self.val, self.val_size, self.yval, self.error_type, 0, 0)

        print('End of testing')
        end_time = time()
        time_min = (end_time -  self.start_time) / 60
        print(f'Time: {time_min:.2f} | Min-Epoch {ep_min} / {self.epoch_w} - Train error: {error_train} / Validation error: {error_val} / Test error: {error_test}')

        self.save_results_complet(Wtrain, Wval, Wtest)
        pickle.dump(self.w_hist, open(self.path_results+'/W_hist'+self.name_save+'.txt', 'wb'))
        pickle.dump(error_ep, open(self.path_results+'/error_ep_w'+self.name_save+'.txt', 'wb'))
        pickle.dump(self.error_ep_f, open(self.path_results+'/error_ep_f'+self.name_save+'.txt', 'wb'))
        pickle.dump(self.error_ep_f_hist, open(self.path_results+'/error_ep_f_hist'+self.name_save+'.txt', 'wb'))
        print(f"Janela Final Aprendida: {W_min}")
        print(f"Joint Final Aprendida: {joint_min}")

    def save_to_csv(self, data, name):
        df = pd.DataFrame(data)
        df.to_csv(name+'.csv', index=False)
    
    def test(self):
        start = time()
        if self.parallel:
            _, error,f_epoch_min = self.get_error_window_parallel(self.W,self.joint)
        else:
            _, error,f_epoch_min = self.get_error_window(self.W,self.joint)
        end = time()
        print('tempo de execução: {}'.format(end - start))
        print('época-min: ',f_epoch_min, ' - com erro: ',error )
    
    def results_after_fit(self):
        joint = np.load(self.path_results+'/joint'+self.name_save+'.txt', allow_pickle=True)
        W = np.load(self.path_results+'/W'+self.name_save+'.txt', allow_pickle=True)
        
        print(W)
        print(self.path_results)
        Wtrain = self.run_window_hood(self.train, self.train_size, W, joint, 0, 0)
        Wval = self.run_window_hood(self.val, self.val_size, W, joint, 0, 0)
        Wtest = self.run_window_hood(self.test, self.test_size, W, joint, 0, 0)

        for img in range(len(Wtrain)):
            s = str(img+1)
            s = s.zfill(2)
            x = copy.deepcopy(Wtrain[img][0])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite(self.path_results+'/train_op1_'+s+self.name_save+'.jpg', x)
            x = copy.deepcopy(Wtrain[img][1])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite(self.path_results+'/train_op2_'+s+self.name_save+'.jpg', x)

        for img in range(len(Wval)):
            s = str(img+1)
            s = s.zfill(2)
            x = copy.deepcopy(Wval[img][0])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite('./'+self.path_results+'/val_op1_'+s+self.name_save+'.jpg', x)
            x = copy.deepcopy(Wval[img][1])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite('./'+self.path_results+'/val_op2_'+s+self.name_save+'.jpg', x)

        for img in range(len(Wtest)):
            s = str(img+1)
            s = s.zfill(2)
            x = copy.deepcopy(Wtest[img][0])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite('./'+self.path_results+'/test_op1_'+s+self.name_save+'.jpg', x)
            x = copy.deepcopy(Wtest[img][1])
            x[(x==0)]=255
            x[(x==1)]=0
            cv2.imwrite('./'+self.path_results+'/test_op2_'+s+self.name_save+'.jpg', x)

    
        
