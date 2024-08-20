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
import pandas as pd
from keras.datasets import mnist

#directory for colab
#work_directory = '.'
#work_output = 'output_colab' 

#directory for linux
work_directory = '/app/'
work_output = f'{work_directory}output' 



class WOMC_load_images:

    def __init__(self, train_size, val_size, test_size, img_type ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        if img_type == 'img_n':
          self.train, self.ytrain, self.val, self.yval, self.test, self.ytest = self.get_image(train_size, val_size, test_size)
        elif (img_type == 'gl') or (img_type == 'classification'):
          for img_type in ['train', 'val', 'test']:
            x_filename = f'./data/x_{img_type}/{img_type}.txt'
            x_data = np.load(x_filename, allow_pickle=True)
            setattr(self, f'{img_type}', np.where(np.array(x_data) == 0, -1, x_data))
            # Carregar e transformar dados Y
            y_filename = f'./data/y_{img_type}/{img_type}.txt'
            y_data = np.load(y_filename, allow_pickle=True)
            #setattr(self, f'y{img_type}', np.where(np.array(y_data) == 0, -1, y_data))
            setattr(self, f'y{img_type}', y_data)
        elif img_type.startswith('GoL'):
          if img_type.endswith('_sp'):
            img_type = img_type[:-3]
            flg_sp = 1
          elif img_type.endswith('_sp1'):
            img_type = img_type[:-4]
            flg_sp = 2
          elif img_type.endswith('_sp2'):
            img_type = img_type[:-4]
            flg_sp = 3
          elif img_type.endswith('_sp3'):
            img_type = img_type[:-4]
            flg_sp = 4
          elif img_type.endswith('_sp4'):
            img_type = img_type[:-4]
            flg_sp = 5
          else:
             flg_sp = 0
          path = f'data/{img_type}'
          start = 0
          self.train = self.load_data(start, train_size, path)
          self.ytrain = self.load_labels(start, train_size, path, flg_sp)

          self.val = self.load_data(start+train_size, train_size + val_size, path)
          self.yval = self.load_labels(start+train_size, train_size + val_size, path,flg_sp)

          self.test = self.load_data(start+train_size + val_size, train_size + val_size + test_size, path)
          self.ytest = self.load_labels(start+train_size + val_size, train_size + val_size + test_size, path,flg_sp)
        elif img_type.startswith('mnist'):
          digito = int(img_type[-1])
          self.train,self.ytrain,self.val, self.yval,self.test, self.ytest =self.transform_mnist_digit(digito, train_size, val_size, test_size)
        elif img_type.startswith('complet_mnist'):
          digito = int(img_type[-1])
          self.train,self.ytrain,self.test, self.ytest =self.transform_mnist_jax(digito, train_size, val_size, test_size)
          self.val = 0
          self.yval = 0
        self.jax_train = jnp.array(self.train).astype(jnp.int8)
        self.jax_ytrain = jnp.array(self.ytrain).astype(jnp.int8)

        self.jax_val = jnp.array(self.val).astype(jnp.int8)
        self.jax_yval = jnp.array(self.yval).astype(jnp.int8)

        self.jax_test = jnp.array(self.test).astype(jnp.int8)
        self.jax_ytest = jnp.array(self.ytest).astype(jnp.int8)
        print('------------------------------------------------------------------')
        print('Imagens Carregadas')
        print('------------------------------------------------------------------')
        #return self.jax_train, self.jax_ytrain, self.jax_val, self.jax_yval, self.jax_test, self.jax_ytest
    
    def get_image(self, train_size, val_size, test_size):
      '''
        retornar imagens train test e val numeros com ruidos
      '''
      train, ytrain = np.array(self.get_images(train_size, 'train'))
      val, yval = np.array(self.get_images(val_size, 'val'))
      test, ytest = np.array(self.get_images(test_size, 'test'))
      return train, ytrain, val, yval, test, ytest

    def _convert_binary(self, img, y):
          (_, img_bin) = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
          img_bin = img_bin.astype('float64')
          img_bin[(img_bin==0)]=1
          #img_bin[(img_bin==255)]=-1
          if y:
            img_bin[(img_bin==255)]=0
          else:
            img_bin[(img_bin==255)]=-1
          return img_bin

    def get_images(self, img_size, img_name):
        ximg = []
        yimg = []
        img_path_x = f'./data/x'
        img_path_y = f'./data/y'
        for img in range(1,img_size+1):
            x = cv2.imread(f'{img_path_x}/{img_name}{img:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            y = cv2.imread(f'{img_path_y}/{img_name}{img:02d}.jpg', cv2.IMREAD_GRAYSCALE)
            ximg.append(self._convert_binary(x, False))
            yimg.append(self._convert_binary(y, True))
        return ximg, yimg
    
    def load_data(self, start_idx, end_idx, path):
        data_list = []
        for i in range(start_idx, end_idx):
            file_path = os.path.join(path, f'x_{i}.csv')
            data = pd.read_csv(file_path, header=0).values  # Ignora a primeira linha
            data_zero = jnp.where(data == 0, -1, data)
            data_list.append(jnp.array(data_zero, dtype=jnp.int8))
        return jnp.array(data_list)

    # Função para carregar os rótulos
    def load_labels(self, start_idx, end_idx, path,flg_sp):
        if flg_sp==1:
            img_name = 'y_sp_'
        elif flg_sp==2:
            img_name = 'y_sp1_'
        elif flg_sp==3:
            img_name = 'y_sp2_'
        elif flg_sp==4:
            img_name = 'y_sp3_'
        elif flg_sp==5:
            img_name = 'y_sp4_'
        else:
            img_name = 'y_'
        label_list = []
        for i in range(start_idx, end_idx):
            file_path = os.path.join(path, f'{img_name}{i}.csv')
            labels = pd.read_csv(file_path, header=0).values  # Ignora a primeira linha
            label_list.append(jnp.array(labels, dtype=jnp.int8))
        return jnp.array(label_list)

    def transform_mnist_digit(self, digito, train_size, val_size, test_size):
        np.random.seed(0)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train_bin = np.where(x_train > 100, 1, -1)
        #x_test_bin = np.where(x_test > 100, 1, -1)
        # Filtrar as imagens que têm o valor digito
        train_dig_idx = np.where(y_train == digito)[0]
        #test_dig_idx = np.where(y_test == digito)[0]

        # Selecionar as imagens restantes
        train_non_dig_idx = np.where(y_train != digito)[0]
        #test_non_dig_idx = np.where(y_test != digito)[0]

        # Garantir pelo menos metade das imagens com valor digito
        num_train_zeros = (train_size+val_size+test_size)//3
        #num_train_zeros = (train_size+val_size+test_size)//2
        #num_test_zeros = (val_size+test_size)//2

        train_selected_zeros_idx = np.random.choice(train_dig_idx, num_train_zeros, replace=False)
        #test_selected_zeros_idx = np.random.choice(test_dig_idx, num_test_zeros, replace=False)

        # Selecionar aleatoriamente imagens restantes para completar o total
        num_train_non_zeros = (train_size+val_size+test_size) - num_train_zeros
        #num_test_non_zeros = (val_size+test_size) - num_test_zeros

        train_selected_non_zeros_idx = np.random.choice(train_non_dig_idx, num_train_non_zeros, replace=False)
        #test_selected_non_zeros_idx = np.random.choice(test_non_dig_idx, num_test_non_zeros, replace=False)

        # Combinar os índices selecionados
        train_idx = np.concatenate([train_selected_zeros_idx, train_selected_non_zeros_idx])
        #test_idx = np.concatenate([test_selected_zeros_idx, test_selected_non_zeros_idx])
        #test_idx = concat_test_idx[:5000]
        #val_idx = concat_test_idx[1000:2000]

        # Embaralhar os índices para garantir que os zeros não estão todos juntos
        np.random.shuffle(train_idx)
        #np.random.shuffle(test_idx)
        #np.random.shuffle(val_idx)

        # Selecionar os dados baseados nos índices embaralhados
        x_train_jax = jnp.array(x_train_bin[train_idx][:train_size])
        y_train_jax = jnp.array(y_train[train_idx][:train_size])

        x_val_jax = jnp.array(x_train_bin[train_idx][train_size:(train_size+val_size)])
        y_val_jax = jnp.array(y_train[train_idx][train_size:(train_size+val_size)])

        x_test_jax = jnp.array(x_train_bin[train_idx][(train_size+val_size):(train_size+val_size+test_size)])
        y_test_jax = jnp.array(y_train[train_idx][(train_size+val_size):(train_size+val_size+test_size)])

        # Modificar os valores dos rótulos para a condição especificada
        y_train_jax = jnp.where(y_train_jax == digito, 1, 0)
        y_test_jax = jnp.where(y_test_jax == digito, 1, 0)
        y_val_jax = jnp.where(y_val_jax == digito, 1, 0)
        return x_train_jax, y_train_jax, x_val_jax, y_val_jax, x_test_jax, y_test_jax
    
    def transform_mnist_jax(self, digito, train_size, val_size, test_size):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train_bin = np.where(x_train > 100, 1, -1)
        x_test_bin = np.where(x_test > 100, 1, -1)

        x_train_jax = jnp.array(x_train_bin)
        y_train_jax = jnp.array(y_train)

        x_test_jax = jnp.array(x_test_bin)
        y_test_jax = jnp.array(y_test)

        # Modificar os valores dos rótulos para a condição especificada
        y_train_jax = jnp.where(y_train_jax == digito, 1, 0)
        y_test_jax = jnp.where(y_test_jax == digito, 1, 0)
        return x_train_jax, y_train_jax, x_test_jax, y_test_jax


class load_window:
  def __init__(self, path_window, path_joint, path_joint_shape):
    self.W = jnp.load(path_window, allow_pickle=True)#.astype(jnp.int8)
    self.joint = jnp.load(path_joint, allow_pickle=True)#.astype(jnp.int8)
    self.joint_shape = jnp.load(path_joint_shape, allow_pickle=True)#.astype(jnp.int8)
