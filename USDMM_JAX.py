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
import USDMM_class as USDMM

#directory for colab
#work_directory = '.'
#work_output = 'output_colab' 

#directory for linux
work_directory = '/app/'
work_output = f'{work_directory}output' 

def load_from_file():
    with open('./data/parameters.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

loaded_data = load_from_file()

WOMC = USDMM.WOMC_JAX(
    nlayer = loaded_data['nlayer'], # INT -> Number of operators in each layer
    wlen = loaded_data['wlen'],  # INT -> Size of the operator (wlen*wlen)
    train_size = loaded_data['train_size'], # INT -> Number of images to train
    val_size = loaded_data['val_size'], # INT -> Number of images to validate
    test_size = loaded_data['test_size'], # INT -> Number of images to test
    img_type = loaded_data['img_type'], # STR -> 'img_n' / 'gl' / 'classification' / 'GoL'
    error_type = loaded_data['error_type'], # 'mae' / 'iou' -> type of error
    neighbors_sample_f = loaded_data['neighbors_sample_f'], # INT/False -> Number of neighbors to sort
    neighbors_sample_w = loaded_data['neighbors_sample_w'], # INT/False -> Number of neighbors to sort
    epoch_f = loaded_data['epoch_f'], # INT -> Number of epochs for the boolean function lattice (fixed windows)
    epoch_w = loaded_data['epoch_w'], # INT -> Number of epochs for the windows lattice
    batch = loaded_data['batch'], # INT -> Batch size
    path_results = loaded_data['path_results'], # STR -> file where we want to save the results
    name_save = loaded_data['name_save'], # STR -> pos fixed name for the results saved
    seed = loaded_data['seed'], #INT -> seed for reproducibilit
    #parallel = 'gpu', # True/False -> use parallel (True) or sequential (False)
    early_stop_round_f = loaded_data['early_stop_round_f'], #20, #INT -> max number of epochs without changes in the boolean function lattice
    early_stop_round_w = loaded_data['early_stop_round_w'], #10 #INT -> max number of epochs without changes in the windows lattice
    w_ini = loaded_data['w_ini'] #inicial operator
    #img_file = loaded_data['img_file']
)


# Criar Joint e W-Matrices
def create_joint_in(key, ni, joint_max_size):
    joint = jax.random.randint(key, (ni,), 0, 2)
    pad_size = joint_max_size - joint.shape[0]
    padding_config = [(0, max(0, pad_size), 0)]
    padded_joint = jax.lax.pad(joint, 0, padding_config)
    #expanded_joint = jnp.expand_dims(padded_joint, axis=tuple(range(1, WOMC.wlen)))
    return padded_joint.astype(jnp.int8) #expanded_joint
create_joint_jit = jax.jit(create_joint_in, static_argnums=(1,2))

def create_joint(W_arrays):
    joints = []
    joint_shape = []
    for W in W_arrays:
    #WOMC.key,subkey = jax.random.split(WOMC.key)
        key = jax.random.PRNGKey(WOMC.seed)
        WOMC.seed+=1
        ni = 2**int(jnp.sum(W))
        joints.append(create_joint_jit(key, ni, WOMC.joint_max_size))
        joint_shape.append(ni)
    return jnp.array(joints), jnp.array(joint_shape)

def create_layer_joint(W):
    key = jax.random.PRNGKey(WOMC.seed)
    WOMC.seed+=1
    ni = 2**int(jnp.sum(W))
    return create_joint_jit(key, ni, WOMC.joint_max_size), ni

def new_joint(joint_old):
    joint = joint_old.copy()
    diff = WOMC.joint_max_size - joint.shape[1]
    if diff>0:
        pad_width = [(0, 0), (0, diff)]
        return jnp.pad(joint, pad_width, mode='constant', constant_values=0)
    elif diff<0:
        return joint[:, :WOMC.joint_max_size]
    else:
        return joint
    

def create_w_matrices_for_dict(W, joint_function, wlen):
    indices = jnp.where(W == 1)[0]
    def apply_minterm(minterm):
        matrix = W.at[indices].set(minterm.astype(W.dtype))
        return matrix.reshape((wlen, wlen))
    return jax.vmap(apply_minterm)(joint_function)

def create_w_matrices_dict(W, wlen):
    ni = jnp.sum(W)
    binary_combinations = jnp.array(list(product([-1, 1], repeat=int(ni))), dtype=W.dtype)
    W_matrices = create_w_matrices_for_dict(W, binary_combinations, wlen)
    return W_matrices

def create_w_matrices(W):
    matrices = []
    #W_arrays = jnp.array(dict_matrices["W"])
    for k in range(WOMC.nlayer):
        #ix = jnp.where(jnp.all(W_arrays == W[k], axis=1))[0][0]
        #matrices_jax = jnp.array(dict_matrices["W_matrices"][ix])
        matrices_jax = create_w_matrices_dict(W[k], WOMC.wlen)
        pad_size = WOMC.joint_max_size - matrices_jax.shape[0]
        padding_config = [(0, pad_size, 0), (0, 0, 0), (0, 0, 0)]
        padding_value = jnp.array(0, dtype=jnp.int8)
        padded_matrices = jax.lax.pad(matrices_jax, padding_value, padding_config)
        matrices.append(padded_matrices)
        #matrices.append(matrices_jax)
    return jnp.array(matrices)

@jax.jit
def mult_w_matrices(W_matrices_all, joint):
    #return jnp.array([W_matrices_all[k] * joint[k] for k  in range(WOMC.nlayer)])
    #matrices = []
    #for k in range(WOMC.nlayer):
    #    ix = jnp.where(joint[k] == 1)[0]
    #    matrices.append(W_matrices_all[k][ix])
    #return matrices
    return W_matrices_all * joint[:, :, None, None]

# Aplicar um W-Operador
def convolve_kernel(img, kernel, bias):
    img_r = convolve2d_jax(img, kernel, mode='same')-bias
    img_i = (img_r > 0).astype(jnp.int8)
    del img_r
    return  img_i
convolve_kernel_vmap = jax.vmap(convolve_kernel, in_axes=(None, 0, None))

def apply_convolve_jax(img, W_matrices, bias):
    img_c = jnp.sum(convolve_kernel_vmap(img,W_matrices, bias), axis=0, dtype=jnp.int8)
    img_c = jnp.where(img_c == 0, jnp.int8(-1), img_c)
    return img_c[WOMC.increase:img_c.shape[0]-WOMC.increase, WOMC.increase:img_c.shape[1]-WOMC.increase].astype(jnp.int8)
apply_convolve_vmap = jax.vmap(apply_convolve_jax, in_axes=(0, None, None))
apply_convolve_vmap_jit = jax.jit(apply_convolve_vmap)

def run_window_convolve_jax_(sample, W_matrices,W_last, layer,bias):
    Wsample = jnp.zeros((WOMC.nlayer, *sample.shape))
    sample_b = jnp.pad(sample, ((0, 0), (WOMC.increase, WOMC.increase), (WOMC.increase, WOMC.increase)), mode='constant', constant_values=-1)

    for i in range(WOMC.nlayer):
        carry_forry = (W_last[i] ,W_matrices[i], bias[i], sample_b )
        Wsample_k = jax.lax.cond(layer > i,
                                lambda c: c[0],  # True branch (using W_last)
                                lambda c: apply_convolve_vmap_jit(c[3], c[1], c[2]),  # False branch
                                carry_forry)
        sample_b = jnp.pad(jnp.array(Wsample_k), ((0, 0), (WOMC.increase, WOMC.increase), (WOMC.increase, WOMC.increase)), mode='constant', constant_values=-1)
        Wsample = Wsample.at[i].set(Wsample_k)
    return Wsample.astype(jnp.int8)

def run_window_convolve_jax(sample, W_matrices,bias):
    sample_b = jnp.pad(sample, ((0, 0), (WOMC.increase, WOMC.increase), (WOMC.increase, WOMC.increase)), mode='constant', constant_values=-1)
    Wsample=[]
    for i in range(WOMC.nlayer):
        Wsample_k = apply_convolve_vmap(sample_b, W_matrices[i], bias[i])#.astype(jnp.int8)
        sample_b = jnp.pad(jnp.array(Wsample_k), ((0, 0), (WOMC.increase, WOMC.increase), (WOMC.increase, WOMC.increase)), mode='constant', constant_values=-1)
        Wsample.append(Wsample_k)
    return jnp.array(Wsample)

def run_window_convolve(sample,y, W_matrices, layer,bias):
    sample_b = jnp.pad(sample, ((0, 0), (WOMC.increase, WOMC.increase), (WOMC.increase, WOMC.increase)), mode='constant', constant_values=-1)
    Wsample_k=[]
    Wsample=[]
    for i in range(WOMC.nlayer):
        for img in sample_b:
          img_c = jnp.sum(convolve_kernel_vmap(img,W_matrices[i], bias[i]), axis=0, dtype=jnp.int8)
          img_c = jnp.where(img_c == 0, jnp.int8(-1), img_c)
          Wsample_k.append(img_c[WOMC.increase:img_c.shape[0]-WOMC.increase, WOMC.increase:img_c.shape[1]-WOMC.increase].astype(jnp.int8))

        sample_b = jnp.pad(jnp.array(Wsample_k), ((0, 0), (WOMC.increase, WOMC.increase), (WOMC.increase, WOMC.increase)), mode='constant', constant_values=-1)
        Wsample.append(Wsample_k)
    W_hood = jnp.array(Wsample)
    error_hood = IoU(y, W_hood)
    return W_hood, error_hood
run_window_convolve_jit = jax.jit(run_window_convolve)

# Calcular o erro
def IoU(y, h):
    def single_iou(y_z, h_z):
        numerator = jnp.sum(2 * y_z * h_z) + 1
        denominator = jnp.sum(y_z + h_z) + 1
        return 1.0 - numerator / denominator

    h_z = jnp.where(h[-1] == -1, 0, h[-1])
    #y_z = jnp.where(y == -1, 0, y)
    ious = jax.vmap(single_iou)(y, h_z)
    error = jnp.mean(ious)
    return error

def window_error_generate(W_matrices, sample, y, layer, bias):
    W_hood = run_window_convolve_jax(sample, W_matrices, layer, bias)
    error_hood = IoU(y, W_hood)
    return W_hood,error_hood
window_error_generate_train = jax.jit(window_error_generate)
window_error_generate_val = jax.jit(window_error_generate)
window_error_generate_test = jax.jit(window_error_generate)
window_error_generate_batch = jax.jit(window_error_generate)

# Reticulado das funções (janela fixa)
def calculate_neighbors(joint,W_matrices_all, k, img,yimg, bias, i):
    '''
        Calculate the function window neighbor
    '''
    joint_k = jnp.array(joint[k])
    joint_k = jnp.where(joint[k][i] == 1, joint_k.at[i].set(0), joint_k)
    joint_k = jnp.where(joint[k][i] == 0, joint_k.at[i].set(1), joint_k)
    joint_temp = copy.deepcopy(joint)
    joint_temp = joint_temp.at[k].set(joint_k)

    W_matrices = mult_w_matrices(W_matrices_all, joint_temp)
    _,error_hood = window_error_generate_batch(W_matrices, img, yimg, k, bias)
    return [error_hood, joint_temp, [k,i]]
calculate_neighbors_jax0 = jax.jit(calculate_neighbors, static_argnums=(2,))  
calculate_neighbors_vmap = jax.vmap(calculate_neighbors_jax0, in_axes=(None, None, 0,None, None,None,0))
calculate_neighbors_jax = jax.jit(calculate_neighbors_vmap)

def get_batches(imgX,imgY, key):
    idx = jax.random.permutation(key, WOMC.train_size)

    imgX_shuffled = imgX[idx]
    imgY_shuffled = imgY[idx]

    imgX_batches = [imgX_shuffled[i * WOMC.batch:(i + 1) * WOMC.batch] for i in range(WOMC.num_batches)]
    imgY_batches = [imgY_shuffled[i * WOMC.batch:(i + 1) * WOMC.batch] for i in range(WOMC.num_batches)]
    return jnp.array(imgX_batches), jnp.array(imgY_batches)
get_batches_jit = jax.jit(get_batches)

def get_random_neighbors(joint_shape):
    neighbors_to_visit = []
    ix = []
    for k, sublist_length in enumerate(joint_shape):
        #sublist_length = joint_shape[k]
        if not WOMC.neighbors_sample_f or WOMC.neighbors_sample_f >= sublist_length:
            neighbors_to_visit.extend(range(sublist_length))
            ix.extend([k] * sublist_length.item())
        else:
            sample_indices = random.sample(range(sublist_length), WOMC.neighbors_sample_f)
            neighbors_to_visit.extend(sample_indices)
            ix.extend([k] * len(sample_indices))
    return jnp.array(neighbors_to_visit), jnp.array(ix)

def batch_run(b,joint,joint_shape, W_matrices,W_matrices_all,train_b,ytrain_b,bias):
    #Wtrain,_ =  window_error_generate_batch(W_matrices,train_b[b],ytrain_b[b], 0, bias)
    neighbors_to_visit, ix = get_random_neighbors(joint_shape)

    error_ep_f_hist = calculate_neighbors_jax(joint, W_matrices_all,ix, train_b[b],ytrain_b[b],bias,neighbors_to_visit)
    ix = jnp.lexsort((error_ep_f_hist[2][1], error_ep_f_hist[2][0],error_ep_f_hist[0]))[0]
    return error_ep_f_hist[1][ix]

def get_error_window(W, joint,joint_shape, ep_w):
    '''
        Find the function with the smallest error
    '''
    W_matrices_all = create_w_matrices(W)#, WOMC.dict_matrices)
    W_matrices = mult_w_matrices(W_matrices_all, joint)
    bias = (jnp.sum(W, axis=1) - 1).astype(jnp.int8)

    if WOMC.batch>=WOMC.train_size:
        train_b = [jnp.array(WOMC.jax_train)]
        ytrain_b = [jnp.array(WOMC.jax_ytrain)]
    flg = 0
    epoch_min = 0
    W_size = jnp.sum(W, axis=1)

    _,w_error =  window_error_generate_train(W_matrices,WOMC.jax_train, WOMC.jax_ytrain, 0, bias)
    WOMC.time_test = {"ep":[],"time":[]}

    for ep in range(WOMC.epoch_f):
        start_time = time()
        if WOMC.batch<WOMC.train_size:
            key = jax.random.PRNGKey(WOMC.seed)
            WOMC.seed+=1
            train_b, ytrain_b = get_batches_jit(WOMC.jax_train,WOMC.jax_ytrain, key)

        for b in range(WOMC.num_batches):
            joint = batch_run(b,joint,joint_shape,W_matrices,W_matrices_all,train_b,ytrain_b,bias)

        W_matrices = mult_w_matrices(W_matrices_all, joint)
        W_train,w_error_min =  window_error_generate_train(W_matrices,WOMC.jax_train, WOMC.jax_ytrain, 0, bias)
        WOMC.error_ep_f["W_key"].append(WOMC.windows_visit)
        WOMC.error_ep_f["epoch_w"].append(0)
        WOMC.error_ep_f["epoch_f"].append(ep)
        WOMC.error_ep_f["error"].append(w_error_min)
        WOMC.error_ep_f["time"].append((time() -  WOMC.start_time))
        for i in range(WOMC.nlayer):
            WOMC.error_ep_f[f"window_size_{i}"].append(W_size[i])

        if w_error_min < w_error:
            w_error = w_error_min
            joint_min = copy.deepcopy(joint)
            flg=1
            epoch_min = ep

        if (ep-epoch_min)>WOMC.early_stop_round_f :
            break
        WOMC.time_test['ep'].append(ep)
        WOMC.time_test['time'].append((time() - start_time))
    if flg==1:
        joint = copy.deepcopy(joint_min)
    W_matrices = mult_w_matrices(W_matrices_all, joint)

    W_val,error_val =  window_error_generate_val(W_matrices,WOMC.jax_val, WOMC.jax_yval, 0, bias)
   #_save_results_in(W_val, 'val',1, ep)
    error = jnp.array([w_error, error_val])
    return joint, error, epoch_min, W_train, W_val

def batch_run_fixed_window(b,joint,joint_shape, W_matrices,W_matrices_all,train_b,ytrain_b,bias):
    error_ep_f_hist = []
    for k in range(WOMC.nlayer):
        for i in range(joint_shape[k]):
            error_ep_f_hist.append(calculate_neighbors_jax0(joint,W_matrices_all, k, train_b[b],ytrain_b[b], bias, i))
    min_error_idx = min(range(len(error_ep_f_hist)), key=lambda i: error_ep_f_hist[i][0])
    min_error_hood = error_ep_f_hist[min_error_idx][0]
    #corresponding_joint_temp = error_ep_f_hist[min_error_idx][1]
    #corresponding_k_i = error_ep_f_hist[min_error_idx][2]

    return error_ep_f_hist[min_error_idx][1]

def get_error_fixed_window(W, joint, joint_shape, j_rand):
    '''
        Find the function with the smallest error for one window, search all neighbors without VMAP
    '''
    start_time_g = time()
    max_w = jnp.max(jnp.sum(W, axis=1))
    WOMC.joint_max_size = 2**max_w.item()
    if j_rand:
      joint,joint_shape = create_joint(W)
    W_matrices_all = create_w_matrices(W)#, WOMC.dict_matrices)
    W_matrices = mult_w_matrices(W_matrices_all, joint)
    bias = (jnp.sum(W, axis=1) - 1).astype(jnp.int8)

    if WOMC.batch>=WOMC.train_size:
        train_b = [jnp.array(WOMC.jax_train)]
        ytrain_b = [jnp.array(WOMC.jax_ytrain)]
    flg = 0
    epoch_min = 0
    error_ep_f = {"epoch":[], "error":[], "time":[] }
    _,w_error =  window_error_generate_train(W_matrices,WOMC.jax_train, WOMC.jax_ytrain, 0, bias)
    time_min = (time() -  start_time_g) / 60
    print(f'Time: {time_min:.2f} | Epoch 0 / {WOMC.epoch_f} - start Train error: {w_error:.4f}')
    for ep in range(WOMC.epoch_f):
        start_time = time()
        if WOMC.batch<WOMC.train_size:
            key = jax.random.PRNGKey(WOMC.seed)
            WOMC.seed+=1
            train_b, ytrain_b = get_batches_jit(WOMC.jax_train,WOMC.jax_ytrain, key)

        for b in range(WOMC.num_batches):
            joint = batch_run_fixed_window(b,joint,joint_shape,W_matrices,W_matrices_all,train_b,ytrain_b,bias)

        W_matrices = mult_w_matrices(W_matrices_all, joint)
        W_train,w_error_min =  window_error_generate_train(W_matrices,WOMC.jax_train, WOMC.jax_ytrain, 0, bias)
        #WOMC.error_ep_f["W_key"].append(WOMC.windows_visit)
        #WOMC.error_ep_f["epoch_w"].append(0)
        error_ep_f["epoch"].append(ep)
        error_ep_f["error"].append(w_error_min)
        error_ep_f["time"].append((time() -  start_time_g))

        if w_error_min < w_error:
            w_error = w_error_min
            joint_min = copy.deepcopy(joint)
            flg=1
            epoch_min = ep

        if (ep-epoch_min)>WOMC.early_stop_round_f :
            break
        if (ep) % 10 == 0:
          time_min = (time() - start_time_g)/60
          print(f'Time: {time_min:.2f} | Epoch {ep} / {WOMC.epoch_f} - Train error: {w_error_min:.4f} | Min-Epoch {epoch_min} / Min-Error {w_error}')
    if flg==1:
        joint = copy.deepcopy(joint_min)
    W_matrices = mult_w_matrices(W_matrices_all, joint)

    W_val,error_val =  window_error_generate_val(W_matrices,WOMC.jax_val, WOMC.jax_yval, 0, bias)
    W_test,error_test =  window_error_generate_test(W_matrices,WOMC.jax_test, WOMC.jax_ytest, 0, bias)
   #_save_results_in(W_val, 'val',1, ep)
    error = jnp.array([w_error, error_val, error_test])
    print('-------------------------------------------------')
    print(f'End of testing fixed window ')
    print(f'Min-Epoch {epoch_min} - Train error: {w_error:.4f} / Validation error: {error_val:.4f} / Test error: {error_test:.4f}')
    save_window_fixed(joint,joint_shape, W)
    save_results_complet_fixed(W_train, W_val,W_test )
    total_time = (time() - start_time_g)/60
    return error_ep_f, error, joint, total_time
    #return joint, error, epoch_min, W_train, W_val

@jax.jit
def check_w_hist(W,k, w_hist):
  cond1 = jnp.any(jnp.all(W[k] == WOMC.windows_continuos, axis=1))
  w_hist_stack = jnp.stack(w_hist)
  cond2 = jnp.any(jnp.all(W == w_hist_stack, axis=(1,2)))
  return cond1 , cond2


def neighboors_func_(W,joint,joint_shape, ep_w, k, i):
    W_temp = jnp.array(W)
    W_temp = jnp.where(W[k][i] == 1, W_temp.at[k,i].set(0), W_temp)
    W_temp = jnp.where(W[k][i] == 0, W_temp.at[k,i].set(1), W_temp)

    cond1, cond2 = check_w_hist(W_temp,k, WOMC.w_hist["W"])

    if cond1 and not cond2:
        WOMC.windows_visit+=1
        joint_temp = copy.deepcopy(joint)
        joint_shape_temp = copy.deepcopy(joint_shape)
        joint_k, shape_k = create_layer_joint(W_temp[k])
        joint_temp = joint_temp.at[k].set(joint_k)
        joint_shape_temp = joint_shape_temp.at[k].set(shape_k)

        joint_temp, w_error, f_epoch_min, W_train, W_val = get_error_window(W_temp, joint_temp,joint_shape_temp, ep_w)

        WOMC.w_hist["W_key"].append(WOMC.windows_visit)
        WOMC.w_hist["W"].append(W_temp)
        WOMC.w_hist["error"].append(w_error)
        WOMC.w_hist["f_epoch_min"].append(f_epoch_min)
        return w_error[0], w_error[1], W_temp, joint_temp, joint_shape_temp, W_train, W_val#, [k,i]]

    return jnp.inf, jnp.inf, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan#, [k,i]]

def neighboors_func(W_temp,joint,joint_shape, ep_w, k):
    WOMC.windows_visit+=1
    joint_temp = joint.copy()
    joint_shape_temp = joint_shape.copy()
    joint_k, shape_k = create_layer_joint(W_temp[k])
    joint_temp = joint_temp.at[k].set(joint_k)
    joint_shape_temp = joint_shape_temp.at[k].set(shape_k)

    joint_temp, w_error, f_epoch_min, W_train, W_val = get_error_window(W_temp, joint_temp,joint_shape_temp, ep_w)

    WOMC.w_hist["W_key"].append(WOMC.windows_visit)
    WOMC.w_hist["W"].append(W_temp)
    WOMC.w_hist["error"].append(w_error)
    WOMC.w_hist["f_epoch_min"].append(f_epoch_min)
    return w_error[0], w_error[1], W_temp, joint_temp, joint_shape_temp, W_train, W_val#, [k,i]]

def get_w_neighbors(W):
    W_tiled = jnp.tile(W, (WOMC.wsize, 1))
    W_neighbors = jnp.where(WOMC.identity_matrix == 1, 1 - W_tiled, W_tiled)
    W_neighbors_mask = jnp.any(jnp.all(W_neighbors[:, None, :] == WOMC.windows_continuos, axis=2), axis=1)
    W_neighbors_valid = W_neighbors[W_neighbors_mask]
    n_neighbors = W_neighbors_valid.shape[0]
    
    if not WOMC.neighbors_sample_w or WOMC.neighbors_sample_w >= n_neighbors:
        return W_neighbors_valid
    else:
        sample_indices = random.sample(range(n_neighbors), WOMC.neighbors_sample_w)
        sample_indices_jax = jnp.array(sample_indices)
        return W_neighbors_valid[sample_indices_jax]

#check_neighboors_vmap = jax.vmap(check_neighboors, in_axes=(None,None,None, None, 0,0))
def check_neighboors(W,joint,joint_shape, ep_w):
    w_error0 = []
    w_error1 = []
    W_temp_l = []
    j_temp_l = []
    joint_shape_temp_l = []
    W_train_list = [] 
    W_val_list = []
    for k in range(WOMC.nlayer):
        W_neighbors_to_visit = get_w_neighbors(W[k])
        W_temp = jnp.array(W)
        for W_k in W_neighbors_to_visit:
            W_temp = W_temp.at[k].set(W_k)
            
            WOMC.windows_visit+=1
            max_w = jnp.max(jnp.sum(W_temp, axis=1))
            WOMC.joint_max_size = 2**max_w.item()
            joint_temp = new_joint(joint)
            joint_shape_temp = joint_shape.copy()
            joint_k, shape_k = create_layer_joint(W_temp[k])
            joint_temp = joint_temp.at[k].set(joint_k)
            joint_shape_temp = joint_shape_temp.at[k].set(shape_k)

            joint_temp, w_error, f_epoch_min, W_train, W_val = get_error_window(W_temp, joint_temp,joint_shape_temp, ep_w)

            WOMC.w_hist["W_key"].append(WOMC.windows_visit)
            WOMC.w_hist["W"].append(W_temp)
            WOMC.w_hist["error"].append(w_error)
            WOMC.w_hist["f_epoch_min"].append(f_epoch_min)

            #w0, w1, wt, jt, jst, wtt, wvt = neighboors_func(W,joint,joint_shape, ep_w, k)
            w_error0.append(w_error[0])
            w_error1.append(w_error[1])
            W_temp_l.append(W_temp)
            j_temp_l.append(joint_temp)
            joint_shape_temp_l.append(joint_shape_temp)
            W_train_list.append(W_train)
            W_val_list.append(W_val)
    ix_error = jnp.argmin(jnp.array(w_error1))
    W_error = jnp.array([w_error0[ix_error], w_error1[ix_error]])
    W_min = W_temp_l[ix_error]
    joint_min = j_temp_l[ix_error]
    joint_shape_min = joint_shape_temp_l[ix_error]
    W_train = W_train_list[ix_error]
    W_val = W_val_list[ix_error]
    return W_error, W_min, joint_min, joint_shape_min, W_train, W_val

def fit():
    WOMC.start_time = time()

    #w_ini = jnp.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).astype(jnp.int8)
    W = jnp.array([WOMC.w_ini.copy() for _ in range(WOMC.nlayer)])
    max_w = jnp.max(jnp.sum(W, axis=1))
    WOMC.joint_max_size = 2**max_w.item()
    joint,joint_shape = create_joint(W)

    joint, error, f_epoch_min, W_train, W_val= get_error_window(W, joint,joint_shape, 0)

    WOMC.w_hist["W_key"].append(WOMC.windows_visit)
    WOMC.w_hist["W"].append(W)
    WOMC.w_hist["error"].append(error)
    WOMC.w_hist["f_epoch_min"].append(f_epoch_min)

    ep_min=0
    error_min = copy.deepcopy(error)
    W_min = copy.deepcopy(W)
    joint_min = copy.deepcopy(joint)
    error_ep = {"epoch":[],"error_train":[], "error_val":[]}
    error_ep['epoch'].append(0)
    error_ep['error_train'].append(error[0])
    error_ep['error_val'].append(error[1])
    
    time_min = (time() -  WOMC.start_time) / 60
    print(f'Time: {time_min:.2f} | Epoch 0 / {WOMC.epoch_w} - start Validation error: {error[1]:.4f}')
    for ep in range(1,WOMC.epoch_w+1):
        error, W, joint, joint_shape, Wtrain, Wval = check_neighboors(W,joint,joint_shape, ep)
        if error[1]<error_min[1]:
          ep_min = ep
          #save_window_in(joint,joint_shape, W, ep)
          error_min = copy.deepcopy(error)
          W_min = copy.deepcopy(W)
          joint_min = copy.deepcopy(joint)
          joint_shape_min = copy.deepcopy(joint_shape)
          Wtrain_min = copy.deepcopy(Wtrain)
          Wval_min = copy.deepcopy(Wval)

          #_save_results_in(Wval, 'val',1, ep)

          #W_matrices_all = create_w_matrices(W_min, WOMC.dict_matrices)
          #W_matrices = mult_w_matrices(W_matrices_all, joint_min)
          #bias = (jnp.sum(W, axis=1) - 1).astype(jnp.int8)
          #W_last = jnp.zeros((WOMC.nlayer,) + WOMC.jax_val.shape).astype(jnp.int8)
          #Wtest,error_test =  window_error_generate_val(W_matrices,WOMC.jax_test, WOMC.jax_ytest, W_last, 0, bias)
          #Wtrain = WOMC.run_window_convolve(WOMC.train, WOMC.train_size,W_matrices, 0, 0, bias)
          #Wval = WOMC.run_window_convolve(WOMC.val, WOMC.val_size,W_matrices, 0, 0, bias)
          #Wtest = WOMC.run_window_convolve(WOMC.test, WOMC.test_size,W_matrices, 0, 0, bias)
          #self.save_results_complet_in(Wtrain, Wval, Wtest, f'_epoch{ep}')
            
        error_ep['epoch'].append(ep)
        error_ep['error_train'].append(error[0])
        error_ep['error_val'].append(error[1])
        time_min = (time() -  WOMC.start_time) / 60
        print(f'Time: {time_min:.2f} | Epoch {ep} / {WOMC.epoch_w} - Validation error: {error[1]:.4f} | Min-Epoch {ep_min}')
        if (ep-ep_min)>WOMC.early_stop_round_w :
            print('End by Early Stop Round')
            break
    error_train = error_min[0]
    error_val = error_min[1]
    max_w = jnp.max(jnp.sum(W_min, axis=1))
    WOMC.joint_max_size = 2**max_w.item()
    W_matrices_all = create_w_matrices(W_min)#, WOMC.dict_matrices)
    W_matrices = mult_w_matrices(W_matrices_all, joint_min)
    bias = (jnp.sum(W_min, axis=1) - 1).astype(jnp.int8)
    

    Wtest,error_test =  window_error_generate_test(W_matrices,WOMC.jax_test, WOMC.jax_ytest, 0, bias)
    print('---------------------------------------------------------')
    print('End of testing')
    end_time = time()
    time_min = (end_time -  WOMC.start_time) / 60
    print(f'Total Time: {time_min:.2f} minutes| Min-Epoch {ep_min} - Train error: {error_train:.4f} / Validation error: {error_val:.4f} / Test error: {error_test:.4f}')

    save_window(joint,joint_shape, W)
    save_results_complet(Wtrain_min, Wval_min, Wtest)
    pickle.dump(WOMC.w_hist, open(f'{work_output}/{WOMC.path_results}/W_hist{WOMC.name_save}.txt', 'wb'))
    pickle.dump(error_ep, open(f'{work_output}/{WOMC.path_results}/error_ep_w{WOMC.name_save}.txt', 'wb'))
    pickle.dump(WOMC.error_ep_f, open(f'{work_output}/{WOMC.path_results}/error_ep_f{WOMC.name_save}.txt', 'wb'))
    print(f"Window learned: {W_min}")

    return error_ep, time_min, error_train, error_val, error_test

def _save_results_in(W, img_type, k, ep = None):
    W_img = jax.device_get(W[k]) 
    for img in range(1,len(W_img)+1):
        x = np.array(W_img[img-1]).astype(np.int64)
        x[(x==-1)]=255
        x[(x==1)]=0
        cv2.imwrite(f'{work_output}/{WOMC.path_results}/run/{img_type}_op{k+1}_{img:02d}{WOMC.name_save}_{ep}.jpg', x)

def save_results_complet_in(Wtrain, Wval, Wtest, ep = None):
  for k in range(WOMC.nlayer):
      _save_results_in(Wtrain, 'train',k, ep)
      _save_results_in(Wval, 'val',k, ep)
      _save_results_in(Wtest, 'test',k, ep)

def _save_results(W, img_type, k):
    W_img = jax.device_get(W[k]) 
    for img in range(1,len(W_img)+1):
        x = np.array(W_img[img-1]).astype(np.int64)
        x[(x==-1)]=255
        x[(x==1)]=0
        cv2.imwrite(f'{work_output}/{WOMC.path_results}/trained/{img_type}_op{k+1}_{img:02d}{WOMC.name_save}.jpg', x)


def save_results_complet(Wtrain, Wval, Wtest):
    for k in range(WOMC.nlayer):
        _save_results(Wtrain, 'train',k)
        _save_results(Wval, 'val',k)
        _save_results(Wtest, 'test',k)

def _save_results_fixed(W, img_type, k):
    W_img = jax.device_get(W[k]) 
    for img in range(1,len(W_img)+1):
        x = np.array(W_img[img-1]).astype(np.int64)
        x[(x==-1)]=255
        x[(x==1)]=0
        cv2.imwrite(f'{work_output}/{WOMC.path_results}/trained/{img_type}_op{k+1}_{img:02d}{WOMC.name_save}_fixed.jpg', x)

def save_results_complet_fixed(Wtrain, Wval, Wtest):
    
    for k in range(WOMC.nlayer):
        _save_results_fixed(Wtrain, 'train',k)
        _save_results_fixed(Wval, 'val',k)
        _save_results_fixed(Wtest, 'test',k)
    Wtrain_np = np.array(Wtrain)
    Wval_np = np.array(Wval)
    Wtest_np = np.array(Wtest)
    name_file = f'{work_output}/{WOMC.path_results}/trained/train_fixed.txt'
    pickle.dump(np.where(Wtrain_np==-1,0, Wtrain_np), open(name_file, 'wb'))
    name_file = f'{work_output}/{WOMC.path_results}/trained/val_fixed.txt'
    pickle.dump(np.where(Wval_np==-1,0, Wval_np), open(name_file, 'wb'))
    name_file = f'{work_output}/{WOMC.path_results}/trained/test_fixed.txt'
    pickle.dump(np.where(Wtest_np==-1,0, Wtest_np), open(name_file, 'wb'))


def save_window_in(joint,joint_shape, W, ep):
  filename_joint = f'{work_output}/{WOMC.path_results}/run/joint{WOMC.name_save}_ep{ep}.txt'
  pickle.dump(joint, open(filename_joint, 'wb'))
  filename_joint_shape = f'{work_output}/{WOMC.path_results}/run/joint_shape{WOMC.name_save}_ep{ep}.txt'
  pickle.dump(joint_shape, open(filename_joint_shape, 'wb'))
  filename_W =f'{work_output}/{WOMC.path_results}/run/W{WOMC.name_save}_ep{ep}.txt'
  pickle.dump(W, open(filename_W, 'wb'))  

def save_window(joint,joint_shape, W):
  filename_joint = f'{work_output}/{WOMC.path_results}/trained/joint{WOMC.name_save}.txt'
  pickle.dump(joint, open(filename_joint, 'wb'))
  filename_joint_shape = f'{work_output}/{WOMC.path_results}/trained/joint_shape{WOMC.name_save}.txt'
  pickle.dump(joint_shape, open(filename_joint_shape, 'wb'))
  filename_W =f'{work_output}/{WOMC.path_results}/trained/W{WOMC.name_save}.txt'
  pickle.dump(W, open(filename_W, 'wb')) 

def save_window_fixed(joint,joint_shape, W):
  filename_joint = f'{work_output}/{WOMC.path_results}/trained/joint{WOMC.name_save}_fixed.txt'
  pickle.dump(joint, open(filename_joint, 'wb'))
  filename_joint_shape = f'{work_output}/{WOMC.path_results}/trained/joint_shape{WOMC.name_save}_fixed.txt'
  pickle.dump(joint_shape, open(filename_joint_shape, 'wb'))
  filename_W =f'{work_output}/{WOMC.path_results}/trained/W{WOMC.name_save}_fixed.txt'
  pickle.dump(W, open(filename_W, 'wb'))

#def save_results_complet_in(Wtrain, Wval, Wtest, ep = None):
#    for k in range(WOMC.nlayer):
#        _save_results_in(Wtrain, 'train',k, ep)
#        _save_results_in(Wval, 'val',k, ep)
#        _save_results_in(Wtest, 'test',k, ep)