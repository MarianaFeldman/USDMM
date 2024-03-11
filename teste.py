import USDMM_C as USDMM
from time import time
import numpy as np

if __name__ == "__main__":
    start_time = time()
    img = 50
    for p in range(1):#['parallel-func']: #['no-parallel','parallel-window','parallel-func','parallel-func-layer']:
        WOMC = USDMM.WOMC(
            new = True, # True/ STR -> If True inicialize with the cross operator/ If STR it will opens the file with the name passed
            nlayer = 3, # INT -> Number of operators in each layer
            wlen = 3,  # INT -> Size of the operator (wlen*wlen)
            train_size = img, # INT -> Number of images to train 
            val_size = 10, # INT -> Number of images to validate
            test_size = 10, # INT -> Number of images to test
            error_type = 'iou', # 'mae' / 'iou' -> type of error
            neighbors_sample = 8, # INT/False -> Number of neighbors to sort
            epoch_f = 100, # INT -> Number of epochs for the boolean function lattice (fixed windows)
            epoch_w = 50, # INT -> Number of epochs for the windows lattice
            batch = img, # INT -> Batch size
            path_results = 'results_C',#+p, # STR -> file where we want to save the results
            name_save='_C',#+p, # STR -> pos fixed name for the results saved
            seed = p, #INT -> seed for reproducibilit
            parallel = 'parallel-func', # 'no-parallel'/'parallel-window'/'parallel-func'/'parallel-func-layer' -> use parallel (True) or sequential (False)
            early_stop_round_f = 1e6, #INT -> max number of epochs without changes in the boolean function lattice
            early_stop_round_w = 1e6 #INT -> max number of epochs without changes in the windows lattice
        )


        #start_time = time()

        #WOMC.fit()
        #WOMC.results_after_fit()
        #WOMC.test_neightbors()
        WOMC.test_window()
        #WOMC.compare_times_window()
        #WOMC.compare_times_neighbor()
        
        #W_matrices = WOMC.create_w_matrices(WOMC.W, WOMC.joint)
        #bias = np.nansum(WOMC.W, axis=1) - 1
        #Wtrain,w_error = WOMC.window_error_generate_c(W_matrices, WOMC.train, WOMC.train_size, WOMC.ytrain, WOMC.error_type, 0,0, bias)
        #print(f'error: {w_error}')

    end_time = time()
    time_min = (end_time -  start_time) / 10
    #print(f'Time test.py func: {time_min:.2f}')
    print(f'Time test.py func: {time_min}')


