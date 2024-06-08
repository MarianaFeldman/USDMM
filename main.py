 #main.py
import USDMM_class 
import jax
import numpy as np
import cv2

def main():
    USDMM_class.WOMC_DATA(
        nlayer = 2, # INT -> Number of operators in each layer
        wlen = 5,  # INT -> Size of the operator (wlen*wlen)
        train_size = 30, # INT -> Number of images to train
        val_size = 10, # INT -> Number of images to validate
        test_size = 10, # INT -> Number of images to test
        error_type = 'iou', # 'mae' / 'iou' -> type of error
        neighbors_sample = 8, # INT/False -> Number of neighbors to sort
        epoch_f = 100, # INT -> Number of epochs for the boolean function lattice (fixed windows)
        epoch_w = 20, # INT -> Number of epochs for the windows lattice
        batch = 10, # INT -> Batch size
        path_results = 'results_V5', # STR -> file where we want to save the results
        name_save='_V5', # STR -> pos fixed name for the results saved
        seed = 0, #INT -> seed for reproducibilit
        early_stop_round_f = 1e6, #20, #INT -> max number of epochs without changes in the boolean function lattice
        early_stop_round_w = 1e6 #10 #INT -> max number of epochs without changes in the windows lattice
    )
    import USDMM_JAX as USDMM
    print(USDMM.WOMC.jax_train.shape)
    print(jax.devices())
    print(USDMM.WOMC.dict_matrices['W'].shape)
    #USDMM.fit()
    #print(USDMM.WOMC.jax_train.shape)
    #USDMM._save_results_in(USDMM.WOMC.jax_train, 'train', 0, 0)
    #W_img = jax.device_get(USDMM.WOMC.jax_train) 
    #for img in range(1,len(W_img)+1):
    #    x = np.array(W_img[img-1]).astype(float)
    #    x[(x==-1)]=255
    #    x[(x==1)]=0
    #    cv2.imwrite(f'/app/output/train_{img:02d}{USDMM.WOMC.name_save}.jpg', x)
    #x = cv2.imread(f'/app/data/x/train01.jpg', cv2.IMREAD_GRAYSCALE)
    #cv2.imwrite(f'/app/output/train_01.jpg', x)
    #print("Imagem Salva!")
if __name__ == "__main__":
    main()

'''
docker build -t jax-app .
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" jax-app
python3 main.py

docker ps
docker cp Documents/GitHub/USDMM/. e52a8931887b:/workspace/

'''