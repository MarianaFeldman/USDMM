import USDMM

WOMC = USDMM.WOMC(
    new = True, # True/ STR -> If True inicialize with the cross operator/ If STR it will opens the file with the name passed
    nlayer = 2, # INT -> Number of operators in each layer
    wlen = 3,  # INT -> Size of the operator (wlen*wlen)
    train_size = 30, # INT -> Number of images to train 
    val_size = 10, # INT -> Number of images to validate
    test_size = 10, # INT -> Number of images to test
    error_type = 'iou', # 'mae' / 'iou' -> type of error
    neighbors_sample = 10, # INT/False -> Number of neighbors to sort
    epoch_f = 500, # INT -> Number of epochs for the boolean function lattice (fixed windows)
    epoch_w = 50, # INT -> Number of epochs for the windows lattice
    batch = 1, # INT -> Batch size
    path_results = 'results_V1', # STR -> file where we want to save the results
    name_save='_V1', # STR -> pos fixed name for the results saved
    seed = 0, #INT -> seed for reproducibilit
    parallel = True, # True/False -> use parallel (True) or sequential (False)
    early_stop_round_f = 50, #INT -> max number of epochs without changes in the boolean function lattice
    early_stop_round_w = 10 #INT -> max number of epochs without changes in the windows lattice
)
WOMC.fit()
