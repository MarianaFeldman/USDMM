<<<<<<< HEAD
# USDMM - Unrestricted Sequential Discrete Morphological Neural Networks


[![Python Version](https://img.shields.io/badge/python-3.8-brightgreen.svg)](https://www.python.org/downloads/)

Train Unrestricted Sequential Discrete Morphological Neural Networks via the stochastic lattice gradiente descent algorithm (LGDA) and the stochastic LGDA.

## Table of Contents

- [Dependencies](#dependencies)
- [Usage](#usage)
- [Features](#features)
- [Examples](#examples)
- [Contributing](#contributing)
- [Reference](#reference)

## Dependencies

numpy==1.20.1
more-itertools==8.7.0
opencv-python==4.5.3.56

## Usage

* It's possible to import the USDMM like bellow:
  ```python
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
  ```
* All input images must be in **./data/x** and output images in **./data/y** folder with names **trainxx.jpg** (where xx is the sequential number (01, 02, 03,....)) for the train dataset, **valxx.jpg** for the validation dataset and **testxx.jpg** for the test dataset.
* At the end of the fit method, inside the folder passe in the inicialization **path_results** will be the saved all the results. Inside the archives "W_**name_save**.txt" and "joint_**name_save**.txt"  will be the window e joint learned.
* It's possible start the learning once again from the latest learned window by sending it pos fixes name (**name_save**) name in the variable new


## Features

* **fit()** -> train the USDMM 

* **results_after_fit** -> open de  "W_**name_save**.txt" and "joint_**name_save**.txt" archives and generate the images

* **test()** -> find error for a fixed window for testing

## Contributing

Author: Mariana Feldman
Maintainer: Mariana Feldman <mariana.feldman@ime.usp.br>

## Reference

 D. Marcondes, M. Feldman, J. Barrera. Discrete Morphological Neural Networks. 2023.
 <https://doi.org/10.48550/arXiv.2310.04584>
=======
# Unrestricted Sequential Discrete Morphological Neural Networks

>>>>>>> 7189eb5b9830ffdc4561bd9d666aba4c522c2aa7
