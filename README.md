
# USDMNN - Unrestricted Sequential Discrete Morphological Neural Networks


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

- Docker
- ROCm-enabled GPU and drivers

## Usage

### Docker Setup

This repository includes a `Dockerfile` that sets up the environment necessary to run your experiments. You can choose which experiment to run by uncommenting the appropriate `ENTRYPOINT` in the `Dockerfile`.

#### Available ENTRYPOINT Options:

- `main.py`: Runs a single experiment. Make sure to configure the `base_config` inside this file to execute the experiment correctly.
- `test_digits.py`: Runs experiments for edge recognition of noisy digits.
- `test_gol.py`: Trains the transition function for the Game of Life.
- `test_mnist.py`: Runs experiments for digit classification using the MNIST dataset.

#### Running Experiments:

After configuring the ENTRYPOINT, you can further customize your experiments by editing the **PARAM_LIST** in the *run_tests.sh* script.
Content of the **PARAM_LIST**:
* nlayer: INT - Number of operators in each layer
* wlen: INT - Size of the operator (wlen*wlen)
* train_size: INT - Number of images to train
* val_size: INT - Number of images to validate
* neighbors_sample_f: INT - Number of neighbors to sort in the boolean function lattice
* neighbors_sample_w: INT - Number of neighbors to sort in the windows lattice
* epoch_f: INT - Number of epochs for the boolean function lattice (fixed windows)
* epoch_w: INT - Number of epochs for the windows lattice
* es_f: INT - Max number of epochs without changes in the boolean function lattice
* es_w: INT - Max number of epochs without changes in the windows lattice
* batch: INT - Batch size
* w_ini: STR - Initial operator
* run: INT - Run number for saving results

After configuring correctly all the **PARAM_LIST** you should:

1. Make the script executable:

   ```bash
   chmod +x run_tests.sh
   ```

2. Run the experiments:

   ```bash
   ./run_tests.sh
   ```


## Features

* **Modular Experiment Setup:** Easily switch between different experiments by changing the ENTRYPOINT in the Dockerfile.
* **Parameter Customization:** Fine-tune your experiments using the run_tests.sh script.
* **Output Management:** All outputs are stored in the /app/output directory within the container.

## Examples

### Running an MNIST Classification Experiment

1. Uncomment the following line in the `Dockerfile`:

   ```Dockerfile
   ENTRYPOINT ["python3", "/app/test_mnist.py"]
   ```

2. Configure your **PARAM_LIST**
  
    ```Python
    PARAMS_LIST=(
      --nlayer 7 --wlen 5 --train_size 100 --val_size 100 --neighbors_sample_f 10 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 30 --es_f 100 --es_w 10 --batch 50 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 1
      --nlayer 7 --wlen 5 --train_size 100 --val_size 500 --neighbors_sample_f 10 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 30 --es_f 100 --es_w 10 --batch 50 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 2
    )
    ```

3. Build and run the Docker container:
   ```bash
   chmod +x run_tests.sh
   ./run_tests.sh
   ```

## Contributing

Author: Mariana Feldman

Maintainer: Mariana Feldman <mariana.feldman@ime.usp.br>

## Reference

 D. Marcondes, M. Feldman, J. Barrera. Discrete Morphological Neural Networks. 2023.
 <https://doi.org/10.48550/arXiv.2310.04584>
