#!/bin/bash

# Nome da imagem
IMAGE_NAME="jax-app"

# Build da imagem Docker
docker build -t $IMAGE_NAME .

# Lista de parâmetros para testar
PARAMS_LIST=(
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 1"
  #"--nlayer 3 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 2"
  #"--nlayer 2 --wlen 3 --train_size 20 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 20 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 3"
  #"--nlayer 2 --wlen 3 --train_size 50 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 4"
  #"--nlayer 2 --wlen 3 --train_size 50 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 50 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 5"
  #"--nlayer 2 --wlen 3 --train_size 5 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 6"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 7"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 1 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 8"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 4 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 9"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 12 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 10"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 11"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 3 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 12"
  #"--nlayer 2 --wlen 3 --train_size 20 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 13"
  #"--nlayer 2 --wlen 3 --train_size 20 --neighbors_sample_f 4 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 14"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 15"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[1,1,1,1,1,1,1,1,1]' --run 16"
  #"--nlayer 1 --wlen 5 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 16 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 17"
  #"--nlayer 2 --wlen 3 --train_size 50 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 18"
  #"--nlayer 2 --wlen 3 --train_size 50 --neighbors_sample_f 12 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 19"
  #"--nlayer 2 --wlen 3 --train_size 50 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 20"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 21" #**#
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 22" 
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 23"
  #"--nlayer 2 --wlen 3 --train_size 50 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 25 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 24"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 7 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 23"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 24"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 12 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 25"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 14 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 26"
  #"--nlayer 3 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 27"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 500 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 28"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 100 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 29"
  #"--nlayer 1 --wlen 5 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 30"
  #"--nlayer 1 --wlen 5 --train_size 30 --neighbors_sample_f 32 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 31"
  #"--nlayer 1 --wlen 5 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 18 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 32"
  #"--nlayer 1 --wlen 5 --train_size 30 --neighbors_sample_f 32 --neighbors_sample_w 18 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 33"
  #"--nlayer 7 --wlen 5 --train_size 1000 --neighbors_sample_f 8 --neighbors_sample_w 16 --epoch_f 20 --epoch_w 3 --batch 100 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 50"
  "--nlayer 7 --wlen 5 --train_size 500 --neighbors_sample_f 12 --neighbors_sample_w 20 --epoch_f 1000 --epoch_w 500 --batch 250 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 53"

)

# Loop através da lista de parâmetros e executa o contêiner para cada conjunto de parâmetros
for PARAMS in "${PARAMS_LIST[@]}"; do
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" $IMAGE_NAME $PARAMS
done


#  chmod +x run_tests.sh

# ./run_tests.sh

