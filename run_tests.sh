#!/bin/bash

# Nome da imagem
IMAGE_NAME="jax-app"

# Build da imagem Docker
docker build -t $IMAGE_NAME .

# Lista de parâmetros para testar
PARAMS_LIST=(
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 1"
  "--nlayer 3 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 2"
  "--nlayer 2 --wlen 3 --train_size 20 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 20 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 3"
  "--nlayer 2 --wlen 3 --train_size 5 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 4"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 5"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 1 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 6"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 4 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 7"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 12 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 8"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 9"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 3 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 10"
  "--nlayer 2 --wlen 3 --train_size 20 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 11"
  "--nlayer 2 --wlen 3 --train_size 20 --neighbors_sample_f 4 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 12"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 13"
  "--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[1,1,1,1,1,1,1,1,1]' --run 14"
  "--nlayer 1 --wlen 5 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 16 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 15"

)

# Loop através da lista de parâmetros e executa o contêiner para cada conjunto de parâmetros
for PARAMS in "${PARAMS_LIST[@]}"; do
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" $IMAGE_NAME $PARAMS
done


#  chmod +x run_tests.sh

# ./run_tests.sh

