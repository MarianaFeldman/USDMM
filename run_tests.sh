#!/bin/bash

# Nome da imagem
IMAGE_NAME="jax-app"

# Build da imagem Docker
docker build -t $IMAGE_NAME .

# Lista de parâmetros para testar
PARAMS_LIST=(

  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 100
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 101
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 102
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 103
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 104
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 105
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 106
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 107
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 108
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 109
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 110
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 111
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 112
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 113
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 114
  --nlayer 2 --wlen 3 --train_size 10 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 115
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 116
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 117
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 118
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 119
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 120
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 121
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 122
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 123
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 124
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 125
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 126
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 127
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 128
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 129
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 130
  --nlayer 2 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 131
  --nlayer 1 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 132
  --nlayer 1 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 133
  --nlayer 1 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 134
  --nlayer 1 --wlen 3 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 135
  --nlayer 1 --wlen 5 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 15 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 136
  --nlayer 1 --wlen 5 --train_size 30 --val_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --es_f 30 --es_w 20 --batch 30 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 137

)


for PARAMS in "${PARAMS_LIST[@]}"; do
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" $IMAGE_NAME $PARAMS
done


#  chmod +x run_tests.sh

# ./run_tests.sh



