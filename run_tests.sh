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
  #"--nlayer 7 --wlen 5 --train_size 500 --neighbors_sample_f 12 --neighbors_sample_w 20 --epoch_f 1000 --epoch_w 500 --batch 250 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 53" #digito 1
  #"--nlayer 7 --wlen 5 --train_size 300 --neighbors_sample_f 10 --neighbors_sample_w 10 --epoch_f 1000 --epoch_w 500 --batch 300 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 54" #digito 9

  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 100"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 101"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 102"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 103"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 104"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 105"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 106"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 107"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 108"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 109"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 110"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 111"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 112"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 5 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 113"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 114"
  #"--nlayer 2 --wlen 3 --train_size 10 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 10 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 115"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 116"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 117"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 118"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 119"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 120"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 121"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 122"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 8 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 123"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 124"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 125"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 126"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 5 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 127"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 128"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 129"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 130"
  #"--nlayer 2 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 131"
  #"--nlayer 1 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 132"
  #"--nlayer 1 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 133"
  #"--nlayer 1 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 134"
  #"--nlayer 1 --wlen 3 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,0,0,0,1,0,0,0,0]' --run 135"
  #"--nlayer 1 --wlen 5 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 15 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 136"
  #"--nlayer 1 --wlen 5 --train_size 30 --neighbors_sample_f 16 --neighbors_sample_w 9 --epoch_f 100 --epoch_w 50 --batch 30 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 137"



  #"--nlayer 1 --wlen 3 --train_size 300 --neighbors_sample_f 16 --neighbors_sample_w 16 --epoch_f 2000 --epoch_w 200 --batch 50 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 41" # GoL
  #"--nlayer 1 --wlen 3 --train_size 100 --neighbors_sample_f 16 --neighbors_sample_w 16 --epoch_f 2000 --epoch_w 200 --batch 100 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 42" # GoL
  #"--nlayer 1 --wlen 3 --train_size 100 --neighbors_sample_f 16 --neighbors_sample_w 16 --epoch_f 2000 --epoch_w 200 --batch 100 --w_ini '[0,1,0,1,1,1,0,1,0]' --run 43" # GoL
  #"--nlayer 7 --wlen 5 --train_size 500 --neighbors_sample_f 12 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 500 --batch 250 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 55" #digito 1
  #"--nlayer 7 --wlen 5 --train_size 600 --neighbors_sample_f 12 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 500 --batch 200 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 56" #digito 1 - 30% DE DIG 1
  #"--nlayer 7 --wlen 5 --train_size 200 --neighbors_sample_f 12 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 5 --batch 200 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 57" #digito 1 - 30% DE DIG 1
  #"--nlayer 7 --wlen 5 --train_size 200 --neighbors_sample_f 8 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 5 --batch 200 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 58" #digito 1 - 30% DE DIG 1
  #"--nlayer 7 --wlen 5 --train_size 100 --neighbors_sample_f 12 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 5 --batch 100 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 59" #digito 1 - 30% DE DIG 1
  #"--nlayer 7 --wlen 5 --train_size 100 --neighbors_sample_f 8 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 5 --batch 100 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 60" #digito 1 - 30% DE DIG 1
  #"--nlayer 7 --wlen 5 --train_size 100 --neighbors_sample_f 10 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 5 --batch 50 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 61" #digito 1 - 30% DE DIG 1
  #"--nlayer 7 --wlen 5 --train_size 100 --neighbors_sample_f 10 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 100 --batch 50 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 64" #digito 1 - 30% DE DIG 1
  "--nlayer 7 --wlen 5 --train_size 100 --neighbors_sample_f 10 --neighbors_sample_w 20 --epoch_f 500 --epoch_w 100 --batch 50 --w_ini '[0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0]' --run 65" #digito 1 - 30% DE DIG 1
)

# Loop através da lista de parâmetros e executa o contêiner para cada conjunto de parâmetros
for PARAMS in "${PARAMS_LIST[@]}"; do
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G -v "$(pwd)/output:/app/output" $IMAGE_NAME $PARAMS
done


#  chmod +x run_tests.sh

# ./run_tests.sh


#df -h
#docker system prune -a --volumes


