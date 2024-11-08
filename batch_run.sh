#!/bin/sh
source /mnt/petrelfs/zhangdi1/miniforge3/bin/activate /mnt/petrelfs/zhangdi1/miniforge3/envs/torch


# export HF_ENDPOINT=https://hf-mirror.com

export all_proxy="http://zhangdi1:vhLylrana9yeoFdiR4SOKaFTOh5t5FL9bBhHd5Bx1MCU0sbGgyNdEKC3Bq64@10.1.20.50:23128"

cd /mnt/hwfile/ai4chem/math/LLaMA-O1/

python offline_collect.py