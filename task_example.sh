#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH --partition=gpulab02
#SBATCH -J pytorch_transConv
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --qos=gpulab02

# Define variables
env_name="trans"
seq_lens=(96)
label_lens=(48)
pred_lens=(96 192 336 720)
features=("M" "S")
python_scripts=("pred_ours.py")
data_sets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "WTH" "ECL")
enc_layers=(3)