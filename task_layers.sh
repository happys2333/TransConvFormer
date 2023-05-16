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
pred_lens=(192)
features=("M" "S")
python_scripts=("pred_ours.py")
data_sets=("ETTh1" "ETTh2" "WTH")
enc_layers=(3)
n_heads=(8 9 10)
factor=(4 5 6)
d_models=(64 128 256 512 1024)
num_layers=(1 2 3 4 5 6 7 8 9 10)

source activate $env_name

for task in "${num_layers[@]}"
do
  for feature in "${features[@]}"
do
    for data_set in "${data_sets[@]}"
    do
        for pred_len in "${pred_lens[@]}"
        do
            # Loop through Python scripts
            for python_script in "${python_scripts[@]}"
            do
                # Print prompt message with script name and parameters, and write it to output file
                prompt_message=("Processing script '$python_script' with parameters: num_layers=$task, pred_len=$pred_len, features=$feature, dataset=$data_set")

                echo "${prompt_message[@]}"

                # Run current Python script with specified parameters and append output to file
                python $python_script --num_layers $task --pred_len $pred_len --features $feature --dataset $data_set
            done
        done
    done
done
done
