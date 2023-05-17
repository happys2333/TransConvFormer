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
seq_len=96
label_len=48
pred_lens=(96 192 336 720)
features=("M" "S")
python_scripts=("pred_ours.py")
data_sets=("ETTm2" "WTH" "ECL")
enc_layers=(2)

source activate $env_name

for enc_layer in "${enc_layers[@]}"
do
  # Loop through pred_lens and features
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
                prompt_message=("Processing script '$python_script' with parameters: seq_len=$seq_len, label_len=$label_len, pred_len=$pred_len, features=$feature, dataset=$data_set, enc_layers=$enc_layer")

                echo "${prompt_message[@]}"

                # Run current Python script with specified parameters and append output to file
                python $python_script --seq_len $seq_len --label_len $label_len --pred_len $pred_len --features $feature --dataset $data_set --enc_layers $enc_layer
            done
        done
    done
done

done
