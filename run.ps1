
# Define variables
$env_name = "DL-Project-2022"
$seq_len = 96
$label_len = 48
$pred_lens = @(96, 192, 336, 720)
$features = @('M', 'S')
$python_scripts = @('pred_informer.py', 'pred_ours.py')
$data_sets = @('ETTh1','ETTh2','ETTm1','ETTm2','WTH','ECL')

cd F:\TransConvFormer
conda activate $env_name
# Loop through pred_lens and features
foreach($data_set in $data_sets){
    foreach ($pred_len in $pred_lens) {
        foreach ($feature in $features) {
            # Loop through Python scripts
            foreach ($python_script in $python_scripts) {
                # Print prompt message with script name and parameters, and write it to output file
                $prompt_message = "Processing script '$python_script' with parameters: seq_len=$seq_len, label_len=$label_len, pred_len=$pred_len, features=$feature, dataset=$data_set"

                $prompt_message | Tee-Object -FilePath .\output.txt -Append

                # Run current Python script with specified parameters and append output to file
                python $python_script --seq_len $seq_len --label_len $label_len --pred_len $pred_len --features $feature --dataset $data_set >> .\output.txt
            }
        }
    }
}
