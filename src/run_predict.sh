#!/bin/bash

# check if config file given as the argument
if [[ $1 == "" ]]; then
    echo "ERROR: no configuration file supplied"
    echo "USAGE: ./run_predict.sh ../config_files/[CONFIG_FILE]"
    exit 1
fi

# include the config file variables
source $1
log_dest="none"

# execute the python training script with the directory passed as input and the gpu instance
# -u means outputs streams are unbuffered, tee sends output streams to a file as well as console,
# -i ensures that ctrl-c will not break the pipe since we want to save the model on ctrl-c and continue logging
python3 -u predict.py $session_name $log_dest $root_dir --train_data_dir $train_data_dir --test_data_dir $test_data_dir $train_labels_csv \
                           $test_labels_csv $k_folds $gpu_i $model_name $optim $loss_freq $val_freq \
                           $batch_size $lr $classification $num_classes $regression $num_epochs \
                           $normalize $weighted_loss $imbalanced_sampler $l2_reg $weight_decay_amnt \
                           $dropout $dropout_prob --load_trained $load_trained --trained_path $trained_path | tee -i $fileName