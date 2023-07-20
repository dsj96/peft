model=mt0-xxl
config_file=./ds_zero3_config/default_config.yaml
dataset_name=./data/opus-100/opus-100-corpus/v1.0/opus_train_dev_tst/opus9_select_train_labse_dev_tst_add_task
raw_dataset_name=opus9_select_train_labse_dev_tst_add_task
r=8
lora_alpha=32
lora_dropout=0.1
text_column=SRC
label_column=TGT
num_epochs=3
batch_size=8
src_max_length=256
max_new_tokens=256
num_beams=5
save_interval=1000
validation_interval=2000
validation_ratio=0.1
model_name_or_path=./pre_trained_model/$model
output_dir=./trained_checkpoint/2w-model-$model-raw_dataset_name-$raw_dataset_name-num_epochs-$num_epochs-batch_size-$batch_size-src_max_length-$src_max_length-max_new_tokens-$max_new_tokens-num_beams-$num_beams-save_interval-$save_interval-validation_interval-$validation_interval-validation_ratio-$validation_ratio
echo $output_dir
accelerate launch --config_file=$config_file examples/mt0_peft_lora_ds_zero3_offload.py --model_name_or_path=$model_name_or_path --dataset_name=$dataset_name\
                    --r=$r --lora_alpha=$lora_alpha --lora_dropout=$lora_dropout --text_column=$text_column --label_column=$label_column \
                    --num_epochs=$num_epochs --batch_size=$batch_size\
                    --save_interval=$save_interval --validation_interval=$validation_interval --validation_ratio=$validation_ratio\
                    --src_max_length=$src_max_length --output_dir=$output_dir --max_new_tokens=$max_new_tokens --num_beams=$num_beams \
                    --monitor_config=./ds_zero3_config/monitor_config.json\
                    --deepspeed_config_file=./ds_zero3_config/ds_offload_config.json
