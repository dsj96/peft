## Repository
Forked from [PEFT](https://github.com/huggingface/peft). Thanks for their outstanding work.

## Prepare Model and Dataset
### Model
```
git clone https://huggingface.co/bigscience/mt0-xxl
```
### Dataset
In order to reproduce the bug, I have released the dataset used in the formal training in `data` file.


## Main modified python script
[examples/mt0_peft_lora_ds_zero3_offload.py](https://github.com/dsj96/peft/blob/main/examples/mt0_peft_lora_ds_zero3_offload.py) modified according [examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py](https://github.com/dsj96/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py)

## Run commend
```
bash mt0.sh
```
