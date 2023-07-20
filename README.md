## Repository
Forked from [PEFT](https://github.com/huggingface/peft). Thanks for their outstanding work.
# Reproduce the Bug
## Step 1: Prepare Model and Dataset
### Model
```
git clone https://github.com/dsj96/peft.git
cd pre_trained_model
git clone https://huggingface.co/bigscience/mt0-xxl
```
### Dataset
In order to reproduce the bug, I have released the dataset used in the formal training in `data` file.

`opus_train_labse_dev_tst_add_task.zip` Always suffer from OOM errors at the **same** steps.

`wmt16enro_dev_dev_test_task.zip` could run correctly (train validation test).

## Step 2: Enverment
### Hardware
Tesla V100(32G) * `4 or 8`.

### python library
More details refer to [pip_list.txt](https://github.com/dsj96/peft/blob/main/pip_list.txt)

### ds_report
```
ds_report
[2023-07-20 04:58:58,855] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /opt/conda/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
/opt/conda/lib/python3.8/site-packages/bitsandbytes/cextension.py:33: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
CUDA SETUP: Loading binary /opt/conda/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so...
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
async_io ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
fused_adam ............. [NO] ....... [OKAY]
fused_lamb ............. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
random_ltd ............. [NO] ....... [OKAY]
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
--------------------------------------------------
No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
DeepSpeed general environment info:
torch install path ............... ['/opt/conda/lib/python3.8/site-packages/torch']
torch version .................... 2.0.1+cu117
deepspeed install path ........... ['/opt/conda/lib/python3.8/site-packages/deepspeed']
deepspeed info ................... 0.10.0, unknown, unknown
torch cuda version ............... 11.7
torch hip version ................ None
nvcc version ..................... 11.3
deepspeed wheel compiled w. ...... torch 2.0, cuda 11.7
```

## Main modified python script
[examples/mt0_peft_lora_ds_zero3_offload.py](https://github.com/dsj96/peft/blob/main/examples/mt0_peft_lora_ds_zero3_offload.py) modified according [examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py](https://github.com/dsj96/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py)

## Step 3: Run commend
```
bash mt0.sh
```
## Note:
To save GPU memory, I have tried some methods (I can't guarantee if there is a conflict between them):
```
1. deepspeed zero-stage 3,2,1 + off load
2. bf16
3. os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
4. model.enable_input_require_grads()
   model.gradient_checkpointing_enable() # dsj
5. gc.collect() torch.cuda.empty_cache() # import gc
   get_accelerator().empty_cache()
   model.empty_partition_cache()
6. outputs = model(**batch, use_cache=False)
7. reduce:
   batch size
   stage3_prefetch_bucket_size
   stage3_param_persistence_threshold
   stage3_max_live_parameters
   stage3_max_reuse_distance
8. increase: gradient_accumulation_steps
```
