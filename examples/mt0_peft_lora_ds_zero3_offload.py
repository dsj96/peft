import gc
import os
import time
import sys
import threading
import random
import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.accelerator import get_accelerator # dsj
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig

import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, GenerationConfig, get_scheduler, Adafactor
import argparse
from datasets import load_dataset, load_metric, load_from_disk, Dataset, DatasetDict

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from accelerate.utils import DummyOptim, DummyScheduler # opt
import pdb # pdb.set_trace()
import evaluate
from accelerate.logging import get_logger
import json

torch.backends.cudnn.benchmark=True # dsj

logger = get_logger(__name__)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--model_name_or_path', type=str, default="./pre_trained_model/mt0-xxl", help='')

    parser.add_argument('--dataset_name', type=str, default="./data/wmt14/hi-en/hugginface_wmt14", help='')

    parser.add_argument('--inference_mode',  action="store_true", help='k.')

    parser.add_argument('--r',  type=int, default=8, help='')

    parser.add_argument('--lora_alpha',  type=int, default=32, help='')

    parser.add_argument('--lora_dropout',  type=float, default=0.1, help='')

    parser.add_argument('--text_column',  type=str, default='SRC', help='')

    parser.add_argument('--label_column',  type=str, default='TGT', help='')

    parser.add_argument('--num_epochs',  type=int, default=3, help='')

    parser.add_argument('--batch_size',  type=int, default=16, help='')
    
    parser.add_argument('--do_validation',  action="store_true", help='k.')

    parser.add_argument('--do_test',  action="store_true", help='k.')

    parser.add_argument('--src_max_length', type=int, default=192, help='')

    parser.add_argument('--output_dir', type=str, default=None, help='')

    parser.add_argument('--push_to_hub', action="store_true", help='k.')
    
    parser.add_argument('--cache_dir', type=str, default='../huggface_cache', help='')
    
    parser.add_argument('--num_proc',  type=int, default=8, help='')
    
    parser.add_argument('--max_new_tokens',  type=int, default=192, help='')
    
    parser.add_argument('--num_beams',  type=int, default=1, help='')
    
    parser.add_argument('--no_repeat_ngram_size',  type=int, default=3, help='')
    
    parser.add_argument('--save_interval',  type=int, default=5000, help='')
    
    parser.add_argument('--validation_interval',  type=int, default=10, help='')
    
    parser.add_argument('--validation_ratio',  type=float, default=0.1, help='')

    parser.add_argument('--lr_scheduler_type', type=str, default='linear', \
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_on_plateau"], \
                        help='')
    
    parser.add_argument('--num_warmup_steps', type=int, default=1000, help='')
    
    parser.add_argument('--deepspeed_config_file', type=str, default='', help='')
    
    parser.add_argument('--lr', type=float, default=2e-5, help='')
    
    parser.add_argument('--monitor_config', type=str, default='', help='')
    
    
    
    args, _ = parser.parse_known_args()
    # args.cuda = not args.cuda and torch.cuda.is_available()
    return args

def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

def setup_seed(seed=42):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     set_seed(seed)

def reduce_validation_data_size(dataset, args):
    tmp_list = []
    selected_num = int(len(dataset['validation']) * args.validation_ratio)
    for item in dataset['validation']:
        tmp_list.append(item)
    selected_validation_data = random.sample(tmp_list, selected_num)
    
    src_list, tgt_list = [], []
    for item in selected_validation_data:
        src_list.append(item[args.text_column])
        tgt_list.append(item[args.label_column])

    tmp_datasets = Dataset.from_dict({
        'SRC': src_list,
        'TGT': tgt_list
    })
    
    train_dev_tst_dataset = {
        'train': dataset["train"],
        'validation': tmp_datasets,
        'test': dataset["test"]
    }
    train_dev_tst_dataset  = DatasetDict(train_dev_tst_dataset)
    return train_dev_tst_dataset
    
def main():
    args = get_args()
    accelerator = Accelerator()
    # model_name_or_path = "bigscience/T0_3B"
    model_name_or_path = args.model_name_or_path # "./pre_trained_model/mt0-xxl"
    dataset_name = args.dataset_name # "twitter_complaints"
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=args.inference_mode, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )
    text_column = args.text_column
    label_column = args.label_column
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed = args.seed
    do_validation = args.do_validation
    do_test = args.do_test
    src_max_length = args.src_max_length
    num_proc = args.num_proc
    save_interval =args.save_interval
    set_seed(seed)
    setup_seed(seed)
    print('args:\n', args)
    
    dataset = load_from_disk(args.dataset_name)
    
    # reduce the eval dataset size
    dataset = reduce_validation_data_size(dataset, args)

    generation_config = {
        "max_new_tokens" : args.max_new_tokens,
        "num_beams" : args.num_beams,
        "no_repeat_ngram_size" : args.no_repeat_ngram_size
    }
    generation_config = GenerationConfig.from_dict(generation_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir = args.cache_dir)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column] # ['no complaint', 'no complaint', 'complaint', 'complaint',
        model_inputs = tokenizer(inputs, truncation=True)
        labels = tokenizer(
            targets, max_length=src_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = labels["input_ids"] # labels.input_ids.shape=torch.Size([50, 4])
        labels[labels == tokenizer.pad_token_id] = -100 # tokenizer.pad_token_id=0 <pad> 
        model_inputs["labels"] = labels # KeyError: 'labels'
        return model_inputs
    print(dataset)
    
    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] # evaluation
    test_dataset = processed_datasets["test"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=int(batch_size/2), pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    # creating model
    # max_memory = {0: "28GIB", 0: "28GIB", "cpu": "180GB"}
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # model = prepare_model_for_int8_training(model)
    # load_dict
    # bin_list = os.listdir(model_name_or_path)
    # for bin_item in bin_list:
    #     if bin_item.endswith(".bin"):
    #         model.load_state_dict(torch.load(model_name_or_path + '/' + bin_item))
    if accelerator.state.deepspeed_plugin is not None:
        if "compression_training" in accelerator.state.deepspeed_plugin.deepspeed_config and \
               (accelerator.state.deepspeed_plugin.deepspeed_config['compression_training']['weight_quantization']['shared_parameters']['enabled']==True or \
               accelerator.state.deepspeed_plugin.deepspeed_config['compression_training']['activation_quantization']['shared_parameters']['enabled']==True):
            model = prepare_model_for_int8_training(model)
            model = init_compression(model, args.deepspeed_config_file)

        if "quantize_training" in accelerator.state.deepspeed_plugin.deepspeed_config and \
               accelerator.state.deepspeed_plugin.deepspeed_config['quantize_training']['enabled']==True:
            model = init_compression(model, args.deepspeed_config_file)

    model = get_peft_model(model, peft_config) # check dtype: model.base_model.model.encoder.block[0].layer[0].SelfAttention.q.weight.dtype
    
    model.print_trainable_parameters()
    model.enable_input_require_grads()    # dsj
    model.gradient_checkpointing_enable() # dsj
        
    # optimizer
    optimizer_cls = (
        Adafactor
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters())


    # lr scheduler
    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=(len(train_dataloader) * num_epochs),
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=(len(train_dataloader) * num_epochs)
        )

    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=(len(train_dataloader) * num_epochs),
    # )

    monitor_config = DeepSpeedConfig(args.monitor_config)

    print(f"{accelerator.state.deepspeed_plugin=}")
    print(f"{accelerator.state.deepspeed_plugin.deepspeed_config=}")
    
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler, monitor_config = accelerator.prepare(
        model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler, monitor_config
    )

    monitor = MonitorMaster(monitor_config.monitor_config)
    # accelerator.print(model)
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
    cur_step = 0

    
    from bleu.bleu import Bleu
    bleu_metric = Bleu()
    sacrebleu_metric = evaluate.load("./examples/evaluate_sacrebleu.py")
    # bleu_metric = evaluate.load("./examples/bleu")
    chrf_metric = evaluate.load("./examples/evaluate_chrf.py")
    google_bleu_metric = evaluate.load("./examples/google_bleu")


    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            accelerator.print('train epoch{}'.format(epoch))
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                cur_step = cur_step + 1
                try:
                    outputs = model(**batch, use_cache=False) # dsj
                    # outputs = model(**batch) # dsj
                    loss = outputs.loss
                    # loss.requires_grad=True # dsj
                    total_loss += loss.detach().float()

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    events = [("LPB", float(loss), model.global_samples)]
                    monitor.write_events(events)
                    events = [("lr",  lr_scheduler.get_lr()[0], model.global_samples)]
                    monitor.write_events(events)

                    if cur_step % save_interval == 0:
                        accelerator.print(f"{epoch=}", '\t', f"{step=}", '\t', f"{loss=}")
                        if args.output_dir is not None:
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                args.output_dir+'_step_' + str(cur_step), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                            )

                    if do_validation and cur_step % args.validation_interval==0:
                        model.eval()
                        total_validation_loss = 0
                        eval_preds = []
                        with TorchTracemalloc() as tracemalloc:
                            for _, batch in enumerate(tqdm(eval_dataloader)):
                                # print(batch) # {input_ids:tensor, attention_mask:tensor, labels:tensor}
                                try:
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                except:
                                    print("except torch.cuda.empty_cache()")
                                    pass
                                try:
                                    get_accelerator().empty_cache()
                                except:
                                    print("except get_accelerator().empty_cache()")
                                    pass
                                total_validation_loss += float(model(**batch, use_cache=False).loss)

    #                             batch = {k: v for k, v in batch.items() if k != "labels"}
    #                             with torch.no_grad():
    #                                 outputs = accelerator.unwrap_model(model).generate(
    #                                     **batch, synced_gpus=is_ds_zero_3, generation_config=generation_config
    #                                 )  # synced_gpus=True for DS-stage 3
    #                             # print(outputs) # tensor([[     0,   1250,    590, 199021,  88482,      1,      0,      0, len=20

    #                             outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
    #                             # print(outputs) # tensor = [     0,    430,  11236,      1,      0,      0
    #                             # pdb.set_trace()
    #                             preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy() 
    #                             # print(preds) # list = [[     0   1250    590 199021  88482      1      0
    #                             # pdb.set_trace()
    #                             eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
    #                             # print(eval_preds) ['@HMRCcustomers', '@KristaMariePark', '@BoseService', 'Poor network', '#BrothersAtHome',
    #                             # pdb.set_trace()

    #                     # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    #                     accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
    #                     accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
    #                     accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
    #                     accelerator.print(
    #                         "GPU Total Peak Memory consumed during the eval (max): {}".format(
    #                             tracemalloc.peaked + b2mb(tracemalloc.begin)
    #                         )
    #                     )

    #                     accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
    #                     accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
    #                     accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
    #                     accelerator.print(
    #                         "CPU Total Peak Memory consumed during the eval (max): {}".format(
    #                             tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
    #                         )
    #                     )
    #                     # modify
    #                     assert len(eval_preds) == len(
    #                         dataset["validation"][label_column]
    #                     ), f"{len(eval_preds)} != {len(dataset['validation'][label_column])}"

                        total_validation_loss = total_validation_loss / len(eval_dataloader)

    #                     sacrebleu_eval_result = sacrebleu_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])
    #                     bleu_eval_result  = bleu_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])
    #                     chrf_eval_result  = chrf_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])
    #                     chrf1_eval_result = chrf_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column], word_order=1)
    #                     chrf2_eval_result = chrf_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column], word_order=2)
    #                     google_bleu_eval_result = google_bleu_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])

    #                     logger.info({"sacrebleu": sacrebleu_eval_result["score"]})
    #                     logger.info({"bleu": bleu_eval_result["bleu"]})
    #                     logger.info({"chrf": chrf_eval_result["score"]})
    #                     logger.info({"chrf+": chrf1_eval_result["score"]})
    #                     logger.info({"chrf++": chrf2_eval_result["score"]})
    #                     logger.info({"google_bleu": google_bleu_eval_result["google_bleu"]})
    #                     logger.info({"total_validation_loss": total_validation_loss})

    #                     accelerator.print(f"{sacrebleu_eval_result=}")
    #                     accelerator.print(f"{bleu_eval_result=}")
    #                     accelerator.print(f"{chrf_eval_result=}")
    #                     accelerator.print(f"{chrf1_eval_result=}")
    #                     accelerator.print(f"{chrf2_eval_result=}")
    #                     accelerator.print(f"{google_bleu_eval_result=}")
    #                     accelerator.print(f"{total_validation_loss=}")
    #                     # eval_result={'score': 4.891362790930584, 'counts': [23, 9, 3, 0], 'totals': [47, 43, 39, 35], 'precisions': [48.93617021276596, 20.930232558139537, 7.6923076923076925, 1.4285714285714286], 'bp': 0.4748858350665527, 'sys_len': 47, 'ref_len': 82}
    #                     accelerator.print(f"{eval_preds[:5]=}")
    #                     # eval_preds[:5]=['Neviens nemainīsies, ja nemainīsies pats, jums ir jādodas un', 'Pekka takes the horse to the racecourse and decides where it will compete.', "« Je pense que je vais appeler des amis pour qu'ils puissent rire ", "Il a ajouté qu'à l'époque, il avait «"]
    #                     accelerator.print(f"{dataset['validation'][label_column][:5]=}")

    #                     # save eval_predicts
    #                     eval_df = dataset["validation"].to_pandas()
    #                     eval_df["predict_txt"] = eval_preds
    #                     os.makedirs(args.output_dir, exist_ok=True)
    #                     eval_df.to_csv(args.output_dir+"/eval_step_{}_predictions.csv".format(cur_step), index=False)

    #                     events = [("validation_sacrebleu",   sacrebleu_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_bleu",        bleu_eval_result["bleu"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_chrf",        chrf_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_chrf+",       chrf1_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_chrf++",      chrf2_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_google_bleu", google_bleu_eval_result["google_bleu"], model.global_samples)]
    #                     monitor.write_events(events)
                        events = [("validation_loss",        total_validation_loss, model.global_samples)]
                        monitor.write_events(events)
                        model.train()
                except:
                    accelerator.print("except")
                    gc.collect()
                    torch.cuda.empty_cache()
                    get_accelerator().empty_cache()
                    # model.empty_partition_cache()
                    outputs = model(**batch, use_cache=False) # dsj
                    # outputs = model(**batch) # dsj
                    loss = outputs.loss
                    # loss.requires_grad=True # dsj
                    total_loss += loss.detach().float()

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    events = [("LPB", float(loss), model.global_samples)]
                    monitor.write_events(events)
                    events = [("lr",  lr_scheduler.get_lr()[0], model.global_samples)]
                    monitor.write_events(events)

                    if cur_step % save_interval == 0:
                        accelerator.print(f"{epoch=}", '\t', f"{step=}", '\t', f"{loss=}")
                        if args.output_dir is not None:
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                args.output_dir+'_step_' + str(cur_step), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                            )

                    if do_validation and cur_step % args.validation_interval==0:
                        model.eval()
                        total_validation_loss = 0
                        eval_preds = []
                        with TorchTracemalloc() as tracemalloc:
                            for _, batch in enumerate(tqdm(eval_dataloader)):
                                # print(batch) # {input_ids:tensor, attention_mask:tensor, labels:tensor}
                                try:
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                except:
                                    print("except torch.cuda.empty_cache()")
                                    pass
                                try:
                                    get_accelerator().empty_cache()
                                except:
                                    print("except get_accelerator().empty_cache()")
                                    pass
                                total_validation_loss += float(model(**batch, use_cache=False).loss)

    #                             batch = {k: v for k, v in batch.items() if k != "labels"}
    #                             with torch.no_grad():
    #                                 outputs = accelerator.unwrap_model(model).generate(
    #                                     **batch, synced_gpus=is_ds_zero_3, generation_config=generation_config
    #                                 )  # synced_gpus=True for DS-stage 3
    #                             # print(outputs) # tensor([[     0,   1250,    590, 199021,  88482,      1,      0,      0, len=20

    #                             outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
    #                             # print(outputs) # tensor = [     0,    430,  11236,      1,      0,      0
    #                             # pdb.set_trace()
    #                             preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy() 
    #                             # print(preds) # list = [[     0   1250    590 199021  88482      1      0
    #                             # pdb.set_trace()
    #                             eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
    #                             # print(eval_preds) ['@HMRCcustomers', '@KristaMariePark', '@BoseService', 'Poor network', '#BrothersAtHome',
    #                             # pdb.set_trace()

    #                     # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    #                     accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
    #                     accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
    #                     accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
    #                     accelerator.print(
    #                         "GPU Total Peak Memory consumed during the eval (max): {}".format(
    #                             tracemalloc.peaked + b2mb(tracemalloc.begin)
    #                         )
    #                     )

    #                     accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
    #                     accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
    #                     accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
    #                     accelerator.print(
    #                         "CPU Total Peak Memory consumed during the eval (max): {}".format(
    #                             tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
    #                         )
    #                     )
    #                     # modify
    #                     assert len(eval_preds) == len(
    #                         dataset["validation"][label_column]
    #                     ), f"{len(eval_preds)} != {len(dataset['validation'][label_column])}"

                        total_validation_loss = total_validation_loss / len(eval_dataloader)

    #                     sacrebleu_eval_result = sacrebleu_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])
    #                     bleu_eval_result  = bleu_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])
    #                     chrf_eval_result  = chrf_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])
    #                     chrf1_eval_result = chrf_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column], word_order=1)
    #                     chrf2_eval_result = chrf_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column], word_order=2)
    #                     google_bleu_eval_result = google_bleu_metric.compute(predictions=eval_preds, references=dataset['validation'][label_column])

    #                     logger.info({"sacrebleu": sacrebleu_eval_result["score"]})
    #                     logger.info({"bleu": bleu_eval_result["bleu"]})
    #                     logger.info({"chrf": chrf_eval_result["score"]})
    #                     logger.info({"chrf+": chrf1_eval_result["score"]})
    #                     logger.info({"chrf++": chrf2_eval_result["score"]})
    #                     logger.info({"google_bleu": google_bleu_eval_result["google_bleu"]})
    #                     logger.info({"total_validation_loss": total_validation_loss})

    #                     accelerator.print(f"{sacrebleu_eval_result=}")
    #                     accelerator.print(f"{bleu_eval_result=}")
    #                     accelerator.print(f"{chrf_eval_result=}")
    #                     accelerator.print(f"{chrf1_eval_result=}")
    #                     accelerator.print(f"{chrf2_eval_result=}")
    #                     accelerator.print(f"{google_bleu_eval_result=}")
    #                     accelerator.print(f"{total_validation_loss=}")
    #                     # eval_result={'score': 4.891362790930584, 'counts': [23, 9, 3, 0], 'totals': [47, 43, 39, 35], 'precisions': [48.93617021276596, 20.930232558139537, 7.6923076923076925, 1.4285714285714286], 'bp': 0.4748858350665527, 'sys_len': 47, 'ref_len': 82}
    #                     accelerator.print(f"{eval_preds[:5]=}")
    #                     # eval_preds[:5]=['Neviens nemainīsies, ja nemainīsies pats, jums ir jādodas un', 'Pekka takes the horse to the racecourse and decides where it will compete.', "« Je pense que je vais appeler des amis pour qu'ils puissent rire ", "Il a ajouté qu'à l'époque, il avait «"]
    #                     accelerator.print(f"{dataset['validation'][label_column][:5]=}")

    #                     # save eval_predicts
    #                     eval_df = dataset["validation"].to_pandas()
    #                     eval_df["predict_txt"] = eval_preds
    #                     os.makedirs(args.output_dir, exist_ok=True)
    #                     eval_df.to_csv(args.output_dir+"/eval_step_{}_predictions.csv".format(cur_step), index=False)

    #                     events = [("validation_sacrebleu",   sacrebleu_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_bleu",        bleu_eval_result["bleu"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_chrf",        chrf_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_chrf+",       chrf1_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_chrf++",      chrf2_eval_result["score"], model.global_samples)]
    #                     monitor.write_events(events)
    #                     events = [("validation_google_bleu", google_bleu_eval_result["google_bleu"], model.global_samples)]
    #                     monitor.write_events(events)
                        events = [("validation_loss",        total_validation_loss, model.global_samples)]
                        monitor.write_events(events)
                        model.train()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss) # torch.exp(torch.tensor(1.0))=tensor(2.7183)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}") # epoch=0: train_ppl=tensor(1.0000, device='cuda:0') train_epoch_loss=tensor(1.8981e-06, device='cuda:0')



    if do_test:
        accelerator.print("testing")
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, generation_config=generation_config
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy()
            # preds = accelerator.gather(outputs).detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        test_df = dataset["test"].to_pandas()
        print("len(test_preds): ", len(test_preds), "len(test_df):", len(test_df)) # len(test_preds):  100 len(test_df): 100
        assert len(test_preds) == len(test_df), f"{len(test_preds)} != {len(test_df)}" # 67328 != 67318

        sacrebleu_test_result = sacrebleu_metric.compute(predictions=test_preds, references=dataset['test'][label_column])
        bleu_test_result  = bleu_metric.compute(predictions=test_preds, references=dataset['test'][label_column])
        chrf_test_result  = chrf_metric.compute(predictions=test_preds, references=dataset['test'][label_column])
        chrf1_test_result = chrf_metric.compute(predictions=test_preds, references=dataset['test'][label_column], word_order=1)
        chrf2_test_result = chrf_metric.compute(predictions=test_preds, references=dataset['test'][label_column], word_order=2)
        google_bleu_test_result  = google_bleu_metric.compute(predictions=test_preds, references=dataset['test'][label_column])
        
        logger.info({"sacrebleu": sacrebleu_test_result["score"]})
        logger.info({"bleu": bleu_test_result["bleu"]})
        logger.info({"chrf": chrf_test_result["score"]})
        logger.info({"chrf+": chrf1_test_result["score"]})
        logger.info({"chrf++": chrf2_test_result["score"]})
        logger.info({"google_bleu": google_bleu_test_result["google_bleu"]})
        
        accelerator.print(f"{sacrebleu_test_result=}")
        accelerator.print(f"{bleu_test_result=}")
        accelerator.print(f"{chrf_test_result=}")
        accelerator.print(f"{chrf1_test_result=}")
        accelerator.print(f"{chrf2_test_result=}")
        accelerator.print(f"{google_bleu_test_result=}")

        accelerator.print(f"{test_preds[:5]=}")
        accelerator.print(f"{dataset['test'][label_column][:5]=}")

    if args.output_dir is not None:
        if do_test:
            test_df["predict_txt"] = test_preds
            # pred_df = test_df[[label_column, "predict_txt"]]
            # pred_df.columns = ["true_txt", "predict_txt"]
            os.makedirs(args.output_dir, exist_ok=True)
            test_df.to_csv(args.output_dir+"/test_predictions.csv", index=False)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
        if do_test:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({
                    "test_sacrebleu": sacrebleu_test_result["score"],
                    "test_bleu": bleu_test_result["bleu"],
                    "test_chrf": chrf_test_result["score"],
                    "test_chrf+": chrf1_test_result["score"],
                    "test_chrf++": chrf2_test_result["score"],
                    "test_google_bleu": google_bleu_test_result["google_bleu"],              
                }, f)

    # accelerator.wait_for_everyone()
    # model.push_to_hub(
    #     "smangrul/"
    #     + f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
    #     state_dict=accelerator.get_state_dict(model),
    #     use_auth_token=True,
    # )
    # accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
