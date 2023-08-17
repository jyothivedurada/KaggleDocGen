from __future__ import absolute_import, division, print_function
from matplotlib.animation import adjusted_figsize
import tqdm
import pandas as pd
import os
import argparse
import glob
import logging
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import sys

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model

LIST_OF_STRUCTURES = tuple(["def", "for ", "while", "class", "with"])

cpu_cont = 16
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

import combine
from CodeBERT import document_prediction_using_codebert
from UnixCoder import document_prediction_using_unixcoder
from GraphCodeBERT import graphcodebert_predict
from CodeT5 import codet5_predict
from PLBART import plbart_predict

import time

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

parser = argparse.ArgumentParser()
args = parser.parse_args()
model, tokenizer, pool, config = "", "", "", ""

def get_example(item):
    prefix_code, next_codeline, label, tokenizer, args, cache = item
    prefix_code = " ".join(("\n".join(prefix_code).strip()).split())
    next_codeline = ' '.join(next_codeline.split())
    prefix_code_tokenized = tokenizer.tokenize(prefix_code)
    next_codeline_tokenized = tokenizer.tokenize(next_codeline)
    return convert_examples_to_features(prefix_code_tokenized, next_codeline_tokenized, label, tokenizer, args, cache)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        
def convert_examples_to_features(prefix_code_tokenized, next_codeline_tokenized, label, tokenizer, args, cache):
    #source
    code1_tokens = prefix_code_tokenized[:args.block_size-2]
    code1_tokens = [tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens= next_codeline_tokenized[:args.block_size-2]
    code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length
    
    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length
    
    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, block_size=512,pool=None):
        self.examples = []
        data=[]
        cache={}
        prefix_code, next_codeline, label = args.prefix_code, args.next_codeline, 0
        data.append((prefix_code, next_codeline, label, tokenizer, args, cache))
        self.examples = pool.map(get_example,tqdm(data,total=len(data)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False,pool=None):
    dataset = TextDataset(tokenizer, args, block_size=args.block_size, pool=pool)
    return dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def test(args, model, tokenizer, prefix="",pool=None,best_threshold=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, test=True,pool=pool)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    # logger.info("***** Running Test {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    y_preds = logits[:,1] > best_threshold
    return y_preds[0]

def initialize_config_and_model():
    
    global args, model, tokenizer, pool, config
    args.output_dir = "./coseg/saved_models"
    args.model_type = "roberta"
    args.config_name = "microsoft/codebert-base"
    args.model_name_or_path = "microsoft/codebert-base"
    args.tokenizer_name = "roberta-base"
    args.block_size = 400
    args.eval_batch_size = 1
    args.learning_rate = 5e-5
    args.max_grad_norm = 1.0
    args.seed = 123456 
    args.local_rank = -1
    args.no_cuda = False
    args.cache_dir = ""
    args.fp16 = False
    args.do_lower_case = False
    
    pool = multiprocessing.Pool(cpu_cont)
        
    # Setup CUDA, GPU & distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir = args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case = args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    
    print("\nModel Building Done")
    
def predict_for_a_single_datapoint(prefix_code, next_codeline):
    global args, model, pool, config
    args.prefix_code = prefix_code
    args.next_codeline = next_codeline
    return test(args, model, tokenizer,pool=pool,best_threshold = 0.5)

def split_code(code_lines):
    splitted_code = [[code_lines[0]]]
    for i in range(1, len(code_lines)):
        is_split_cell = predict_for_a_single_datapoint(splitted_code[-1], code_lines[i])
        if(is_split_cell == 1):
            splitted_code.append([code_lines[i]])
        else:
            splitted_code[-1].append(code_lines[i])
    return splitted_code

def remove_empty_lines(code_lines):
  cleaned_code = [code for code in code_lines if len(code.strip()) != 0]
  return cleaned_code

# Method to adjust using the breakpoints information to handle nested structures
def adjust_splits(original_splits, original_code):
    adjusted_splits, index_in_the_code, indentations = [], 0, []
    for original_split in original_splits:
        if original_split[0].strip().startswith(LIST_OF_STRUCTURES):
            gap_length = len(original_split[0]) - len(original_split[0].lstrip())
            index = index_in_the_code + 1
            while(index < len(original_code)):
                if len(original_code[index]) - len(original_code[index].lstrip()) <= gap_length:
                    break
                index += 1
            new_split = original_code[index_in_the_code: index]
            adjusted_splits.append(new_split)
            indentations.append(gap_length)
            index_in_the_code += len(original_split)
        else:
            gap_length = len(original_split[0]) - len(original_split[0].lstrip())
            adjusted_splits.append(original_split)
            indentations.append(gap_length)
            index_in_the_code += len(original_split)
    return adjusted_splits, indentations

def calculate_inferrence_time(examples):
    
    total_time = 0
    for example in tqdm(examples):
        
        code_lines = remove_empty_lines(example)
        
        start_time = time.time()
        splitted_code = split_code(code_lines)
        total_time += (time.time() - start_time)
        
        adjusted_splits, indentaions = adjust_splits(splitted_code, code_lines)

        start_time = time.time()
        predictions = document_prediction_using_codebert.generate_doc(adjusted_splits)
        total_time += (time.time() - start_time)
        
    print("\nAverage time taken: ", total_time/len(examples))
    sys.exit()
        
if __name__ == "__main__":
    initialize_config_and_model()
    
    code_lines_1 = ['parameters = {', 
                    "    'application': 'binary',", 
                    "    'objective': 'binary',", 
                    "    'metric': 'auc',", 
                    "    'is_unbalance': 'true',", 
                    "    'boosting': 'gbdt',", 
                    "    'num_leaves': 31,", 
                    "    'feature_fraction': 0.5,", 
                    "    'bagging_fraction': 0.5,", 
                    "    'bagging_freq': 20,", 
                    "    'learning_rate': 0.05,", 
                    "    'verbose': 0", 
                    '}', 
                    'train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=cat_cols)', 
                    'val_data = lightgbm.Dataset(X_val, label=y_val)', 
                    'model = lightgbm.train(parameters,', 
                    '                       train_data,', 
                    '                       valid_sets=val_data,', 
                    '                       num_boost_round=5000,', 
                    '                       early_stopping_rounds=100)']
    
    code_lines = remove_empty_lines(code_lines_1)
    result = ["Original Code: \n\n"]
    result.append(code_lines)
    print("\nOriginal Code: \n\n")
    for o in code_lines:
        print(o)
        
    splitted_code = split_code(code_lines)
    print("\nOriginally splitted Code : \n\n", splitted_code)
    result.append("\n\nOriginally splitted Code : \n\n")
    result.append(splitted_code)
        
    adjusted_splits, indentaions = adjust_splits(splitted_code, code_lines)
    print("\nAdjusted splits : \n\n", adjusted_splits)
    result.append("\n\nAdjusted splits : \n\n")
    result.append(adjusted_splits)
    
    predictions = combine.combine_predictions(document_prediction_using_codebert.generate_doc(adjusted_splits), indentaions)
    print("\nCodeBERT Documentation : \n")
    result.append("\n\nCodeBERT Documentation : \n\n")
    result.append(predictions)
    for output in predictions:
        print(output)
        
    predictions = combine.combine_predictions(document_prediction_using_unixcoder.generate_doc(adjusted_splits), indentaions)
    print("\nUnixCoder Documentation : \n")
    result.append("\n\nUnixCoder Documentation : \n\n")
    result.append(predictions)
    for output in predictions:
        print(output)
        
    predictions = combine.combine_predictions(graphcodebert_predict.generate_doc(adjusted_splits), indentaions)
    print("\nGraphCodeBERT Documentation : \n")
    result.append("\n\nGraphCodeBERT Documentation : \n\n")
    result.append(predictions)
    for output in predictions:
        print(output)
    
    predictions = combine.combine_predictions(codet5_predict.generate_doc(adjusted_splits), indentaions)
    print("\nCodeT5 Documentation : \n")
    result.append("\n\nCodeT5 Documentation : \n\n")
    result.append(predictions)
    for output in predictions:
        print(output)
        
    predictions = combine.combine_predictions(plbart_predict.generate_doc(adjusted_splits), indentaions)
    print("\nPLBART Documentation : \n")
    result.append("\n\nPLBART Documentation : \n\n")
    result.append(predictions)
    for output in predictions:
        print(output)
        
    with open("./coseg/inference/output.txt", "w") as output:
        for line in result:
            output.write(str(line))
    