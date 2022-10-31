from __future__ import absolute_import
import gc
import os
import sys
import pickle
import torch
import json
import random
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import combine

# Class to represent each sample
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

# Method to convert code-tokens to examples
def read_examples(code_as_tokens):
    examples=[]
    code=' '.join(code_as_tokens).replace('\n',' ')
    code=' '.join(code.strip().split())
    examples.append(
            Example(
                    idx = 0,
                    source=code,
                    target = "",
                    ) 
            )
    return examples

# Final features
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       

# Convert examples to features
def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

# Set the seed for random generators
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# Code to tokenize python code
def tokenize_code(code_lines):
    code_as_string = " ".join(code_lines)
    tokenized_code = tokenize(BytesIO(code_as_string.encode('utf-8')).readline)
    code_tokens = []
    unnecessary_tokens = ["\n", "", "utf-8"]
    try:
        for _, tokval, _, _, _ in tokenized_code:
            if tokval not in unnecessary_tokens:
                #if(len(tokval) > 1 and (tokval.islower() or tokval.isupper() or english_check.match(tokval))):
                    code_tokens.append(tokval)
    except:
        return []
    return code_tokens

# Method to generate doc from code
def generate_doc(clusters):
    
    # Configs
    model_type = "roberta"
    model_name_or_path = "microsoft/codebert-base"
    load_model_path = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18-outlier-removed/checkpoint-best-bleu/pytorch_model.bin"
    max_source_length = 256
    max_target_length = 128
    beam_size = 10
    eval_batch_size = 64
    
    # Other default configs
    local_rank = -1
    no_cuda = False
    seed = 42
    config_name = ""
    tokenizer_name = ""
    do_lower_case = True

    # Setup CUDA, GPU & distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    gc.collect()
    torch.cuda.empty_cache()
    
    # Set seed
    set_seed(seed)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(config_name if config_name else model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,do_lower_case = do_lower_case)
    
    # Build model
    encoder = model_class.from_pretrained(model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=beam_size,max_length=max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if load_model_path is not None:
        #print("\nLoaded model: {}".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))
        
    model.to(device)
    if local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    
    # Generate predictions for each section
    predictions = []
    for cluster in clusters:
        code_tokens = tokenize_code(cluster)
        #print("\nTokenized code: {}".format(code_tokens))
        eval_examples = read_examples(code_tokens)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, max_source_length, max_target_length,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
        eval_data = TensorDataset(all_source_ids,all_source_mask)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        model.eval() 
        p=[]
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask= batch                  
            with torch.no_grad():
                preds = model(source_ids=source_ids,source_mask=source_mask)  
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    p.append(text)
        model.train()
        #print("\nPrediction: ", p)
        predictions.extend(p)
    return predictions

if __name__ == "__main__":
    code_lines = [['import numpy as np ', 'import pandas as pd ']]
    
    for output in combine.combine_predictions(generate_doc(code_lines)):
        print(output)


