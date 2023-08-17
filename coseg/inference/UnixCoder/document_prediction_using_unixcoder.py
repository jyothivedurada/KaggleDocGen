# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from UnixCoder.document_model_unixcoder import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)

from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

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

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids     
        
def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length,stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length-5]
        source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
            )
        )
    return features

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

# Method to generate doc for splits
def generate_doc(code_splits):
    
    # Argumnents
    seed = 42
    model_name_or_path = "microsoft/unixcoder-base"
    beam_size = 10
    max_target_length = 128
    load_model_path = None
    output_dir = "./codoc/UniXcoder/output/Notebooks/scscm"
    max_source_length = 256
    max_target_length = 128
    eval_batch_size = 48 
    
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    # Set seed
    set_seed(seed)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    config = RobertaConfig.from_pretrained(model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(model_name_or_path,config=config) 

    model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=beam_size,max_length=max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)

    if load_model_path is not None:
        print("\nreload model from {}".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))
    
    model.to(device)   
    
    if n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    checkpoint_prefix = 'checkpoint-best-bleu/pytorch_model.bin'
    output_dir = os.path.join(output_dir, checkpoint_prefix)  
    model_to_load = model.module if hasattr(model, 'module') else model  
    model_to_load.load_state_dict(torch.load(output_dir))  
    
    predictions = []
    for code in code_splits:
        code_tokens = tokenize_code(code)             

        eval_examples = read_examples(code_tokens)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, max_source_length, max_target_length, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)   

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        model.eval() 
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]                  
            with torch.no_grad():
                preds = model(source_ids)   
                # convert ids to text
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    p.append(text)
                    
        model.train()
        predictions.extend(p)
        
    return predictions
                
if __name__ == "__main__":
    code_lines = [['def indices(data, feat_index):', 
                     '    value_dict = value_dicts[feat_index]', 
                     '    return data[cat_cols[feat_index]].apply(lambda x: value_dict[x])', 
                     '', 
                     'def one_hot(indices, feat_index):', 
                     '    return to_categorical(indices, num_classes=len(value_dicts[feat_index]))']]
    print(generate_doc(code_lines))


