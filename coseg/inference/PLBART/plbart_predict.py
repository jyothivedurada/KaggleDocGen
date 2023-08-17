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

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import sys

import tensorflow as tf
import tensorboard as tb

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from PLBART.models import build_or_load_gen_model
from PLBART.utils import read_examples_summarize_2
from PLBART.configs import add_args, set_seed, set_dist

def make_prediction(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    for batch in eval_dataloader:
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if args.n_gpu > 1:
                    preds = model.module.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                else:
                    preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    return pred_nls

def generate_doc(code_splits):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.adam_epsilon=1e-08
    args.add_lang_ids=False
    args.add_task_prefix=False
    args.always_save_model=True
    args.beam_size=10
    # args.cache_path='/raid/cs21mtech12001/Research/CodeT5/Repository/output/summarize/python-notebook/plbart_large_scscm(todo-18)_lr5_bs8_src256_trg128_pat2_e15/cache_data'
    args.config_name=''
    # args.data_dir='/raid/cs21mtech12001/Research/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18'
    args.data_num=-1
    args.dev_filename=None
    args.do_eval=True
    args.do_eval_bleu=True
    args.do_lower_case=False
    args.do_test=True
    args.do_train=True
    args.eval_batch_size=8
    args.eval_steps=-1
    args.eval_task=''
    args.gradient_accumulation_steps=1
    args.lang='python-notebook'
    args.learning_rate=5e-05
    args.load_model_path='./codoc/CodeT5-PLBART/output/summarize/python-notebook/plbart_large_scscm(todo-18)_lr5_bs8_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin'
    args.local_rank=-1
    args.log_steps=-1
    args.max_grad_norm=1.0
    args.max_source_length=256
    args.max_steps=-1
    args.max_target_length=128
    args.model_name_or_path='uclanlp/plbart-large'
    args.model_type='plbart'
    args.no_cuda=False
    args.num_train_epochs=15
    # args.output_dir='/raid/cs21mtech12001/Research/CodeT5/Repository/output/summarize/python-notebook/plbart_large_scscm(todo-18)_lr5_bs8_src256_trg128_pat2_e15'
    args.patience=2
    # args.res_dir='/raid/cs21mtech12001/Research/CodeT5/Repository/output/summarize/python-notebook/plbart_large_scscm(todo-18)_lr5_bs8_src256_trg128_pat2_e15/prediction'
    args.res_fn='python_notebooks_summarization_PLBART_scscm'
    args.save_last_checkpoints=True
    args.save_steps=-1
    args.seed=1234
    args.start_epoch=0
    args.sub_task='python-notebook'
    # args.summary_dir='/raid/cs21mtech12001/Research/CodeT5/Repository/summary'
    args.task='summarize'
    args.test_filename=None
    args.tokenizer_name='uclanlp/plbart-large'
    args.train_batch_size=8
    args.train_filename=None
    args.train_steps=-1
    args.warmup_steps=1000
    args.weight_decay=0.0
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(1)

    predictions = []
    for code in code_splits:
        eval_examples, eval_data = read_examples_summarize_2(code, pool, tokenizer, args, 'test')
        result = make_prediction(args, eval_data, eval_examples, model, tokenizer, 'test', "best-bleu")
        predictions.extend(result)
        
    return predictions
        
if __name__ == "__main__":
    code_lines = [['parameters = {', 
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
                    '                       early_stopping_rounds=100)']]
    print(generate_doc(code_lines))
