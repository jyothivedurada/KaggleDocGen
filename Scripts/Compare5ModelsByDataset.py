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
from ast_split_model.model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

import CalculateBLEUScore3
import CalculateROUGEScore
from bert_score import BERTScorer
from bleurt import score

from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re

english_check = re.compile(r'[a-z]')

# Code to tokenize python code
def tokenize_code(code_lines):
    code_as_string = " ".join(code_lines)
    tokenized_code = tokenize(BytesIO(code_as_string.encode('utf-8')).readline)
    code_tokens = []
    unnecessary_tokens = ["\n", "", "utf-8"]
    try:
        for _, tokval, _, _, _ in tokenized_code:
            if tokval not in unnecessary_tokens:
                if(len(tokval) > 1 and (tokval.isalpha() or tokval.islower() or tokval.isupper() or english_check.match(tokval))):
                    code_tokens.append(tokval)
    except:
        return []
    return code_tokens

# Use CUDA/GPU for BLEURT
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,6"

# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

# Load BERT score model
bert_scorer = BERTScorer(lang = "en", model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline = True)

# Load BLEURT score model
bleurt_checkpoint = "/home/cs19btech11056/cs21mtech12001-Tamal/BLEURT/BLEURT-20"
bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

# Method to calculate BERT score between 2 lists
def CalculateBERTscoreBetweenLists(predictions_as_list, gt_as_list):
    bert_scores = bert_scorer.score(predictions_as_list, gt_as_list)
    average_bert_score = (sum(bert_scores[2])/len(bert_scores[2])).item() * 100
    print("\nAverage BERT score: ", "{:.3f}".format(average_bert_score))
    return average_bert_score

# Method to calculate BERT score between 2 lists
def CalculateBLEURTscoreBetweenLists(predictions_as_list, gt_as_list):
    bleurt_score = bleurt_scorer.score(candidates = predictions_as_list, references = gt_as_list)
    average_bleurt_score = (sum(bleurt_score)/len(bleurt_score)) * 100
    print("\nAverage BLEURT score: ", "{:.3f}".format(average_bleurt_score))
    return average_bleurt_score

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

# Method to generate doc from code
def generate_doc(list_of_code_tokens, model_path):
    
    # Configs
    model_type = "roberta"
    model_name_or_path = "microsoft/codebert-base"
    load_model_path = model_path
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
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
    for code_tokens in tqdm(list_of_code_tokens):
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
    
    # Test data file
    #dataset_file_path = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/test_dataset.jsonl"
    dataset_file_path = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl"
    dataset_file = list(open(dataset_file_path, "r"))
    
    # CASE 1 Model
    model_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/checkpoint-best-bleu/pytorch_model.bin"
    
    # CASE 2 Model
    model_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/checkpoint-best-bleu/pytorch_model.bin"
    
    # CASE 3 Model
    model_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm/checkpoint-best-bleu/pytorch_model.bin"
    
    # CASE 4 Model
    model_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment/checkpoint-best-bleu/pytorch_model.bin"
    
    # CASE 5 Model
    model_5 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/checkpoint-best-bleu/pytorch_model.bin"
    
    models = [model_1, model_2, model_3, model_4, model_5]
    
    model_descriptions = ["For only code - cleaned markdown",
                          "For only code - cleaned and summarized markdown",
                          "For english code tokens - cleaned and summarized markdown",
                          "For code + comment - cleaned and summarized markdown",
                          "For split code - cleaned and summarized markdown and comments"]
    
    list_of_codes, list_of_code_tokens, ref_texts = [], [], []
    for datapoint_str in dataset_file:
        datapoint = json.loads(datapoint_str)
        list_of_codes.append(datapoint["code"])
        list_of_code_tokens.append(datapoint["code_tokens"])
        ref_texts.append(datapoint["docstring"])
        
    for i in range(len(model_descriptions)):
        if(i != 4):
            continue
        print("\n", model_descriptions[i])
        if(i == 2):
            only_english_code_tokens = [tokenize_code(code) for code in list_of_codes]
            print("\n", only_english_code_tokens[:5])
            predicted_texts = generate_doc(only_english_code_tokens, models[i])
        else:
            predicted_texts = generate_doc(list_of_code_tokens, models[i])
        print("\nSample Ref Texts: ", ref_texts[:5])
        print("\nSample Pred Texts: ", predicted_texts[:5])
        CalculateROUGEScore.RougeScoreBetween2Lists(predicted_texts, ref_texts)
        CalculateBLEUScore3.BleuScoreBetween2Lists(predicted_texts, ref_texts)
        CalculateBERTscoreBetweenLists(predicted_texts, ref_texts)
        CalculateBLEURTscoreBetweenLists(predicted_texts, ref_texts)
        
    
    