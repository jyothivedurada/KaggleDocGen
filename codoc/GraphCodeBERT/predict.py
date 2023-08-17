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
from model import Seq2Seq
from tqdm import tqdm, trange
import bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

# hide the loading messages
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
}

logger = logging.getLogger(__name__)
#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./codoc/GraphCodeBERT/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
#remove comments, tokenize code and extract dataflow     
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        print("\nParsing failed")
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        print("\nDFG creation failed")
        dfg=[]
    return code_tokens,dfg

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

def read_examples(code_lines):
    """Read examples from filename."""
    examples=[]
    code = ';'.join(code_lines)       
    examples.append(
        Example(
            idx = 0,
            source = code,
            target = "",
        ) 
    )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./codoc/GraphCodeBERT/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        code_tokens,dfg=extract_dataflow(example.source, parsers["python"], "python")
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        
        #truncating
        code_tokens=code_tokens[:args.max_source_length-3][:512-3]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:args.max_source_length-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.max_source_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length      
        source_mask = [1] * (len(source_tokens))
        source_mask+=[0]*padding_length        
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
      

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("position_idx: {}".format(position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.max_source_length,self.args.max_source_length),dtype=np.bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].source_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes         
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask), 
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask),)
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def generate_doc(code_splits):

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.seed = 42
    args.model_type = "roberta"
    args.config_name = "microsoft/graphcodebert-base"
    args.tokenizer_name = "microsoft/graphcodebert-base"
    args.model_name_or_path = "microsoft/graphcodebert-base"
    args.beam_size = 10
    args.max_source_length = 320
    args.max_target_length = 256
    args.load_model_path = "./codoc/GraphCodeBERT/output/notebooks/scscm/checkpoint-best-bleu/pytorch_model.bin"
    args.eval_batch_size = 1

    # Setup CUDA, GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args.seed)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    
    predictions=[]
    for code in code_splits:
        eval_examples = read_examples(code)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
        eval_data = TextDataset(eval_features,args) 
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

        model.eval() 
        p=[]
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                    
            with torch.no_grad():
                preds = model(source_ids,source_mask,position_idx,att_mask)  
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    p.append(text)
        model.train()
        predictions.extend(p)

    return predictions
            
if __name__ == "__main__":

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
    code_lines_2 = ['t4_params = {', 
                    "    'boosting_type': 'gbdt', 'objective': 'multiclass', 'nthread': -1, 'silent': True,", 
                    "    'num_leaves': 2**4, 'learning_rate': 0.05, 'max_depth': -1,", 
                    "    'max_bin': 255, 'subsample_for_bin': 50000,", 
                    "    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6, 'reg_alpha': 1, 'reg_lambda': 0,", 
                    "    'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 1}",
                    't4 = lgbm.sklearn.LGBMClassifier(n_estimators=1000, seed=0, **t4_params)']
    code_lines_3 = ['def region_plot(df):', 
                    '    data = df.copy()', 
                    "    data['time_to_failure'] = data['time_to_failure']*100", 
                    "    data['time'] = data.index", 
                    "    data['time'] = data['time']*(1/4e6)", 
                    "    data['Time [sec]'] = data['time'] - data['time'].min()", 
                    "    data[['acoustic_data','time_to_failure','Time [sec]']].plot(x='Time [sec]', figsize=(8,5))", 
                    '    return']
    code_lines_4 = ['import numpy as np ', 
                    'import pandas as pd ', 
                    'import os', 
                    "for dirname, _, filenames in os.walk('/kaggle/input'):", 
                    '    for filename in filenames:', 
                    '        print(os.path.join(dirname, filename))']
    code_lines_5 = ['test_filenames = test_gen.filenames',
                    "df_preds['file_names'] = test_filenames", 
                    'def extract_id(x):', 
                    "    a = x.split('/')", 
                    "    b = a[1].split('.')", 
                    '    extracted_id = b[0]', 
                    '    return extracted_id', 
                    "df_preds['id'] = df_preds['file_names'].apply(extract_id)", 
                    'df_preds.head()']
    code_lines_6 = ['def submit(predictions):', 
                   "    submit = pd.read_csv('../input/sample_submission.csv')", 
                   '    submit["target"] = predictions', 
                   '    submit.to_csv("submission.csv", index=False)', 
                   'def fallback_auc(y_true, y_pred):', 
                   '    try:', 
                   '        return metrics.roc_auc_score(y_true, y_pred)', 
                   '    except:', 
                   '        return 0.5', 
                   'def auc(y_true, y_pred):', 
                   '    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)']
    code_lines_7 = ['def catboost_fit(model, X_train, y_train, X_val, y_val):', 
                    '    train_pool = Pool(X_train, y_train)', 
                    '    val_pool = Pool(X_val, y_val)', 
                    '    model.fit(train_pool, eval_set=val_pool)', 
                    '    ', 
                    '    return model', 
                    '', 
                    'model = CatBoostRegressor(iterations=20000, ', 
                    '                          max_depth=9,', 
                    "                          objective='MAE',", 
                    "                          task_type='GPU',", 
                    '                          verbose=False)', 
                    'model = catboost_fit(model, tr_X, tr_y, val_X, val_y)']
    code_lines_8 = ["dict_columns = ['belongs_to_collection', 'genres', 'production_companies',", 
                    "                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']", 
                    '', 
                    'def text_to_dict(df):', 
                    '    for column in dict_columns:', 
                    '        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )', 
                    '    return df', 
                    '        ', 
                    'train = text_to_dict(train)', 
                    'test = text_to_dict(test)']
    code_lines_9 = ['n_classes = 12  ', 
                    'data, labels = make_classification(n_samples=2000, n_features=100, ', 
                    '                                   n_informative=50, n_classes=n_classes, ', 
                    '                                   random_state=random_state)', 
                    '', 
                    'X, X_test, y, y_test = train_test_split(data, labels, test_size=0.2, ', 
                    '                                        random_state=random_state)', 
                    '    ', 
                    'X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, ', 
                    '                                                      random_state=random_state)', 
                    '', 
                    "print('Data shape:')", 
                    "print('X_train: %s, X_valid: %s, X_test: %s \\n' %(X_train.shape, X_valid.shape, ", 
                    '                                                  X_test.shape))', 
                    '    ']
    code_lines_10 = ['lr = LogisticRegressionCV(Cs=10, dual=False, fit_intercept=True, ', 
                     '                          intercept_scaling=1.0, max_iter=100,', 
                     "                          multi_class='ovr', n_jobs=1, penalty='l2', ", 
                     '                          random_state=random_state,', 
                     "                          solver='lbfgs', tol=0.0001)", 
                     '', 
                     'lr.fit(XV, y_valid)', 
                     'y_lr = lr.predict_proba(XT)', 
                     "print('{:20s} {:2s} {:1.7f}'.format('Log_Reg:', 'logloss  =>', log_loss(y_test, y_lr)))", 
                     '']
    code_lines_11 = ['categorical_list = []', 
                     'numerical_list = []', 
                     'for i in application.columns.tolist():', 
                     "    if application[i].dtype=='object':", 
                     '        categorical_list.append(i)', 
                     '    else:', 
                     '        numerical_list.append(i)', 
                     "print('Number of categorical features:', str(len(categorical_list)))", 
                     "print('Number of numerical features:', str(len(numerical_list)))"]
    code_lines_12 = ['def cel(y_true, y_pred):', 
                     '    y_true = torch.argmax(y_true, axis=-1)', 
                     '    return nn.CrossEntropyLoss()(y_pred, y_true.squeeze())', 
                     '', 
                     'def accuracy(y_true, y_pred):', 
                     '    y_true = torch.argmax(y_true, axis=-1).squeeze()', 
                     '    y_pred = torch.argmax(y_pred, axis=-1).squeeze()', 
                     '    return (y_true == y_pred).float().sum()/len(y_true)']
    code_lines_13 = ['train_images = []', 
                     "image_dirs = np.take(os.listdir('../input/train'), select_rows)", 
                     '', 
                     'for image_dir in tqdm(sorted(image_dirs)):', 
                     "    image = imread('../input/train/'+image_dir)", 
                     '    train_images.append(image)', 
                     '    del image', 
                     '    gc.collect()', 
                     '    ', 
                     'train_images = np.array(train_images)']
    code_lines_14 = ['def indices(data, feat_index):', 
                     '    value_dict = value_dicts[feat_index]', 
                     '    return data[cat_cols[feat_index]].apply(lambda x: value_dict[x])', 
                     '', 
                     'def one_hot(indices, feat_index):', 
                     '    return to_categorical(indices, num_classes=len(value_dicts[feat_index]))']
    code_lines_15 = ['def submit(predictions):', 
                     "    submit = pd.read_csv('sample_submission.csv')", 
                     '    print(len(submit), len(predictions))   ', 
                     '    submit["scalar_coupling_constant"] = predictions', 
                     '    submit.to_csv("/kaggle/working/workingsubmission-test.csv", index=False)', 
                     'submit(test_prediction)', 
                     '', 
                     "print ('Total training time: ', datetime.now() - start_time)", 
                     '', 
                     'i=0', 
                     'for mol_type in mol_types: ', 
                     '    print(mol_type,": cv score is ",cv_score[i])', 
                     '    i+=1', 
                     'print("total cv score is",cv_score_total)']
    code_lines_16 = ['try:', 
                     '    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()', 
                     "    print('Running on TPU ', tpu.master())", 
                     'except ValueError:', 
                     '    tpu = None', 
                     '', 
                     'if tpu:', 
                     '    tf.config.experimental_connect_to_cluster(tpu)', 
                     '    tf.tpu.experimental.initialize_tpu_system(tpu)', 
                     '    strategy = tf.distribute.experimental.TPUStrategy(tpu)', 
                     'else:', 
                     '    strategy = tf.distribute.get_strategy()', 
                     '', 
                     'print("REPLICAS: ", strategy.num_replicas_in_sync)']
    code_lines_17 = ['from datetime import datetime', 
                     'from os import scandir', 
                     '', 
                     'def convert_date(timestamp):', 
                     '    d = datetime.utcfromtimestamp(timestamp)', 
                     "    formated_date = d.strftime('%d %b %Y')", 
                     '    return formated_date', 
                     '', 
                     'def get_files():', 
                     "    dir_entries = scandir('my_directory/')", 
                     '    for entry in dir_entries:', 
                     '        if entry.is_file():', 
                     '            info = entry.stat()', 
                     "            print(f'{entry.name}\\t Last Modified: {convert_date(info.st_mtime)}')", 
                     '']
    code_lines_18 = ['model = Net.build(width = 96, height = 96, depth = 3, classes = 2)', 
                     'from keras.optimizers import SGD, Adam, Adagrad', 
                     "model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])"]
    code_lines_19 = ['def seed_everything(seed=1029):', 
                     '    random.seed(seed)', 
                     "    os.environ['PYTHONHASHSEED'] = str(seed)", 
                     '    np.random.seed(seed)', 
                     '    torch.manual_seed(seed)', 
                     '    torch.cuda.manual_seed(seed)', 
                     '    torch.backends.cudnn.deterministic = True', 
                     'seed_everything()']
    code_lines_20 = ['def image_pad(image, new_height, new_width):', 
                     '    height, width = image.shape', 
                     '', 
                     '    im_bg = np.zeros((new_height, new_width))', 
                     '', 
                     '    pad_left = int( (new_width - width) / 2)', 
                     '    pad_top = int( (new_height - height) / 2)', 
                     '', 
                     '    im_bg[pad_top:pad_top + height,', 
                     '          pad_left:pad_left + width] = image', 
                     '', 
                     '    return im_bg']
    code_lines_21 = ['from sklearn import preprocessing', 
                     '', 
                     'def normalizeFeatures(df):', 
                     '    min_max_scaler = preprocessing.MinMaxScaler()', 
                     '    x_scaled = min_max_scaler.fit_transform(df)', 
                     '    df_normalized = pd.DataFrame(x_scaled, columns=df.columns)', 
                     '    ', 
                     '    return df_normalized', 
                     '', 
                     'def normalizePanel(pf):', 
                     '    ', 
                     '    pf2 = {}', 
                     '    for i in range(pf.shape[2]):', 
                     '        pf2[i] = normalizeFeatures(pf.ix[:,:,i])', 
                     '        ', 
                     '    return pd.Panel(pf2)']
    code_lines_22 = ['def convertMatToDictionary(path):', 
                     '    ', 
                     '    try: ', 
                     '        mat = loadmat(path)', 
                     "        names = mat['dataStruct'].dtype.names", 
                     "        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}", 
                     '        ', 
                     '    except ValueError:     ', 
                     "        print('File ' + path + ' is corrupted. Will skip this file in the analysis.')", 
                     '        ndata = None', 
                     '    ', 
                     '    return ndata']
    code_lines_23 = ['files=[]', 
                     "for dirname, _, filenames in os.walk('../input/osic-pulmonary-fibrosis-progression/train'):", 
                     '    for filename in filenames:', 
                     '        files.append(os.path.join(dirname, filename))']
    code_lines_24 = ['sample_audio = []', 
                     'total = 0', 
                     'for x in subFolderList:', 
                     '    ', 
                     "    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]", 
                     '    total += len(all_files)', "    sample_audio.append(audio_path  + x + '/'+ all_files[0])", 
                     '    ', 
                     "    print('count: %d : %s' % (len(all_files), x ))", 
                     'print(total)']
    code_lines_25 = ['from dlib import get_frontal_face_detector', 
                     'detector = get_frontal_face_detector()', 
                     '', 
                     'def detect_dlib(detector, images):', 
                     '    faces = []', 
                     '    for image in images:', 
                     '        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)', 
                     '        boxes = detector(image_gray)', 
                     '        box = boxes[0]', 
                     '        face = image[box.top():box.bottom(), box.left():box.right()]', 
                     '        faces.append(face)', 
                     '    return faces', 
                     '', 
                     'times_dlib = []']

    code_lines = [code_lines_15]
    print(generate_doc(code_lines))

