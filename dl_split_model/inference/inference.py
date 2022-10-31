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
import document_prediction_using_codebert
import document_prediction_using_unixcoder

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
    args.output_dir = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/saved_models/by-ast-and-comments/positives-first/epoch-2"
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
    
    # examples = [code_lines_1,code_lines_2,code_lines_3,code_lines_4,code_lines_5,code_lines_6,
    #             code_lines_7,code_lines_8,code_lines_9,code_lines_10,code_lines_11,code_lines_12,code_lines_13,
    #             code_lines_14,code_lines_15,code_lines_16,code_lines_17,code_lines_18,code_lines_19,code_lines_20,
    #             code_lines_21,code_lines_22,code_lines_23,code_lines_24,code_lines_25]
    # calculate_inferrence_time(examples)
    
    code_lines = remove_empty_lines(code_lines_15)
    print("\nOriginal Code: \n\n")
    for o in code_lines:
        print(o)
        
    splitted_code = split_code(code_lines)
    print("\nOriginally splitted Code : \n\n", splitted_code)
        
    adjusted_splits, indentaions = adjust_splits(splitted_code, code_lines)
    print("\nAdjusted splits : \n\n", adjusted_splits)
    
    #adjusted_splits = [['try:', '    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()', "    print('Running on TPU ', tpu.master())", 'except ValueError:', '    tpu = None'], ['    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()', "    print('Running on TPU ', tpu.master())", 'except ValueError:', '    tpu = None'], ['if tpu:', '    tf.config.experimental_connect_to_cluster(tpu)', '    tf.tpu.experimental.initialize_tpu_system(tpu)', '    strategy = tf.distribute.experimental.TPUStrategy(tpu)', 'else:', '    strategy = tf.distribute.get_strategy()'], ['print("REPLICAS: ", strategy.num_replicas_in_sync)']]
    
    predictions = combine.combine_predictions(document_prediction_using_codebert.generate_doc(adjusted_splits), indentaions)
    print("\nCodeBERT Documentation : \n")
    for output in predictions:
        print(output)
        
    predictions = combine.combine_predictions(document_prediction_using_unixcoder.generate_doc(adjusted_splits), indentaions)
    print("\nUnixCoder Documentation : \n")
    for output in predictions:
        print(output)
    