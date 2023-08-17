# download packages
#!pip install transformers==4.8.2

# import packages
import re
import sys
import torch
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, BloomTokenizerFast, BloomForCausalLM

import os
import json

## Define class and functions

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task
        
def read_summarize_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
    return examples

class customTrainingArguments(TrainingArguments):
    def __init__(self,*args, **kwargs):
        super(customTrainingArguments, self).__init__(*args, **kwargs)

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        Name the device the number you use.
        """
        return torch.device("cuda")

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        # _ = self._setup_devices
        # I set to one manullay
        self._n_gpu = 1
        return self._n_gpu

# Dataset class
class NotebookSummarizationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        # define variables    
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        examples = read_summarize_examples(file_path)
        # iterate through the dataset
        for example in tqdm(examples):
            # prepare the text
            prep_txt = f'<|startoftext|>Code: {example.source}<|pad|>Documentation: {example.target}<|endoftext|>'
            # tokenize
            encodings_dict = tokenizer(prep_txt, truncation=True,
                                       max_length=max_length, padding="max_length")
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(example.target)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

## Load model and data
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"

# set model name
model_name = "bigscience/bloomz-560m"
# seed
torch.manual_seed(42)

# load tokenizer and model
tokenizer = BloomTokenizerFast.from_pretrained(model_name, bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')
model = BloomForCausalLM.from_pretrained("./codoc/BLOOMz/results-bloomz-560m-csn/checkpoint-125910").cuda()
model.resize_token_embeddings(len(tokenizer))
# device = torch.device("cuda:3")
# model.to(device)
# model = torch.nn.DataParallel(model)
print("GPUs: ", torch.cuda.device_count())

# prepare and load dataset
data_dir = "./notebooks-dataset/splitted_data/scscm"
train_fn = '{}/train_dataset.jsonl'.format(data_dir)
dev_fn = '{}/valid_dataset.jsonl'.format(data_dir)
test_fn = '{}/test_dataset.jsonl'.format(data_dir)
train_dataset = NotebookSummarizationDataset(train_fn, tokenizer, max_length=512)
dev_dataset = NotebookSummarizationDataset(dev_fn, tokenizer, max_length=512)
#test_dataset = NotebookSummarizationDataset(test_fn, tokenizer, max_length=512)

## Train
#--------
# creating training arguments
training_args = customTrainingArguments(output_dir='./codoc/BLOOMz/results-bloomz-560m-scscm', num_train_epochs=2, logging_steps=10,
                                 load_best_model_at_end=True, save_strategy="epoch", evaluation_strategy="epoch",
                                 per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                 warmup_steps=100, weight_decay=0.01, logging_dir='logs')

# start training
Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])}).train()

## Test

# set the model to eval mode
_ = model.eval()

# run model inference on all test data
test_examples = read_summarize_examples(test_fn)
original_labels, predicted_labels, original_codes, predicted_texts = [], [], [], []
# iter over all of the test data
for test_example in tqdm(test_examples):
    # create prompt (in compliance with the one used during training)
    prompt = f'<|startoftext|>Code: {test_example.source}<|pad|>Documentation:'
    # generate tokens
    generated = tokenizer(f"{prompt}", return_tensors="pt").input_ids.cuda()
    # perform prediction
    sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=512, top_p=0.90, 
            temperature=0, num_return_sequences=0)
    # decode the predicted tokens into texts
    predicted_text  = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    # extract the predicted sentiment
    try:
        pred_documentation = re.findall("Documentation: (.*)", predicted_text)[-1]
    except:
        pred_documentation = "None"
    # append results
    original_labels.append(test_example.target)
    predicted_labels.append(pred_documentation)
    original_codes.append(test_example.source)
    predicted_texts.append(predicted_text)

# transform result into dataframe
df = pd.DataFrame({'original_code': original_codes, 'predicted_label': predicted_labels, 
                    'original_label': original_labels, 'predicted_text': predicted_texts})
df.to_csv("predicted-output-bloomz-560m-scscm.csv")
