import os
import json
from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import re
import tqdm
import random

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
                #if(len(tokval) > 1 and (tokval.isalpha() or tokval.islower() or tokval.isupper() or english_check.match(tokval))):
                    code_tokens.append(tokval)
    except:
        return []
    return code_tokens

def change_the_format(notebook, notebook_name):
    dataset_as_list = []
    for code_number in notebook:
        for positive_negative_example in notebook[code_number]["positive_negative_examples"]:
            datapoint = {}
            datapoint["notebook"] = notebook_name
            datapoint["prefix_code"] = positive_negative_example[0]
            datapoint["next_codeline"] = positive_negative_example[1]
            datapoint["label"] = positive_negative_example[2]
            if len("".join(datapoint["prefix_code"]).strip()) != 0 and len(datapoint["next_codeline"].strip()) != 0:
                dataset_as_list.append(datapoint)
    return dataset_as_list

def main():
    dataset_file = open(f"./coseg/dataset/processed_data/dataset.json", "r") 
    file_as_json = json.loads(dataset_file.read())
    number_of_notebooks = len(file_as_json)
    train_split, test_split, valid_split = int(number_of_notebooks * 0.8), int(number_of_notebooks * 0.1), int(number_of_notebooks * 0.1)
    test_dataset, train_dataset, valid_dataset = [], [], []
    counter = 0
    for notebook_name in tqdm.tqdm(file_as_json):
        dataset_as_list = change_the_format(file_as_json[notebook_name], notebook_name)
        if(counter < train_split):
            train_dataset.extend(dataset_as_list)
        elif(counter >= train_split and counter < train_split + test_split):
            test_dataset.extend(dataset_as_list)
        else:
            valid_dataset.extend(dataset_as_list)
        counter += 1
    print("Train dataset size: {}, test dataset size: {} and valid dataset size: {}".format(len(train_dataset), len(test_dataset), len(valid_dataset)))

    # Shuffle each split and save
    random.shuffle(train_dataset)
    random.shuffle(valid_dataset)
    random.shuffle(test_dataset)
    dataset_folder = "./coseg/dataset/splitted_data"
    with open(f"{dataset_folder}/train.json", 'w') as f:
        for item in train_dataset:
            f.write(json.dumps(item) + "\n")
    with open(f"{dataset_folder}/test.json", 'w') as f:
        for item in test_dataset:
            f.write(json.dumps(item) + "\n")
    with open(f"{dataset_folder}/valid.json", 'w') as f:
        for item in valid_dataset:
            f.write(json.dumps(item) + "\n")

main()