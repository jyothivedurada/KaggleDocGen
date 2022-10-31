import os
import json
from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import re
import tqdm

english_check = re.compile(r'[a-z]')

# Remove "." from the end
def remove_dot_from_end(text):
    text = text.strip()
    return text[:-1].strip() if len(text) != 0 and text[-1] == "." else text

# Method to only take out first sentence
def get_first_sentance(text):
    index = text.strip().find(". ")
    if(index != -1):
        text = text[:index].strip()
    return remove_dot_from_end(text.strip())

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

def change_the_format(notebook, notebook_name):
    dataset_as_list = []
    for pair in notebook:
        datapoint = {}
        datapoint["notebook"] = notebook_name
        datapoint["code"] = notebook[pair]["code"]
        datapoint["code_comment_tokens"] = notebook[pair]["code_comment"]
        datapoint["docstring"] = get_first_sentance(notebook[pair]["processed_documentation"].strip())
        datapoint["docstring_tokens"] = word_tokenize(datapoint["docstring"])
        if len(datapoint["docstring"].strip()) != 0 and len(" ".join(datapoint["code"]).strip()) != 0:
            
            # Condition to ensure code and documentation quality
            if (len(datapoint["docstring_tokens"]) >= 3 and len(datapoint["docstring_tokens"]) <= 13) and (len(notebook[pair]["code"]) >= 3 and len(tokenize_code(datapoint["code"])) < 100):
                if "".join(re.split(' |,|\.', datapoint["docstring"])).isalpha():
                    dataset_as_list.append(datapoint)
    return dataset_as_list

def main():
    dataset_file = open(f"/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment/dataset.json", "r") 
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

    dataset_folder = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment"
    with open(f"{dataset_folder}/train_dataset.json", 'w') as f:
        for item in train_dataset:
            f.write(json.dumps(item) + "\n")
    with open(f"{dataset_folder}/test_dataset.json", 'w') as f:
        for item in test_dataset:
            f.write(json.dumps(item) + "\n")
    with open(f"{dataset_folder}/valid_dataset.json", 'w') as f:
        for item in valid_dataset:
            f.write(json.dumps(item) + "\n")

main()