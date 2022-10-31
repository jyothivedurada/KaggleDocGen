import os
import json
from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import re
import tqdm

import pandas as pd

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

def main():
    dataset_file = open(f"/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/todo-18/dataset.json", "r") 
    file_as_json = json.loads(dataset_file.read())
    
    code_doc_counts, code_comment_counts = 0, 0
    markdown_doc_length_list, markdown_code_length_list = [], []
    comment_doc_length_list, comment_code_length_list = [], []
    for notebook_name in tqdm.tqdm(file_as_json):
        for pair in file_as_json[notebook_name]:
            if(len(pair.strip().split("-")) == 2):
                tokens_from_code = tokenize_code(file_as_json[notebook_name][pair]["code"])
                tokens_from_doc = word_tokenize(file_as_json[notebook_name][pair]["cleaned_documentation"].strip())
                if(len(tokens_from_doc) >= 3 and len(tokens_from_code) > 0 and len(file_as_json[notebook_name][pair]["code"]) >= 3 and len(tokens_from_code) < 100):
                    code_doc_counts += 1
                    markdown_doc_length_list.append(len(tokens_from_doc))
                    markdown_code_length_list.append(len(tokens_from_code))
                    if(len(tokens_from_code) == 2):
                        print("\nCode: ", file_as_json[notebook_name][pair]["code"])
                        print("\nDocumentation: ", file_as_json[notebook_name][pair]["cleaned_documentation"].strip())
                        print("\n===============================================")
            else:
                tokens_from_code = tokenize_code(file_as_json[notebook_name][pair]["code"])
                tokens_from_doc = word_tokenize(file_as_json[notebook_name][pair]["cleaned_documentation"].strip())
                if(len(tokens_from_doc) >= 3 and len(tokens_from_code) > 0 and len(file_as_json[notebook_name][pair]["code"]) >= 3 and len(tokens_from_code) < 100):
                    code_comment_counts += 1
                    comment_doc_length_list.append(len(tokens_from_doc))
                    comment_code_length_list.append(len(tokens_from_code))
                
    markdown_doc_length_list = pd.Series(markdown_doc_length_list)
    markdown_code_length_list = pd.Series(markdown_code_length_list)
    comment_doc_length_list = pd.Series(comment_doc_length_list)
    comment_code_length_list = pd.Series(comment_code_length_list)
    print("\nTotal code-doc pairs: ", code_doc_counts)
    print("\nTotal code-comment pairs: ", code_comment_counts)
    print("\n\nDoc stats in case of markdown: \n\n", markdown_doc_length_list.describe())
    print("\n\nCode stats in case of markdown: \n\n", markdown_code_length_list.describe())
    print("\n\nDoc stats in case of comments: \n\n", comment_doc_length_list.describe())
    print("\n\nCode stats in case of comments: \n\n", comment_code_length_list.describe())
main()