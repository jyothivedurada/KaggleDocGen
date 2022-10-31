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
    spacy_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/todo-18/dataset.json"
    bart_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/dataset.json"
    dataset_file = open(spacy_file, "r") 
    file_as_json = json.loads(dataset_file.read())
    
    op_file = open("/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/table_creation_scripts/spacy_output.txt","w")
    
    for notebook_name in tqdm.tqdm(file_as_json):
        for pair in file_as_json[notebook_name]:
            cleaned_documentation = file_as_json[notebook_name][pair]["cleaned_documentation"].strip()
            processed_documentation = file_as_json[notebook_name][pair]["processed_documentation"].strip()
            if(len(cleaned_documentation.split(" ")) < 20 and len(cleaned_documentation.split(" ")) > 8 and len(cleaned_documentation)-2 >= len(processed_documentation) and "." in cleaned_documentation):
                op_file.write("\nCleaned documentation: " + cleaned_documentation)
                op_file.write("\nProcessed documentation: " + processed_documentation)
                op_file.write("\n===========================================================================================")
                
    op_file.close()
                
main()