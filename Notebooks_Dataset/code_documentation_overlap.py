import os
import json
from io import BytesIO
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

nltk.download('stopwords')
from nltk.corpus import stopwords

# Code to tokenize python code
def tokenize_code(code_lines):
    code_as_string = " ".join(code_lines)
    tokenized_code = tokenize(BytesIO(code_as_string.encode('utf-8')).readline)
    code_tokens = []
    unnecessary_tokens = ["\n", "", "utf-8"]
    try:
        for _, tokval, _, _, _ in tokenized_code:
            if tokval not in unnecessary_tokens:
                code_tokens.append(tokval)
    except:
        return []
    return code_tokens

def check_overlap_from_json(file_as_json):
    cells_with_overlap, cells_without_overlap = 0,0
    total_cells, total_overlap_count = 0,0
    overlap_frequency, overlap_count_for_each_cell = {}, {0:0}
    show_examples = 5
    for notebook_name in file_as_json:
        for pair in file_as_json[notebook_name]:
            
            code_tokens = tokenize_code(file_as_json[notebook_name][pair]["code"])
            code_tokens = [token.lower() for token in code_tokens if(len(token) > 1 and token.isascii() and token not in stopwords.words('english'))]
            
            documentation_tokens = word_tokenize(file_as_json[notebook_name][pair]["processed_documentation"].strip())
            documentation_tokens = [token.lower() for token in documentation_tokens if(len(token) > 1 and token.isascii() and token not in stopwords.words('english'))]
            
            if len(documentation_tokens) != 0 and len(code_tokens) != 0:
                total_cells += 1
                overlapped_tokens = set(documentation_tokens).intersection(set(code_tokens))
                if(len(overlapped_tokens) != 0):
                    cells_with_overlap += 1
                    total_overlap_count += len(overlapped_tokens)
                    for token in list(overlapped_tokens):
                        try:
                            overlap_frequency[token] += 1
                        except:
                            overlap_frequency[token] = 1
                    try:
                        overlap_count_for_each_cell[len(overlapped_tokens)] += 1
                    except:
                        overlap_count_for_each_cell[len(overlapped_tokens)] = 1
                else:
                    cells_without_overlap += 1
                    overlap_count_for_each_cell[0] += 1
                    if(show_examples > 0):
                        print("\nExample: ", show_examples)
                        print("\n", file_as_json[notebook_name][pair]["code"])
                        print("\n", file_as_json[notebook_name][pair]["processed_documentation"].strip())
                        print("\n=============================================================================")
                        show_examples -= 1
    
    overlap_count_for_each_cell = {k: v for k, v in sorted(overlap_count_for_each_cell.items(), key=lambda item: item[1], reverse=True)}
    overlap_frequency = {k: v for k, v in sorted(overlap_frequency.items(), key=lambda item: item[1], reverse=True)}
    print("\nTotal cells with overlap: {} and without overlap: {}".format(cells_with_overlap, cells_without_overlap))
    print("\nAvergae overlap per cell: {}".format(total_overlap_count/total_cells))   
    print("\nOverlap count for each cell: {}".format(overlap_count_for_each_cell))    
    print("\nOverlap frequency for tokens: {}".format(overlap_frequency)) 

def check_overlap_from_jsonl(file_as_jsonl):
    pass

def main():
    dataset_file = open(f"/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/notebooks_with_atleast_100_upvotes/without_summarization/dataset.json", "r") 
    file_type = "JSON"
    if(file_type == "JSON"):
        file_as_json = json.loads(dataset_file.read())
        check_overlap_from_json(file_as_json)
    elif(file_type == "JSONL"):
        file_as_jsonl = json.loads(dataset_file.read())
        check_overlap_from_jsonl(file_as_jsonl)

main()