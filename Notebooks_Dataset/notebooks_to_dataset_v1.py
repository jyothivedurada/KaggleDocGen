from curses import pair_number
import os
import re
import sys
import json
import tqdm
import copy
import pandas as pd
from io import BytesIO
from ntpath import join
from genericpath import isfile
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
	
import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

import logging
logging.basicConfig(level=logging.ERROR)

# Extract raw code and documentation pairs for each notebook file
def extract_code_doc_from_notebook(filename, file_as_json, raw_dataset):
    cells = file_as_json["cells"]
    raw_dataset[filename] = {}
    pair_number = 1
    
    # Starting from 3rd cell as 1st cell(markdown) can explain notebook rather than 2nd code cell 
    for i in range(2, len(cells), 1):
        if(cells[i]["cell_type"] == "code"):
            pair_name = "pair-" + str(pair_number)
            raw_dataset[filename][pair_name] = {}
            raw_dataset[filename][pair_name]["code"] = cells[i]["source"]
            if(cells[i-1]["cell_type"] == "markdown"):
                raw_dataset[filename][pair_name]["documentation"] = cells[i-1]["source"]
            else:
                raw_dataset[filename][pair_name]["documentation"] = []
            pair_number += 1
    return raw_dataset

# Clean code(remove "\n", "\r", magic, comments from code) and extract the inline comments
def preprocess_code(code_lines):
    comments, cleaned_code = [],[]

    # As sometimes the "code" is a single string and sometimes it's list
    if(isinstance(code_lines, list)):
        code_lines = "".join(code_lines)

    # Remove multiline comment(It can handle only one multiline comment per code-cell)
    while(True):
        code_lines = code_lines.replace('\'\'\'', "\"\"\"")
        indexes_for_multiline_comments = [i for i in range(len(code_lines)) if code_lines.startswith("\"\"\"", i)]
        if(len(indexes_for_multiline_comments) >= 2):
            comments.append((code_lines[indexes_for_multiline_comments[0] + 3:indexes_for_multiline_comments[1]]).strip())
            code_lines = code_lines[:indexes_for_multiline_comments[0]] + code_lines[indexes_for_multiline_comments[1] + 3:]
        else:
            break
    code_lines = code_lines.split("\n")
    
    for i in range(0, len(code_lines), 1):
        code_lines[i] = code_lines[i].replace("\n", '').replace("\r", '')
        if(code_lines[i].strip().startswith(("%", "!"))):
            pass
        elif(code_lines[i].strip().startswith("#")):
            code_lines[i] = code_lines[i].replace("#", '').strip()
            comments.append(code_lines[i])
        elif("#" in code_lines[i]):
            index = code_lines[i].find("#")
            #comments.append(code_lines[i][index+1:].strip())
            cleaned_code.append(code_lines[i][:index])
        else:
            cleaned_code.append(code_lines[i])

    return comments, cleaned_code

# Code to summarize the long documents
def summarize_document(document):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)

    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)

    freq_word = Counter(keyword)

    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():  
        freq_word[word] = (freq_word[word]/max_freq)
    freq_word.most_common(5)

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]

    summarized_sentences = nlargest(1, sent_strength, key=sent_strength.get)

    final_sentences = [ w.text for w in summarized_sentences ]
    summary = '. '.join(final_sentences)
    return summary

# Summarize documentation using BART(BERT+GPT)
def bart_summarize(text, num_beams = 4, length_penalty = 2.0, max_length = 50, min_length = 10, no_repeat_ngram_size = 3):
    #torch_device = 'cuda' if torch.cuda.is_available() else "cpu"
    torch_device = "cpu"
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    text = text.replace('\n','')
    text_input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(text_input_ids, num_beams=int(num_beams), length_penalty=float(length_penalty), max_length=int(max_length), min_length=int(min_length), no_repeat_ngram_size=int(no_repeat_ngram_size))          
    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt

# Remove html tags from a string
def remove_html_tags(text):
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Method to remove links
def remove_links(text):
    regex = r"(?P<url>https?://[^\s]+)"
    return re.sub(regex, " ", text)

# Method to trim multiple spaces to one
def trim_multiple_spaces(text):
    regex = r" {2,}"
    return re.sub(regex, " ", text)

# Method to remove unnecessary chars from begining
def clean_the_begining(text):
    res = re.search(r'[a-z]', text, re.I)
    if res is not None:
        return text[res.start():]
    elif len(text)!= 0:
        return ""
    else:
        return text

# Method to only take out first sentence
def get_first_sentance(text):
    index = text.strip().find(". ")
    if(index != -1):
        return text[:index+1].strip()
    else:
        return text.strip()

# Remove <number>. from begining
def remove_section_from_begining(text):
    regex_result = re.search(r'^[0-9]+.', text)
    if(regex_result != None):
        return text[regex_result.end():].strip()
    else:
        return text.strip()
    
# Method to get the header text as documentation from markdown cell
def get_header_as_doc(documentation_lines):
    first_line = ""
    for line in documentation_lines:
        if len(line.strip()) != 0:
            first_line = line.strip()
            break
    if first_line[0] == '#':
        regex_result = re.search(r'(#)\1{0,}', first_line)
        line = line[regex_result.end():].strip()
        if(len(line.split(" ")) >= 3):
            return [line]
        else:
            return documentation_lines
    else:
        return documentation_lines

# Method to clean documentation until there is any change
def clean_documentation_until_change(documentation_lines):
    changed = True
    while(changed):
        changed = False
        for i in range(0, len(documentation_lines), 1):

            initial_length = len(documentation_lines[i])

            # Remove common escape tokens
            documentation_lines[i] = documentation_lines[i].strip().replace("\n", '').replace("\r", '').replace("\t", '')\
                                                               .replace("#", '').replace("\u00b4", '')\
                                                               .replace("*", '').strip()

            # Remove HTML tags
            documentation_lines[i] = remove_html_tags(documentation_lines[i]).strip()

            # Remove links
            documentation_lines[i] = remove_links(documentation_lines[i]).strip()

            # Trim multiple spaces
            documentation_lines[i] = trim_multiple_spaces(documentation_lines[i]).strip()

            # Remove <number>. from begining
            documentation_lines[i] = remove_section_from_begining(documentation_lines[i]).strip()

            # Remove unnecessary chars from begining
            documentation_lines[i] = clean_the_begining(documentation_lines[i]).strip()

            if initial_length != len(documentation_lines[i]):
                changed = True

    # Only consider those lines that are not empty
    empty_removed = [doc for doc in documentation_lines if len(doc) != 0]

    return empty_removed

# Code to pre-process the documentation
def preprocess_documentation(documentation_lines):

    # As sometimes the "code" is a single string and sometimes it's list
    if(isinstance(documentation_lines, str)):
        documentation_lines = documentation_lines.strip().split("\n")

    # Copy the original documentation
    original_document = copy.deepcopy(documentation_lines)
    
    # If Header is >= 3 length, use it as doc
    documentation_lines = get_header_as_doc(documentation_lines)
    
    # Clean the documentation
    documentation_lines = clean_documentation_until_change(documentation_lines)

    # Summarize document/comment of longer length than 13
    summarized_document = ""
    documentation = (" ".join(documentation_lines)).strip()
    if(len(documentation.split(" ")) > 13):
        summarized_document = summarize_document(documentation).strip()
    else:
        summarized_document = documentation

    return original_document, documentation, summarized_document

# Take out code and comment pairs
def get_code_comment_pairs(code_lines):
    multiline_comments = []

    # As sometimes the "code" is a single string and sometimes it's list
    if(isinstance(code_lines, list)):
        code_lines = "".join(code_lines)

    # Remove multiline comments
    while(True):
        code_lines = code_lines.replace('\'\'\'', "\"\"\"")
        indexes_for_multiline_comments = [i for i in range(len(code_lines)) if code_lines.startswith("\"\"\"", i)]
        if(len(indexes_for_multiline_comments) >= 2):
            multiline_comments.append((code_lines[indexes_for_multiline_comments[0] + 3:indexes_for_multiline_comments[1]]).strip())
            code_lines = code_lines[:indexes_for_multiline_comments[0]] + code_lines[indexes_for_multiline_comments[1] + 3:]
        else:
            break
    code_lines = code_lines.split("\n")
    
    # Label "code" and "comment" lines
    processed_code_lines = []
    for i in range(0, len(code_lines), 1):
        code_lines[i] = code_lines[i].replace("\n", '').replace("\r", '')
        if(code_lines[i].strip().startswith(("%", "!"))):
            pass
        elif(code_lines[i].strip().startswith("#")):
            comment = code_lines[i].replace("#", '').strip()
            processed_code_lines.append("comment:" + comment)
        elif("#" in code_lines[i]):
            index = code_lines[i].find("#")
            #comments.append(code_lines[i][index+1:].strip())
            processed_code_lines.append("code:" + code_lines[i][:index])
        elif(len(code_lines[i].strip()) != 0):
            processed_code_lines.append("code:" + code_lines[i])
            
    # Collect code-comment pairs
    code_comment_pairs = dict({})
    pair_number = 0
    for i in range(len(processed_code_lines)):
        if(processed_code_lines[i].startswith("comment")):
            pair_number += 1
            code_comment_pairs[pair_number] = dict({})
            code_comment_pairs[pair_number]["comment"] = processed_code_lines[i][8:].strip()
            code_comment_pairs[pair_number]["code"] = []
        else:
            code_comment_pairs[pair_number]["code"].append(processed_code_lines[i][5:])

    return code_comment_pairs

# Apply all the cleaning/preprocessing steps over code/documentation
def data_cleaning(raw_dataset):
    processesd_dataset = dict({})
    print("\nCleaning code-doc pairs....")
    for filename in tqdm.tqdm(raw_dataset):
        pair_number = 1
        processesd_dataset[filename] = dict({})
        for pair in raw_dataset[filename]:
            
            # Clean the documentation
            original_documentation, cleaned_documentation, processed_documentation = preprocess_documentation(raw_dataset[filename][pair]["documentation"])

            # If documnentation from markdown is proper, use that
            # Otherwise extract code-comment pairs
            if(len(processed_documentation) != 0):
                
                # Clean the code and extract comments
                comment, cleaned_code = preprocess_code(raw_dataset[filename][pair]["code"])
                
                # Add to the dataset
                pair_name = "pair-" + str(pair_number)
                processesd_dataset[filename][pair_name] = dict({})
                processesd_dataset[filename][pair_name]["code"] = cleaned_code
                processesd_dataset[filename][pair_name]["comment"] = comment
                processesd_dataset[filename][pair_name]["documentation"] = original_documentation
                processesd_dataset[filename][pair_name]["cleaned_documentation"] = cleaned_documentation
                processesd_dataset[filename][pair_name]["processed_documentation"] = processed_documentation
                pair_number += 1
            else:
                
                # Get code-comment pairs
                code_comment_pairs = get_code_comment_pairs(raw_dataset[filename][pair]["code"])
                for i in range(1, len(code_comment_pairs) + 1, 1):
                    original_documentation = code_comment_pairs[i]["comment"]
                    
                    # Clean the comment
                    cleaned_documentation = "".join(clean_documentation_until_change([code_comment_pairs[i]["comment"]]))
                    
                    # Summarize comment if needed
                    if(len(cleaned_documentation.split(" ")) <= 13):
                        processed_documentation = cleaned_documentation
                    else:
                        processed_documentation = summarize_document(cleaned_documentation)
                        
                    # Add to the dataset
                    pair_name = "pair-" + str(pair_number)
                    processesd_dataset[filename][pair_name] = dict({})
                    processesd_dataset[filename][pair_name]["code"] = code_comment_pairs[i]["code"]
                    processesd_dataset[filename][pair_name]["comment"] = original_documentation
                    processesd_dataset[filename][pair_name]["documentation"] = original_documentation
                    processesd_dataset[filename][pair_name]["cleaned_documentation"] = cleaned_documentation
                    processesd_dataset[filename][pair_name]["processed_documentation"] = processed_documentation
                    pair_number += 1
                    
    return processesd_dataset

# Count total number of pairs in the dataset
def count_pairs(raw_dataset):
    count_pairs = 0
    for f in raw_dataset:
        count_pairs += len(list(raw_dataset[f].keys()))
    return count_pairs

# Convert the dataset from dict to pandas dataframe format
def convert_to_dataframe(dataset):
    dataset_as_list = []
    for filename in list(dataset.keys()):
        for pair in list(dataset[filename].keys()):
            dataset_as_list.append([filename, 
            dataset[filename][pair]["code"], 
            dataset[filename][pair]["comment"], 
            dataset[filename][pair]["documentation"],
            dataset[filename][pair]["cleaned_documentation"],
            dataset[filename][pair]["processed_documentation"]])
    df = pd.DataFrame(dataset_as_list, columns=["notebook_name", "code", "comment", "documentation", "cleaned_documentation", "processed_documentation"])
    return df

def main():

    # Directory of the notebooks
    path = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes"

    # Get the file names
    filenames = [f for f in os.listdir(path) if f.endswith(".ipynb")]

    # Extract code-documentation pairs from notebooks
    print("\nExtracting code-doc pairs from notebooks....")
    batch = int(len(filenames)/32)
    for i in range(0, 32, 1):
        start, end = batch*i, min(batch*(i+1), len(filenames))
        print("\nprocessing from {} to {} in batch {}".format(start, end, i))

        # Extract code-documentation pairs from notebooks
        raw_dataset = {}
        JSONDecodeError = 0
        for f in tqdm.tqdm(filenames[start:end]):
            file = open(f"{path}/{f}", "r")
            try:
                file_as_json = json.loads(file.read())
                raw_dataset = extract_code_doc_from_notebook(f, file_as_json, raw_dataset)
            except json.decoder.JSONDecodeError:
                JSONDecodeError += 1
        print("\ndecoding error: ", JSONDecodeError)
        print("\nTotal number of notebooks: {} and pairs: {}".format(len(filenames[start:end]), count_pairs(raw_dataset)))

        # Clean the code and documentation
        raw_dataset = data_cleaning(raw_dataset)

        # Convert to dataframe and save as csv
        dataset_as_dataframe = convert_to_dataframe(raw_dataset)
        print("\nShape of the dataframe: ", dataset_as_dataframe.shape)
        dataset_as_dataframe.to_csv(f"/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/dataset_{i}.csv")

        # Convert to json file and save
        dataset_as_json = json.dumps(raw_dataset, indent=4)
        with open(f"/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/dataset_{i}.json", "w") as outfile:
            outfile.write(dataset_as_json)

main()
