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

# Extract raw code and documentation pairs for each notebook file
def extract_code_doc_from_notebook(filename, file_as_json, raw_dataset):
    cells = file_as_json["cells"]
    raw_dataset[filename] = {}
    pair_number = 1
    for i in range(1, len(cells), 1):
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
    comments,cleaned_code = [],[]

    # As sometimes the "code" is a single string and sometimes it's list
    if(isinstance(code_lines, list)):
        code_lines = "".join(code_lines)

    # Remove multiline comment(It can handle only one multiline comment per code-cell)
    code_lines = code_lines.replace('\'\'\'', "\"\"\"")
    indexes_for_multiline_comments = [i for i in range(len(code_lines)) if code_lines.startswith("\"\"\"", i)]
    if(len(indexes_for_multiline_comments) >= 2):
        comments.append((code_lines[indexes_for_multiline_comments[0] + 3:indexes_for_multiline_comments[1]]).strip())
        code_lines = code_lines[:indexes_for_multiline_comments[0]] + code_lines[indexes_for_multiline_comments[1] + 3:]
    code_lines = code_lines.split("\n")
    
    for i in range(0, len(code_lines), 1):
        code_lines[i] = code_lines[i].replace("\n", '').replace("\r", '')
        if(code_lines[i].strip().startswith("%")):
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
def preprocess_documentation(documentation_lines, comment):

    # As sometimes the "code" is a single string and sometimes it's list
    if(isinstance(documentation_lines, str)):
        documentation_lines = documentation_lines.strip().split("\n")

    # Clean the documentation
    original_document = copy.deepcopy(documentation_lines)
    documentation_lines = clean_documentation_until_change(documentation_lines)

    # Summarize document/comment of longer length than 10
    summarized_document = ""
    documentation = (" ".join(documentation_lines)).strip()

    if(len(documentation) == 0):
        comment = clean_documentation_until_change(comment)
        comment = ". ".join(comment)
        if(len(comment.split(" ")) <= 10):
            summarized_document = comment
        else:
            summarized_document = summarize_document(comment)
    elif(len(documentation.split(" ")) > 10):
        summarized_document = summarize_document(documentation)
    else:
        summarized_document = documentation

    return original_document, summarized_document

def extract_tokens_from_code(code_lines):
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

    important_tokens = []
    for token in code_tokens:
        if(len(token) > 2):
            important_tokens.append(token)
    return important_tokens

# Apply all the cleaning/preprocessing steps over code/documentation
def data_cleaning(raw_dataset):
    pair_number = 1
    print("\nCleaning code-doc pairs....")
    for filename in tqdm.tqdm(raw_dataset):
        #print("\nProcessing file: {}".format(filename))
        for pair in raw_dataset[filename]:

            # Clean the code and extract comments
            comment, cleaned_code = preprocess_code(raw_dataset[filename][pair]["code"])
            raw_dataset[filename][pair]["code"] = cleaned_code
            raw_dataset[filename][pair]["comment"] = comment

            # Clean the documentation
            original_documentation, processed_documentation = preprocess_documentation(raw_dataset[filename][pair]["documentation"], \
                                                                                       raw_dataset[filename][pair]["comment"])
            raw_dataset[filename][pair]["documentation"] = original_documentation
            raw_dataset[filename][pair]["processed_documentation"] = processed_documentation

            #print("\npair {} processed".format(pair_number))
            pair_number += 1
    return raw_dataset

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
            dataset[filename][pair]["processed_documentation"]])
    df = pd.DataFrame(dataset_as_list, columns=["notebook_name", "code", "comment", "documentation", "processed_documentation"])
    return df

def main():

    # Directory of the notebooks
    path = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes"

    # Get the file names
    filenames = [f for f in os.listdir(path) if f.endswith(".ipynb")]

    # Extract code-documentation pairs from notebooks
    print("\nExtracting code-doc pairs from notebooks....")
    raw_dataset = {}
    JSONDecodeError = 0
    for f in tqdm.tqdm(filenames):
        file = open(f"{path}/{f}", "r")
        try:
            file_as_json = json.loads(file.read())
            raw_dataset = extract_code_doc_from_notebook(f, file_as_json, raw_dataset)
        except json.decoder.JSONDecodeError:
            JSONDecodeError += 1
    print("\ndecoding error: ", JSONDecodeError)
    print("\nTotal number of notebooks: {} and pairs: {}".format(len(filenames), count_pairs(raw_dataset)))

    # Clean the code and documentation
    raw_dataset = data_cleaning(raw_dataset)

    # Convert to dataframe and save as csv
    dataset_as_dataframe = convert_to_dataframe(raw_dataset)
    print("\nShape of the dataframe: ", dataset_as_dataframe.shape)
    dataset_as_dataframe.to_csv("/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/dataset.csv")

    # Convert to json file and save
    dataset_as_json = json.dumps(raw_dataset, indent=4)
    with open("/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/dataset.json", "w") as outfile:
        outfile.write(dataset_as_json)

main()
