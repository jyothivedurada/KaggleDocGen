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

SUMMARIZATION_ERROR = 0

# Code to summarize using Spacy
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
def bart_summarize(text, num_beams = 10, length_penalty = 2.0, max_length = 30, min_length = 3, no_repeat_ngram_size = 3):
    #torch_device = 'cuda' if torch.cuda.is_available() else "cpu"
    torch_device = "cpu"
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    text = text.replace('\n','')
    text_input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(text_input_ids, num_beams=int(num_beams), length_penalty=float(length_penalty), max_length=int(max_length), min_length=int(min_length), no_repeat_ngram_size=int(no_repeat_ngram_size))          
    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary_txt

def summarize_text(text):
    summarized_by_spacy = summarize_document(text)
    summarized_by_bart = bart_summarize(text)
    print("\n\nOriginal: ", text)
    print("\n\nSpacy output: ", summarized_by_spacy)
    print("\n\nBART output: ", summarized_by_bart)
    
text = "Get back to building a CNN using Keras. Much better frameworks then others. You will enjoy for sure."
summarize_text(text)