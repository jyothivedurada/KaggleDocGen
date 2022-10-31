
from cv2 import line
import rouge
import tqdm
import json
import pandas as pd
import CalculateBLEUScore3
import CalculateROUGEScore
import sys

from bert_score import BERTScorer

from bleurt import score

# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

bert_scorer = BERTScorer(lang = "en", model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline = True)

bleurt_checkpoint = "/home/cs19btech11056/cs21mtech12001-Tamal/BLEURT/BLEURT-20"
bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

dataset_file_location = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/notebooks_dataset/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/old/test_dataset.jsonl"
pred_file_location = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/new_model_on_old_data/test_1.output"
output_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/longer_code_snippets_new_model_old_data_90.csv"
dataset_file = list(open(dataset_file_location, "r"))
pred_file = open(pred_file_location, "r").readlines()
minimum_no_of_lines = 30
minimum_no_of_tokens = 90

line_number = 0
output_as_list = []
for datapoint_str in tqdm.tqdm(dataset_file):
    
    datapoint = json.loads(datapoint_str)
    predicted_text = pred_file[line_number].split("\t")[1].strip()
    
    if len(datapoint["code_tokens"]) >= minimum_no_of_tokens:
        
        bleu_score = CalculateBLEUScore3.BleuScoreBetween2Texts(datapoint["docstring"].strip(), predicted_text.strip())
        rouge_score = CalculateROUGEScore.RougeScoreBetween2Texts(predicted_text.strip(), datapoint["docstring"].strip())
        bert_score = bert_scorer.score([predicted_text.strip()], [datapoint["docstring"].strip()])
        bert_score = {"P":"{:.3f}".format(bert_score[0].item()), 
                      "R":"{:.3f}".format(bert_score[1].item()), 
                      "F1":"{:.3f}".format(bert_score[2].item())}
        bleurt_score = bleurt_scorer.score(candidates = [predicted_text.strip()], references = [datapoint["docstring"].strip()])
        
        output_as_list.append([datapoint["notebook"],
                               datapoint["code_tokens"],
                               datapoint["docstring"],
                               predicted_text,
                               bleu_score,
                               rouge_score,
                               bert_score,
                               "{:.3f}".format(bleurt_score[0])])
        
    line_number += 1
    
df = pd.DataFrame(output_as_list, columns=["notebook_name", 
                                            "code", 
                                            "doc_string", 
                                            "prediction", 
                                            "BLEU score", 
                                            "ROUGE score", 
                                            "BERT score", 
                                            "BLEURT score"])
df.to_csv(output_file)
    