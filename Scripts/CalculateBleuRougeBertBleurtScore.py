
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

dataset_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/notebooks_dataset/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/test_dataset.jsonl"
pred_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/test_1.output"
output_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/results.csv"
dataset_file_1 = list(open(dataset_file_location_1, "r"))
pred_file_1 = open(pred_file_location_1, "r").readlines()

dataset_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/notebooks_dataset/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/additional_punctuations/test_dataset.jsonl"
pred_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/additional_punctuations/test_1.output"
dataset_file_2 = list(open(dataset_file_location_2, "r"))
pred_file_2 = open(pred_file_location_2, "r").readlines()

line_number_1 = 0
dataset_as_list = []
for datapoint_str_1 in tqdm.tqdm(dataset_file_1):
    
    # Collect for CASE 5
    datapoint_1 = json.loads(datapoint_str_1)
    predicted_text_1 = pred_file_1[line_number_1].split("\t")[1].strip()
    bleu_score_1 = CalculateBLEUScore3.BleuScoreBetween2Texts(datapoint_1["docstring"].strip(), predicted_text_1.strip())
    rouge_score_1 = CalculateROUGEScore.RougeScoreBetween2Texts(predicted_text_1.strip(), datapoint_1["docstring"].strip())
    bert_score_1 = bert_scorer.score([predicted_text_1.strip()], [datapoint_1["docstring"].strip()])
    bert_score_1 = {"P":"{:.3f}".format(bert_score_1[0].item()), 
                    "R":"{:.3f}".format(bert_score_1[1].item()), 
                    "F1":"{:.3f}".format(bert_score_1[2].item())}
    bleurt_score_1 = bleurt_scorer.score(candidates = [predicted_text_1.strip()], references = [datapoint_1["docstring"].strip()])
    
    # Collect for CASE 7
    line_number_2 = 0
    predicted_text_2, bleu_score_2, rouge_score_2, bert_score_2 = -1, -1, -1, -1
    for datapoint_str_2 in dataset_file_2:
        datapoint_2 = json.loads(datapoint_str_2)
    
        if((" ".join(datapoint_1["code_tokens"]) == " ".join(datapoint_2["code_tokens"])) and (datapoint_1["docstring"].strip() == datapoint_2["docstring"].strip())):
    
            predicted_text_2 = pred_file_2[line_number_2].split("\t")[1].strip()
            bleu_score_2 = CalculateBLEUScore3.BleuScoreBetween2Texts(datapoint_2["docstring"].strip(), predicted_text_2.strip())
            rouge_score_2 = CalculateROUGEScore.RougeScoreBetween2Texts(predicted_text_2.strip(), datapoint_2["docstring"].strip())
            bert_score_2 = bert_scorer.score([predicted_text_2.strip()], [datapoint_2["docstring"].strip()])
            bert_score_2 = {"P":"{:.3f}".format(bert_score_2[0].item()), 
                            "R":"{:.3f}".format(bert_score_2[1].item()), 
                            "F1":"{:.3f}".format(bert_score_2[2].item())}
            bleurt_score_2 = bleurt_scorer.score(candidates = [predicted_text_2.strip()], references = [datapoint_2["docstring"].strip()])
            break
        line_number_2 += 1
    
    if(predicted_text_2 == -1):
        print("\nHaven't found case 7 output for line number: {}".format(line_number_1))
        
    dataset_as_list.append([datapoint_1["notebook"],
                            datapoint_1["code_tokens"],
                            datapoint_1["docstring"],
                            predicted_text_1,
                            bleu_score_1,
                            rouge_score_1,
                            bert_score_1,
                            "{:.3f}".format(bleurt_score_1[0]),
                            predicted_text_2,
                            bleu_score_2,
                            rouge_score_2,
                            bert_score_2,
                            "{:.3f}".format(bleurt_score_2[0])])
    line_number_1 += 1

df = pd.DataFrame(dataset_as_list, columns=["notebook_name", "code_tokens", "doc_string", "CASE 5 prediction", "CASE 5 BLEU score", "CASE 5 ROUGE score", "CASE 5 BERT score", "CASE 5 BLEURT score", "CASE 7 prediction", "CASE 7 BLEU score", "CASE 7 ROUGE score", "CASE 7 BERT score", "CASE 7 BLEURT score"])
df.to_csv(output_file)