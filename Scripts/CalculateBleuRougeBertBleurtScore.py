
import rouge
import tqdm
import json
import pandas as pd
import CalculateBLEUScore3
import CalculateROUGEScore
import sys
import os

from bert_score import BERTScorer

from bleurt import score

# Use CUDA/GPU for BLEURT
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

bert_scorer = BERTScorer(lang = "en", model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline = True)

bleurt_checkpoint = "/home/cs19btech11056/cs21mtech12001-Tamal/BLEURT/BLEURT-20"
bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

# For CASE 5
dataset_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/old/test_dataset.jsonl"
pred_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/old/test_1.output"
output_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/results.csv"
dataset_file_1 = list(open(dataset_file_location_1, "r"))
pred_file_1 = open(pred_file_location_1, "r").readlines()

# For CASE 7
dataset_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/additional_punctuations/test_dataset.jsonl"
pred_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/additional_punctuations/test_1.output"
dataset_file_2 = list(open(dataset_file_location_2, "r"))
pred_file_2 = open(pred_file_location_2, "r").readlines()

# For CASE 12
dataset_file_location_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/new(todo-11)/test_dataset.jsonl"
pred_file_location_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/new(todo-11)/test_1.output"
dataset_file_3 = list(open(dataset_file_location_3, "r"))
pred_file_3 = open(pred_file_location_3, "r").readlines()

# For CASE 13
dataset_file_location_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl"
pred_file_location_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_1.output"
dataset_file_4 = list(open(dataset_file_location_4, "r"))
pred_file_4 = open(pred_file_location_4, "r").readlines()

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
    predicted_text_2, bleu_score_2, rouge_score_2, bert_score_2, bleurt_score_2 = -1, -1, -1, -1, [-1]
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
        
    # Collect for CASE 12
    line_number_3 = 0
    predicted_text_3, bleu_score_3, rouge_score_3, bert_score_3, bleurt_score_3 = -1, -1, -1, -1, [-1]
    for datapoint_str_3 in dataset_file_3:
        datapoint_3 = json.loads(datapoint_str_3)
    
        if(" ".join(datapoint_1["code_tokens"]) == " ".join(datapoint_3["code_tokens"])):
    
            predicted_text_3 = pred_file_3[line_number_3].split("\t")[1].strip()
            bleu_score_3 = CalculateBLEUScore3.BleuScoreBetween2Texts(datapoint_3["docstring"].strip(), predicted_text_3.strip())
            rouge_score_3 = CalculateROUGEScore.RougeScoreBetween2Texts(predicted_text_3.strip(), datapoint_3["docstring"].strip())
            bert_score_3 = bert_scorer.score([predicted_text_3.strip()], [datapoint_3["docstring"].strip()])
            bert_score_3 = {"P":"{:.3f}".format(bert_score_3[0].item()), 
                            "R":"{:.3f}".format(bert_score_3[1].item()), 
                            "F1":"{:.3f}".format(bert_score_3[2].item())}
            bleurt_score_3 = bleurt_scorer.score(candidates = [predicted_text_3.strip()], references = [datapoint_3["docstring"].strip()])
            break
        line_number_3 += 1
    
    if(predicted_text_3 == -1):
        print("\nHaven't found case 12 output for line number: {}".format(line_number_1))
        
    # Collect for CASE 13
    line_number_4 = 0
    predicted_text_4, bleu_score_4, rouge_score_4, bert_score_4, bleurt_score_4 = -1, -1, -1, -1, [-1]
    for datapoint_str_4 in dataset_file_4:
        datapoint_4 = json.loads(datapoint_str_4)
    
        if(" ".join(datapoint_1["code_tokens"]) == " ".join(datapoint_4["code_tokens"])):
    
            predicted_text_4 = pred_file_4[line_number_4].split("\t")[1].strip()
            bleu_score_4 = CalculateBLEUScore3.BleuScoreBetween2Texts(datapoint_4["docstring"].strip(), predicted_text_4.strip())
            rouge_score_4 = CalculateROUGEScore.RougeScoreBetween2Texts(predicted_text_4.strip(), datapoint_4["docstring"].strip())
            bert_score_4 = bert_scorer.score([predicted_text_4.strip()], [datapoint_4["docstring"].strip()])
            bert_score_4 = {"P":"{:.3f}".format(bert_score_4[0].item()), 
                            "R":"{:.3f}".format(bert_score_4[1].item()), 
                            "F1":"{:.3f}".format(bert_score_4[2].item())}
            bleurt_score_4 = bleurt_scorer.score(candidates = [predicted_text_4.strip()], references = [datapoint_4["docstring"].strip()])
            break
        line_number_4 += 1
    
    if(predicted_text_4 == -1):
        print("\nHaven't found case 13 output for line number: {}".format(line_number_1))
        
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
                            "{:.3f}".format(bleurt_score_2[0]),
                            predicted_text_3,
                            bleu_score_3,
                            rouge_score_3,
                            bert_score_3,
                            "{:.3f}".format(bleurt_score_3[0]),
                            predicted_text_4,
                            bleu_score_4,
                            rouge_score_4,
                            bert_score_4,
                            "{:.3f}".format(bleurt_score_4[0])])
    line_number_1 += 1

df = pd.DataFrame(dataset_as_list, columns=["notebook_name", 
                                            "code_tokens", 
                                            "doc_string", 
                                            "CASE 5 prediction", 
                                            "CASE 5 BLEU score", 
                                            "CASE 5 ROUGE score", 
                                            "CASE 5 BERT score", 
                                            "CASE 5 BLEURT score", 
                                            "CASE 7 prediction", 
                                            "CASE 7 BLEU score", 
                                            "CASE 7 ROUGE score", 
                                            "CASE 7 BERT score", 
                                            "CASE 7 BLEURT score",
                                            "CASE 12 prediction", 
                                            "CASE 12 BLEU score", 
                                            "CASE 12 ROUGE score", 
                                            "CASE 12 BERT score", 
                                            "CASE 12 BLEURT score",
                                            "CASE 13 prediction", 
                                            "CASE 13 BLEU score", 
                                            "CASE 13 ROUGE score", 
                                            "CASE 13 BERT score", 
                                            "CASE 13 BLEURT score"])
df.to_csv(output_file)