
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

import torch

# Use CUDA/GPU for BLEURT
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,6,7"

# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

# Load BERT score model
mydevice = torch.device("cpu")
bert_scorer = BERTScorer(lang = "en", model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline = True, device = mydevice)

# Load BLEURT score model
mydevice = torch.device("cuda")
bleurt_checkpoint = "/home/cs19btech11056/cs21mtech12001-Tamal/BLEURT/BLEURT-20"
bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

# Method to calculate BERT score between 2 files
def CalculateBERTscoreBetweenFiles(predictions, gt):
    predictions_as_list = []
    gt_as_list = []
    for row in predictions:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            predictions_as_list.append('') 
        else:
            predictions_as_list.append(cols[1].strip().lower()) 
    
    for row in gt:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            gt_as_list.append('') 
        else:
            gt_as_list.append(cols[1].strip().lower())
    
    bert_scores = bert_scorer.score(predictions_as_list, gt_as_list)
    average_bert_score = (sum(bert_scores[2])/len(bert_scores[2])).item() * 100
    print("\nAverage BERT score: ", "{:.3f}".format(average_bert_score))
    return average_bert_score

# Method to calculate BERT score between 2 files
def CalculateBLEURTscoreBetweenFiles(predictions, gt):
    predictions_as_list = []
    gt_as_list = []
    for row in predictions:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            predictions_as_list.append('') 
        else:
            predictions_as_list.append(cols[1].strip().lower()) 
    
    for row in gt:
        cols = row.strip().split('\t')
        if len(cols) == 1:
            gt_as_list.append('') 
        else:
            gt_as_list.append(cols[1].strip().lower())
    
    bleurt_score = bleurt_scorer.score(candidates = predictions_as_list, references = gt_as_list)
    average_bleurt_score = (sum(bleurt_score)/len(bleurt_score)) * 100
    print("\nAverage BLEURT score: ", "{:.3f}".format(average_bleurt_score))
    return average_bleurt_score

# For only code - cleaned markdown
ref_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_1.gold"
pred_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_1.output"
pred_file_1 = open(pred_file_location_1, "r").readlines()
ref_file_1 = open(ref_file_location_1, "r").readlines()

# For only code - cleaned and summarized markdown
ref_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/test_1.gold"
pred_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/test_1.output"
pred_file_2 = open(pred_file_location_2, "r").readlines()
ref_file_2 = open(ref_file_location_2, "r").readlines()

# For english code tokens - cleaned and summarized markdown
ref_file_location_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm/test_1.gold"
pred_file_location_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm/test_1.output"
pred_file_3 = open(pred_file_location_3, "r").readlines()
ref_file_3 = open(ref_file_location_3, "r").readlines()

# For code + comment - cleaned and summarized markdown
ref_file_location_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment/test_1.gold"
pred_file_location_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment/test_1.output"
pred_file_4 = open(pred_file_location_4, "r").readlines()
ref_file_4 = open(ref_file_location_4, "r").readlines()

# For split code - cleaned and summarized markdown and comments
ref_file_location_5 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_1.gold"
pred_file_location_5 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_1.output"
pred_file_5 = open(pred_file_location_5, "r").readlines()
ref_file_5 = open(ref_file_location_5, "r").readlines()

# For split code - cleaned and summarized markdown and comments - UnixCoder
ref_file_location_6 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test.gold"
pred_file_location_6 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test.output"
pred_file_6 = open(pred_file_location_6, "r").readlines()
ref_file_6 = open(ref_file_location_6, "r").readlines()

# For only code - cleaned and summarized markdown - UnixCoder
ref_file_location_7 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code_with_sm_only/test.gold"
pred_file_location_7 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code_with_sm_only/test.output"
pred_file_7 = open(pred_file_location_7, "r").readlines()
ref_file_7 = open(ref_file_location_7, "r").readlines()

# For only code - cleaned markdown - UnixCoder
ref_file_location_8 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code_with_usm_only_2/test.gold"
pred_file_location_8 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code_with_usm_only_2/test.output"
pred_file_8 = open(pred_file_location_8, "r").readlines()
ref_file_8 = open(ref_file_location_8, "r").readlines()

# For english code tokens - cleaned and summarized markdown - UnixCoder
ref_file_location_9 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english_code_tokens_with_sm/test.gold"
pred_file_location_9 = "/home/cs19btech11056/cs21mtech12001-Tamal/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english_code_tokens_with_sm/test.output"
pred_file_9 = open(pred_file_location_9, "r").readlines()
ref_file_9 = open(ref_file_location_9, "r").readlines()

# For split code - cleaned and summarized markdown and comments - GraphCodeBERT
ref_file_location_10 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/all_constraints/todo-18/test_1.gold"
pred_file_location_10 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/all_constraints/todo-18/test_1.output"
pred_file_10 = open(pred_file_location_10, "r").readlines()
ref_file_10 = open(ref_file_location_10, "r").readlines()

# For only code - cleaned markdown - GraphCodeBERT
ref_file_location_11 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/without_summarization/code_with_usm_only_2/test_1.gold"
pred_file_location_11 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/without_summarization/code_with_usm_only_2/test_1.output"
pred_file_11 = open(pred_file_location_11, "r").readlines()
ref_file_11 = open(ref_file_location_11, "r").readlines()

# For only code - cleaned and summarized markdown - GraphCodeBERT
ref_file_location_12 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/code-with-sm-only/test_1.gold"
pred_file_location_12 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/code-with-sm-only/test_1.output"
pred_file_12 = open(pred_file_location_12, "r").readlines()
ref_file_12 = open(ref_file_location_12, "r").readlines()

# For english code tokens - cleaned and summarized markdown - GraphCodeBERT
ref_file_location_13 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/english-code-tokens-only-2/test_1.gold"
pred_file_location_13 = "/home/cs19btech11056/cs21mtech12001-Tamal/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/english-code-tokens-only-2/test_1.output"
pred_file_13 = open(pred_file_location_13, "r").readlines()
ref_file_13 = open(ref_file_location_13, "r").readlines()

# For split code - cleaned and summarized markdown and comments - CodeT5
ref_file_location_14 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/SCSCM(todo-18)/test_best-bleu.gold"
pred_file_location_14 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/SCSCM(todo-18)/test_best-bleu.output"
pred_file_14 = open(pred_file_location_14, "r").readlines()
ref_file_14 = open(ref_file_location_14, "r").readlines()

# For only code - cleaned markdown - CodeT5
ref_file_location_15 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/CM/test_best-bleu.gold"
pred_file_location_15 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/CM/test_best-bleu.output"
pred_file_15 = open(pred_file_location_15, "r").readlines()
ref_file_15 = open(ref_file_location_15, "r").readlines()

# For only code - cleaned and summarized markdown - CodeT5
ref_file_location_16 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/CSM/test_best-bleu.gold"
pred_file_location_16 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/CSM/test_best-bleu.output"
pred_file_16 = open(pred_file_location_16, "r").readlines()
ref_file_16 = open(ref_file_location_16, "r").readlines()

# For english code tokens - cleaned and summarized markdown - CodeT5
ref_file_location_17 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/ECSM/test_best-bleu.gold"
pred_file_location_17 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/ECSM/test_best-bleu.output"
pred_file_17 = open(pred_file_location_17, "r").readlines()
ref_file_17 = open(ref_file_location_17, "r").readlines()

# For split code - cleaned and summarized markdown and comments - PLBART
ref_file_location_18 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/SCSCM(todo-18)/test_best-bleu.gold"
pred_file_location_18 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/SCSCM(todo-18)/test_best-bleu.output"
pred_file_18 = open(pred_file_location_18, "r").readlines()
ref_file_18 = open(ref_file_location_18, "r").readlines()

# For only code - cleaned markdown - PLBART
ref_file_location_19 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/CM/test_best-bleu.gold"
pred_file_location_19 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/CM/test_best-bleu.output"
pred_file_19 = open(pred_file_location_19, "r").readlines()
ref_file_19 = open(ref_file_location_19, "r").readlines()

# For only code - cleaned and summarized markdown - PLBART
ref_file_location_20 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/CSM/test_best-bleu.gold"
pred_file_location_20 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/CSM/test_best-bleu.output"
pred_file_20 = open(pred_file_location_20, "r").readlines()
ref_file_20 = open(ref_file_location_20, "r").readlines()

# For english code tokens - cleaned and summarized markdown - PLBART
ref_file_location_21 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/ECSM/test_best-bleu.gold"
pred_file_location_21 = "/home/cs19btech11056/cs21mtech12001-Tamal/PLBART/Predictions/ECSM/test_best-bleu.output"
pred_file_21 = open(pred_file_location_21, "r").readlines()
ref_file_21 = open(ref_file_location_21, "r").readlines()

pred_files = [pred_file_1, pred_file_2, pred_file_3, pred_file_4, pred_file_5, 
              pred_file_6, pred_file_7, pred_file_8, pred_file_9,
              pred_file_10, pred_file_11, pred_file_12, pred_file_13,
              pred_file_14, pred_file_15, pred_file_16, pred_file_17,
              pred_file_18, pred_file_19, pred_file_20, pred_file_21]
ref_files = [ref_file_1, ref_file_2, ref_file_3, ref_file_4, ref_file_5, 
             ref_file_6, ref_file_7, ref_file_8, ref_file_9,
             ref_file_10, ref_file_11, ref_file_12, ref_file_13,
             ref_file_14, ref_file_15, ref_file_16, ref_file_17,
             ref_file_18, ref_file_19, ref_file_20, ref_file_21]
model_descriptions = ["For only code - cleaned markdown",
                      "For only code - cleaned and summarized markdown",
                      "For english code tokens - cleaned and summarized markdown",
                      "For code + comment - cleaned and summarized markdown",
                      "For split code - cleaned and summarized markdown and comments",
                      "For split code - cleaned and summarized markdown and comments - UnixCoder",
                      "For only code - cleaned and summarized markdown - UnixCoder",
                      "For only code - cleaned markdown - UnixCoder",
                      "For english code tokens - cleaned and summarized markdown - UnixCoder",
                      "For split code - cleaned and summarized markdown and comments - GraphCodeBERT",
                      "For only code - cleaned markdown - GraphCodeBERT",
                      "For only code - cleaned and summarized markdown - GraphCodeBERT",
                      "For english code tokens - cleaned and summarized markdown - GraphCodeBERT",
                      "For split code - cleaned and summarized markdown and comments - CodeT5",
                      "For only code - cleaned markdown - CodeT5",
                      "For only code - cleaned and summarized markdown - CodeT5",
                      "For english code tokens - cleaned and summarized markdown - CodeT5",
                      "For split code - cleaned and summarized markdown and comments - PLBART",
                      "For only code - cleaned markdown - PLBART",
                      "For only code - cleaned and summarized markdown - PLBART",
                      "For english code tokens - cleaned and summarized markdown - PLBART"]

for i in range(len(model_descriptions)):
    if i not in [17, 18, 19, 20]:
        continue
    print("\nModel/Representation: ", model_descriptions[i])
    CalculateROUGEScore.RougeScoreBetween2Files(pred_files[i], ref_files[i])
    CalculateBLEUScore3.BleuScoreBetween2Files(pred_files[i], ref_files[i])
    CalculateBERTscoreBetweenFiles(pred_files[i], ref_files[i])
    # CalculateBLEURTscoreBetweenFiles(pred_files[i], ref_files[i])


