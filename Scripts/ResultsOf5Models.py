import rouge
import tqdm
import json
import pandas as pd
import sys
import os

output_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/CodeT5_CodeBERT_Predictions.csv"

# For only code - cleaned markdown - CodeBERT
dataset_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_dataset.jsonl"
pred_file_location_1 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_1.output"
dataset_file_1 = list(open(dataset_file_location_1, "r"))
pred_file_1 = open(pred_file_location_1, "r").readlines()

# For only code - cleaned and summarized markdown - CodeBERT
dataset_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/test_dataset.jsonl"
pred_file_location_2 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/test_1.output"
dataset_file_2 = list(open(dataset_file_location_2, "r"))
pred_file_2 = open(pred_file_location_2, "r").readlines()

# For english code tokens - cleaned and summarized markdown - CodeBERT
dataset_file_location_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm/test_dataset.jsonl"
pred_file_location_3 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm/test_1.output"
dataset_file_3 = list(open(dataset_file_location_3, "r"))
pred_file_3 = open(pred_file_location_3, "r").readlines()

# For code + comment - cleaned and summarized markdown - CodeBERT
dataset_file_location_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment/test_dataset.jsonl"
pred_file_location_4 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-comment/test_1.output"
dataset_file_4 = list(open(dataset_file_location_4, "r"))
pred_file_4 = open(pred_file_location_4, "r").readlines()

# For split code - cleaned and summarized markdown and comments - CodeBERT
dataset_file_location_5 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl"
pred_file_location_5 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_1.output"
dataset_file_5 = list(open(dataset_file_location_5, "r"))
pred_file_5 = open(pred_file_location_5, "r").readlines()

# For only code - cleaned markdown - CodeT5
dataset_file_location_6 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2/test_dataset.jsonl"
pred_file_location_6 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/CM/test_best-bleu.output"
dataset_file_6 = list(open(dataset_file_location_6, "r"))
pred_file_6 = open(pred_file_location_6, "r").readlines()

# For only code - cleaned and summarized markdown - CodeT5
dataset_file_location_7 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only/test_dataset.jsonl"
pred_file_location_7 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/CSM/test_best-bleu.output"
dataset_file_7 = list(open(dataset_file_location_7, "r"))
pred_file_7 = open(pred_file_location_7, "r").readlines()

# For english code tokens - cleaned and summarized markdown - CodeT5
dataset_file_location_8 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm/test_dataset.jsonl"
pred_file_location_8 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/ECSM/test_best-bleu.output"
dataset_file_8 = list(open(dataset_file_location_8, "r"))
pred_file_8 = open(pred_file_location_8, "r").readlines()

# For split code - cleaned and summarized markdown and comments - CodeT5
dataset_file_location_9 = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl"
pred_file_location_9 = "/home/cs19btech11056/cs21mtech12001-Tamal/CodeT5/Predictions/SCSCM(todo-18)/test_best-bleu.output"
dataset_file_9 = list(open(dataset_file_location_9, "r"))
pred_file_9 = open(pred_file_location_9, "r").readlines()

pred_files = [pred_file_1, pred_file_2, pred_file_3, pred_file_4, pred_file_5, 
              pred_file_6, pred_file_7, pred_file_8, pred_file_9]
dataset_files = [dataset_file_1, dataset_file_2, dataset_file_3, dataset_file_4, dataset_file_5, 
                 dataset_file_6, dataset_file_7, dataset_file_8, dataset_file_9]
model_descriptions = ["For only code - cleaned markdown - CodeBERT",
                      "For only code - cleaned and summarized markdown - CodeBERT",
                      "For english code tokens - cleaned and summarized markdown - CodeBERT",
                      "For code + comment - cleaned and summarized markdown - CodeBERT",
                      "For split code - cleaned and summarized markdown and comments - CodeBERT",
                      "For only code - cleaned markdown - CodeT5",
                      "For only code - cleaned and summarized markdown - CodeT5",
                      "For english code tokens - cleaned and summarized markdown - CodeT5",
                      "For split code - cleaned and summarized markdown and comments - CodeT5"]

line_number_1 = 0
dataset_as_list = []
for datapoint_str_1 in tqdm.tqdm(dataset_file_1):
    
    predicted_texts, summarized_gt = [], "NA"
    datapoint_1 = json.loads(datapoint_str_1)
    predicted_text_1 = pred_file_1[line_number_1].split("\t")[1].strip()
    predicted_texts.append(predicted_text_1)
    
    for i in range(1,9):
        dataset_file = dataset_files[i]
        line_number = 0
        predicted_text = -1
        for datapoint_str in dataset_file:
            datapoint = json.loads(datapoint_str)
    
            if(" ".join(datapoint_1["code"]) == " ".join(datapoint["code"])):
                predicted_text = pred_files[i][line_number].split("\t")[1].strip()
                if(i==1):
                    summarized_gt = datapoint["docstring"]
                break
            line_number += 1
            
        if(predicted_text != -1):
            predicted_texts.append(predicted_text)
        else:
            predicted_texts.append("NA")
            
    dataset_as_list.append([datapoint_1["notebook"],
                            datapoint_1["code"],
                            datapoint_1["docstring"], 
                            summarized_gt] + predicted_texts)
    
    line_number_1 += 1
    
df = pd.DataFrame(dataset_as_list, columns=["notebook_name", 
                                            "code", 
                                            "raw documentation",
                                            "summarized documentation"] + model_descriptions)
df.to_csv(output_file)