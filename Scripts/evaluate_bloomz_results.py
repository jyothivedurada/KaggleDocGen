
import rouge
import tqdm
import json
import pandas as pd
import CalculateBLEUScore3
import CalculateROUGEScore
import sys
import os
import glob

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
# mydevice = torch.device("cuda")
# bleurt_checkpoint = "/home/cs19btech11056/cs21mtech12001-Tamal/BLEURT/BLEURT-20"
# bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)

def CalculateBERTscoreBetweenLists(predictions_as_list, gt_as_list):
    bert_scores = bert_scorer.score(predictions_as_list, gt_as_list)
    average_bert_score = (sum(bert_scores[2])/len(bert_scores[2])).item() * 100
    print("\nAverage BERT score: ", "{:.3f}".format(average_bert_score))
    return average_bert_score

# def CalculateBLEURTscoreBetweenFiles(predictions_as_list, gt_as_list):
#     bleurt_score = bleurt_scorer.score(candidates = predictions_as_list, references = gt_as_list)
#     average_bleurt_score = (sum(bleurt_score)/len(bleurt_score)) * 100
#     print("\nAverage BLEURT score: ", "{:.3f}".format(average_bleurt_score))
#     return average_bleurt_score

for file_path in glob.glob("/home/siddharthsa/cs21mtech12001-Tamal/BLOOMZ/results-560M-with-csn/*.csv"):
    if "560m-ecsm" not in file_path:
        continue
    predictions, references = [], []
    results = pd.read_csv(file_path)
    for index, row in results.iterrows():
        prediction, reference = row["predicted_text"], row["original_label"].strip()
        all_predictions = [pred.strip() for pred in prediction.strip().split("Documentation:")[1:]]
        prediction = all_predictions[0].replace(".", "").strip()
        predictions.append(prediction)
        references.append(reference)
    
    results["processed_prediction"] = predictions
    processed_file_name = "/".join(file_path.split("/")[:-1] + [file_path.split("/")[-1].strip(".csv") + "-processed.csv"])
    results.to_csv(processed_file_name)
    
    print("\nModel/Representation: ", file_path.split("/")[-1].strip())
    CalculateROUGEScore.RougeScoreBetween2Lists(predictions, references)
    CalculateBLEUScore3.BleuScoreBetween2Lists(predictions, references)
    CalculateBERTscoreBetweenLists(predictions, references)
    # CalculateBLEURTscoreBetweenFiles(pred_files[i], ref_files[i])


