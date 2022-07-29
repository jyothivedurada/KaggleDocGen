import os
import json
from io import BytesIO
import pandas as pd

# Count total number of pairs in the dataset
def count_pairs(dataset):
    count_pairs = 0
    for notebook_name in dataset.keys():
        count_pairs += len(list(dataset[notebook_name].keys()))
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
    folder_path = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization"
    dataset_file = open(f"{folder_path}/dataset_{0}.json", "r") 
    final_dataset = json.loads(dataset_file.read())
    for i in range(1,32):
        dataset_file = open(f"{folder_path}/dataset_{i}.json", "r")
        dataset = json.loads(dataset_file.read())
        final_dataset.update(dataset)
    print("\nNumber of notebooks in final dataset: ", len(final_dataset))
    print("\nNumber of datapoints in final dataset: ", count_pairs(final_dataset))
    
    # Convert to json file and save
    dataset_as_json = json.dumps(final_dataset, indent=4)
    with open(f"{folder_path}/dataset.json", "w") as outfile:
        outfile.write(dataset_as_json)
        
    # Convert to dataframe and save as csv
    dataset_as_dataframe = convert_to_dataframe(final_dataset)
    print("\nShape of the dataframe: ", dataset_as_dataframe.shape)
    dataset_as_dataframe.to_csv(f"{folder_path}/dataset.csv")
    

main()