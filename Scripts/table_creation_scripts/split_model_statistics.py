import json
import pandas as pd
import tqdm

if __name__ == "__main__":
    
    dataset_file_path = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/dataset/splitted_data/by-ast-and-comments/positives-first/test.jsonl"
    dataset_file = list(open(dataset_file_path, "r"))
    
    set_of_notebooks, class_0, class_1 = set([]), 0, 0
    for datapoint_str in tqdm.tqdm(dataset_file):
        datapoint = json.loads(datapoint_str)
        set_of_notebooks.add(datapoint["notebook"])
        if(datapoint["label"] == 0):
            class_0 += 1
        else:
            class_1 += 1
            
    print("\n\nNumber of notebooks: ", len(set_of_notebooks))
    print("\n\nNumber of datapoints: ", len(dataset_file))
    print("\n\nCount of class 0: ", class_0)
    print("\n\nCount of class 1: ", class_1)
    