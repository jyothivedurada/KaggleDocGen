import json
import pandas as pd
import tqdm

if __name__ == "__main__":
    
    dataset_file_path = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl"
    dataset_file = list(open(dataset_file_path, "r"))
    
    set_of_notebooks, code_doc_pairs_count, doc_tokens_count_list, code_tokens_count_list = set([]), 0, [], []
    for datapoint_str in tqdm.tqdm(dataset_file):
        datapoint = json.loads(datapoint_str)
        set_of_notebooks.add(datapoint["notebook"])
        code_doc_pairs_count += 1
        doc_tokens_count_list.append(len(datapoint["docstring_tokens"]))
        code_tokens_count_list.append(len(datapoint["code_tokens"]))
        
    doc_tokens_count_list = pd.Series(doc_tokens_count_list)
    code_tokens_count_list = pd.Series(code_tokens_count_list)
    print("\n\nNumber of notebooks: ", len(set_of_notebooks))
    print("\n\nNumber of code-doc pairs: ", code_doc_pairs_count)
    print("\n\nDocumentation stats: \n\n", doc_tokens_count_list.describe())
    print("\n\nCode stats: \n\n", code_tokens_count_list.describe())
    