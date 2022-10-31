import os
import sys
import tqdm
import json
import main
import predict
import combine
import split_cell_by_pure_recursion

# Method to make prediction for each data-point
def make_prediction_for_one_datapoint(code_lines, minimum_length_of_structures):
    clusters = split_cell_by_pure_recursion.split_code_cell(code_lines, minimum_length_of_structures)
    print("\nClusters: ", clusters)
    predictions = predict.generate_doc(clusters)
    output = combine.combine_predictions(predictions)
    return output

# Main
if __name__ == "__main__":
    
    # test_dataset_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl"
    # test_dataset_file = list(open(test_dataset_file, "r"))
    # output_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/pipeline/output_predictions/predictions.txt"
    # error_file = "/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/pipeline/output_predictions/error.txt"
    # with open(output_file, 'w') as f1, open(error_file, 'w') as f2:
    #     for datapoint_str in tqdm.tqdm(test_dataset_file):
    #         try:
    #             datapoint = json.loads(datapoint_str)
    #             output = make_prediction_for_one_datapoint(datapoint["code"], 10)
    #             f1.write(", ".join(output) + "\n")
    #         except Exception as e:
    #             f1.write("\n")
    #             if "line 1" not in str(e):
    #                 f2.write("\nCode: " + "\n".join(datapoint["code"]) + "\n")
    #                 f2.write("\n" + str(e) + "\n")
    #                 f2.write("\n=============================================================================================\n")
    
    code_lines =  ['n_classes = 12  ', 
                    'data, labels = make_classification(n_samples=2000, n_features=100, ', 
                    '                                   n_informative=50, n_classes=n_classes, ', 
                    '                                   random_state=random_state)', 
                    '', 
                    'X, X_test, y, y_test = train_test_split(data, labels, test_size=0.2, ', 
                    '                                        random_state=random_state)', 
                    '    ', 
                    'X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, ', 
                    '                                                      random_state=random_state)', 
                    '', 
                    "print('Data shape:')", 
                    "print('X_train: %s, X_valid: %s, X_test: %s \\n' %(X_train.shape, X_valid.shape, ", 
                    '                                                  X_test.shape))', 
                    '    ']

    print("\n\n")
    for o in code_lines:
        print(o)

    output = make_prediction_for_one_datapoint(code_lines, 10)
    print("\n\n")
    for o in output:
        print(o)