from curses import pair_number, raw
import os
import re
import sys
import json
import tqdm
import copy
import pandas as pd
from io import BytesIO
from ntpath import join
from genericpath import isfile
import numpy as np

import ast
import astunparse

NUMBER_OF_POSITIVE_EXAMPLES, NUMBER_OF_NEGATIVE_EXAMPLES, NUMBER_OF_POSITIVE_EXAMPLES_AST, NUMBER_OF_NEGATIVE_EXAMPLES_AST = 0, 0, 0, 0
NUMBER_OF_AST_PARSING_ERROR = 0

# Structure elements to cosider
LIST_OF_STRUCTURES = tuple(["def", "for", "while", "if", "class", "with", "try"])
STRUCTURES_TO_EXCLUDE = tuple(["if"])

# Method to collect all child-nodes of a node
class MyVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        childs = []
        for stmt in ast.iter_child_nodes(node):
          if not isinstance(stmt, ast.Name) and \
             not isinstance(stmt, ast.Attribute) and \
             not isinstance(stmt, ast.arguments) and \
             not isinstance(stmt, ast.Compare) :
            child = []
            child.append(astunparse.unparse(stmt))
            child.append(stmt)
            childs.append(child)
        return childs

# Get first line of the sctructure
def get_first_line_of_code(code):
  start_line_of_structure = ""
  for code_line in code.split("\n"):
    if len(code_line.strip()) != 0:
      start_line_of_structure = code_line
      break
  return start_line_of_structure

# Method to break large block of code
def break_code(code, ast_node):
  
  # Get first line of the sctructure
  start_line_of_structure = get_first_line_of_code(code)
  
  # Get code-lines of all childs
  visitor = MyVisitor()
  #print("\nUnparsed code: ", code)
  parsed_code = visitor.generic_visit(ast_node)
  #print("\nParsed code: ", parsed_code)
  
  # Call merge on the codelines of childs
  merged_clusters = merge_code(parsed_code)
  
  # Add starting part of the structure
  if merged_clusters[0][0].strip().startswith(LIST_OF_STRUCTURES):
    merged_clusters.insert(0, [start_line_of_structure])
  else:
    merged_clusters[0].insert(0, start_line_of_structure)
    
  return merged_clusters

# Method to merge code-lines 
def merge_code(parsed_code):
    clusters_after_merging = [[]]
    
    # Recursively break large structures, merge the code-lines
    for code, ast_node in parsed_code:
      start_line_of_structure = get_first_line_of_code(code)
      STRUCTURES_TO_BREAK = tuple(set(LIST_OF_STRUCTURES).difference(set(STRUCTURES_TO_EXCLUDE)))
      if start_line_of_structure.strip().startswith(STRUCTURES_TO_BREAK):
        code_of_structure = break_code(code, ast_node)
        clusters_after_merging.extend(code_of_structure)
      else:
        
        code_lines = [code_line for code_line in code.split("\n") if len(code_line.strip())!= 0]

        # Don't merge if any of the 2 clusters are 2 structures
        if (len(clusters_after_merging[-1]) != 0 and clusters_after_merging[-1][0].strip().startswith(LIST_OF_STRUCTURES)) or \
          code_lines[0].strip().startswith(LIST_OF_STRUCTURES):
          clusters_after_merging.append(code_lines)
        else:
          clusters_after_merging[-1].extend(code_lines)
    
    #print("\nCleaned and merged code: ", clusters_after_merging)
      
    # Remove empty clusters that might be created above
    after_removing_empty = []
    for cluster in clusters_after_merging:
      if len(cluster) != 0:
        after_removing_empty.append(cluster)
      
    #print("\nAfter removing empty: ", after_removing_empty)
    
    return after_removing_empty
        
# Method to split a code-cell
def split_code_cell(code_lines):
    
    # Traverse and take-out elements from "body"(to get larger structures)
    visitor = MyVisitor()
    try:
      clusters = visitor.visit(ast.parse("\n".join(code_lines)))
    except Exception as e:
      raise e

    #print("\nAfter first round of parsing: ", clusters)
    
    # Merge code-lines and break large block of code by recursion
    clusters = merge_code(clusters)
    #print("\nClusters: ", clusters)
    #print("\nNo of clusters: ", len(clusters))
    
    return clusters

# Extract raw code from notebooks
def extract_code_from_notebook(filename, file_as_json, raw_dataset):
    cells = file_as_json["cells"]
    raw_dataset[filename] = {}
    cell_number = 1
    for i in range(0, len(cells), 1):
        if(cells[i]["cell_type"] == "code"):
            code_name = "code-" + str(cell_number)
            raw_dataset[filename][code_name] = {}
            raw_dataset[filename][code_name]["code"] = cells[i]["source"]
            cell_number += 1
    return raw_dataset

# Clean code(remove "\n", "\r", magic, comments from code) and extract the inline comments
def preprocess_code(code_lines):

    global NUMBER_OF_POSITIVE_EXAMPLES, NUMBER_OF_NEGATIVE_EXAMPLES, NUMBER_OF_POSITIVE_EXAMPLES_AST, NUMBER_OF_NEGATIVE_EXAMPLES_AST, NUMBER_OF_AST_PARSING_ERROR
    
    # As sometimes the "code" is a single string and sometimes it's list
    if(isinstance(code_lines, list)):
        code_lines = "".join(code_lines)

    # Convert multi-line comment to single line comment
    while(True):
        code_lines = code_lines.replace('\'\'\'', "\"\"\"")
        indexes_for_multiline_comments = [i for i in range(len(code_lines)) if code_lines.startswith("\"\"\"", i)]
        if(len(indexes_for_multiline_comments) >= 2):
            multiline_comment = (code_lines[indexes_for_multiline_comments[0] + 3:indexes_for_multiline_comments[1]]).strip()
            multiline_comment = multiline_comment.split("\n")
            multiline_comment = [comment.strip() + "." if len(comment.strip()) != 0 and comment.strip()[-1].isalpha() else comment.strip() for comment in multiline_comment]
            multiline_comment = "#" + " ".join(multiline_comment) + "\n"
            code_lines = code_lines[:indexes_for_multiline_comments[0]] + multiline_comment + code_lines[indexes_for_multiline_comments[1] + 3:]
        else:
            break
    code_lines = code_lines.split("\n")
    
    # Get cleaned code and code boundaries
    code_boundaries, cleaned_code = [0], []
    for i in range(0, len(code_lines), 1):
        code_lines[i] = code_lines[i].replace("\n", '').replace("\r", '')
        if(code_lines[i].strip().startswith(("%", "!"))):
            pass
        elif(code_lines[i].strip().startswith("#")):
            code_lines[i] = code_lines[i].replace("#", '').strip()
            code_boundaries.append(len(cleaned_code))
        elif("#" in code_lines[i]):
            index = code_lines[i].find("#")
            if(len(code_lines[i][:index].strip()) != 0):
                cleaned_code.append(code_lines[i][:index])
        else:
            if(len(code_lines[i].strip()) != 0):
                cleaned_code.append(code_lines[i])
           
    negative_examples_to_sample_at_a_time = 2
    positive_negative_examples = []
    positive_negative_next_codelines = []
    
    # Get positive examples using AST based splits
    splitted_code = []
    try:
        splitted_code = split_code_cell(cleaned_code)
        for i in range(1, len(splitted_code)):
            prefix_code = splitted_code[i-1]
            next_codeline = "" if len(splitted_code[i][0].strip()) == 0 else splitted_code[i][0]
        
            # Get positive example
            if len("".join(prefix_code).strip()) != 0 and len(next_codeline.strip()) != 0:
                positive_example = [prefix_code, next_codeline, 1]
                positive_negative_examples.append(positive_example)
                positive_negative_next_codelines.append(next_codeline.strip())
                NUMBER_OF_POSITIVE_EXAMPLES_AST += 1
    except Exception as e:
        NUMBER_OF_AST_PARSING_ERROR += 1
        print("\nAST splitting error: ", e)
        
    # Get positive examples using comment based splits
    for i in range(1, len(code_boundaries)):
        prefix_code = cleaned_code[code_boundaries[i-1] : code_boundaries[i]]
        next_codeline = "" if code_boundaries[i] >= len(cleaned_code) else cleaned_code[code_boundaries[i]]
        is_next_codeline_taken = False
        for line in positive_negative_next_codelines:
            if line.strip().startswith(next_codeline.strip()):
                is_next_codeline_taken = True
        if len("".join(prefix_code).strip()) != 0 and len(next_codeline.strip()) != 0 and not is_next_codeline_taken:
            positive_example = [prefix_code, next_codeline, 1]
            positive_negative_examples.append(positive_example)
            positive_negative_next_codelines.append(next_codeline.strip())
            NUMBER_OF_POSITIVE_EXAMPLES += 1
        
    # Get negative example using AST based splits
    for i in range(0, len(splitted_code)):
        code_split = splitted_code[i]
        for _ in range(negative_examples_to_sample_at_a_time - 1):
            if len(code_split) > 1 and len("".join(code_split).strip()) != 0:
                random_codeline_index = np.random.randint(1, len(splitted_code[i]))
                negative_prefix_code = code_split[: random_codeline_index]
                negative_next_codeline = code_split[random_codeline_index]
                is_next_codeline_taken = False
                for line in positive_negative_next_codelines:
                    if negative_next_codeline.strip().startswith(line.strip()):
                        is_next_codeline_taken = True
                if len("".join(negative_prefix_code).strip()) != 0 and len(negative_next_codeline.strip()) != 0 and not is_next_codeline_taken:
                    negative_example = [negative_prefix_code, negative_next_codeline, 0]
                    positive_negative_examples.append(negative_example)
                    positive_negative_next_codelines.append(negative_next_codeline.strip())
                    NUMBER_OF_NEGATIVE_EXAMPLES_AST += 1
    
    # Get negative examples using comment based splits       
    for i in range(1, len(code_boundaries)):
        prefix_code = cleaned_code[code_boundaries[i-1] : code_boundaries[i]]
        for _ in range(negative_examples_to_sample_at_a_time):
            if len(prefix_code) > 1 and len("".join(prefix_code).strip()) != 0:
                random_codeline_index = np.random.randint(code_boundaries[i-1] + 1, code_boundaries[i])
                negative_prefix_code = cleaned_code[code_boundaries[i-1] : random_codeline_index]
                negative_next_codeline = cleaned_code[random_codeline_index]
                is_next_codeline_taken = False
                for line in positive_negative_next_codelines:
                    if line.strip().startswith(negative_next_codeline.strip()):
                        is_next_codeline_taken = True
                if len("".join(negative_prefix_code).strip()) != 0 and len(negative_next_codeline.strip()) != 0 and not is_next_codeline_taken:
                    negative_example = [negative_prefix_code, negative_next_codeline, 0]
                    positive_negative_examples.append(negative_example)
                    positive_negative_next_codelines.append(negative_next_codeline.strip())
                    NUMBER_OF_NEGATIVE_EXAMPLES += 1
    
    return positive_negative_examples

# Apply all the cleaning/preprocessing steps over code/documentation
def data_cleaning(raw_dataset):
    
    processesd_dataset = dict({})
    print("\nCollecting +ve/-ve exaples for splitting ...")
    for filename in tqdm.tqdm(raw_dataset):
        processesd_dataset[filename] = dict({})
        for code in raw_dataset[filename]:
                
            # Clean the code and extract +ve/-ve examples for splitting
            try:
                positive_negative_examples = preprocess_code(raw_dataset[filename][code]["code"])
            except Exception as e:
                print("\n Code-cell: ", raw_dataset[filename][code]["code"])
                raise e
                        
            # Add to the dataset
            processesd_dataset[filename][code] = dict({})
            processesd_dataset[filename][code]["code"] = raw_dataset[filename][code]["code"]
            processesd_dataset[filename][code]["positive_negative_examples"] = positive_negative_examples
                    
    return processesd_dataset

# Count total number of pairs in the dataset
def count_code_cells(raw_dataset):
    count_pairs = 0
    for f in raw_dataset:
        count_pairs += len(list(raw_dataset[f].keys()))
    return count_pairs

# Convert the dataset from dict to pandas dataframe format
def convert_to_dataframe(dataset):
    dataset_as_list = []
    for filename in list(dataset.keys()):
        for code_number in list(dataset[filename].keys()):
            dataset_as_list.append([filename, 
            dataset[filename][code_number]["code"], 
            dataset[filename][code_number]["positive_negative_examples"]])
    df = pd.DataFrame(dataset_as_list, columns=["notebook_name", "code", "positive_negative_examples"])
    return df

def main():

    # Directory of the notebooks
    path = "./notebooks-dataset/notebooks"

    # Get the file names
    filenames = [f for f in os.listdir(path) if f.endswith(".ipynb")]

    # Extract code-documentation pairs from notebooks
    print("\nExtracting code-doc pairs from notebooks....")
    raw_dataset = {}
    JSONDecodeError = 0
    for f in tqdm.tqdm(filenames):
        file = open(f"{path}/{f}", "r")
        try:
            file_as_json = json.loads(file.read())
            raw_dataset = extract_code_from_notebook(f, file_as_json, raw_dataset)
        except json.decoder.JSONDecodeError:
            JSONDecodeError += 1
    print("\ndecoding error: ", JSONDecodeError)
    print("\nTotal number of notebooks: {} and code-cells: {}".format(len(filenames), count_code_cells(raw_dataset)))

    # Clean the code and documentation
    processesd_dataset = data_cleaning(raw_dataset)
    print("\nNumber +ve and -ve examples by comments: {} and {}".format(NUMBER_OF_POSITIVE_EXAMPLES, NUMBER_OF_NEGATIVE_EXAMPLES))
    print("\nNumber +ve and -ve examples by AST: {} and {}".format(NUMBER_OF_POSITIVE_EXAMPLES_AST, NUMBER_OF_NEGATIVE_EXAMPLES_AST))
    print("\nNumber of AST parsing error: ", NUMBER_OF_AST_PARSING_ERROR)

    # Convert to dataframe and save as csv
    dataset_as_dataframe = convert_to_dataframe(processesd_dataset)
    print("\nShape of the dataframe: ", dataset_as_dataframe.shape)
    dataset_as_dataframe.to_csv("./coseg/dataset/processed_data/dataset.csv")

    # Convert to json file and save
    dataset_as_json = json.dumps(processesd_dataset, indent=4)
    with open("./coseg/dataset/processed_data/dataset.json", "w") as outfile:
        outfile.write(dataset_as_json)

main()
