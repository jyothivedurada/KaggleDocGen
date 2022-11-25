# Cell2Doc - Machine Learning Pipeline for Generating Documentation in Computational Notebooks

## Data Collection

We have collected the notebooks having atleast "10 upvotes" or "1 madel" which we considered as well-documented notebooks from KGTorrent(https://arxiv.org/abs/2103.10558). These critetia resulted into 5430 notebooks. The SQL queries are present in the sql_scripts.txt file.

KGTorrent Repository: https://github.com/collab-uniba/KGTorrent
<br/>KGTorrent Documentation: https://collab-uniba.github.io/KGTorrent/docs_build/html/index.html
<br/>KGTorrent Corpus: https://zenodo.org/record/4468523#.Y2LYG3ZBy3A

## Notebooks Dataset

Raw notebooks, processed data with the pre-processing scripts are avaiable in this location: /home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset

"notebooks_to_dataset"(v3 is the latest) converts raw notebooks to code-documentation pairs and stores in "processed_data" folder. "split_notebook_dataset" script divides the processed data in train/test/validation split.

Depending on the preprocessing logic, "notebooks_to_dataset" and "split_notebook_dataset" scripts needs to be modified.

## Training and Testing of Code Documentation Model(CDM)

CDM can ideally be any model that can perform code documnetation. For our study, we have tested with CodeBERT(https://aclanthology.org/2020.findings-emnlp.139/), GraphCodeBERT(https://openreview.net/pdf?id=jLoC4ez43PZ) and UnixCoder(https://aclanthology.org/2022.acl-long.499.pdf).

All the models are implemented and tested in similar fashion. "fine_tuning_script.sh" is reponsible for fine-tuning, please change the dataset file(train and validation split) and output folder locations in the script before running. Like that, "testing_script.sh" is responsible for testing the fine-tuned models.

CodeBERT Implementation: /home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/repository/Code-Text/code-to-text/code

GraphCodeBERT Implementation: /raid/cs21mtech12001/Research/CodeBERT/Repository/GraphCodeBERT/code-summarization

UnixCoder Implementation: /raid/cs21mtech12001/Research/CodeBERT/Repository/UniXcoder/downstream-tasks/code-summarization

## Different Input Representations for CDM and Their Testing

We use the follwing 5 different input representations which are tested using CodeBERT, GraphCodeBERT and UnixCoder. Here the only difference is in the dataset creations; trainining and testing of the model remains same(ECSM in GraphCodeBERT and Code + Comment setup need small changes in training/testing script).

To apply the dataset related changes, use "notebooks_to_dataset" and "split_notebook_dataset" scripts.

Processed Data Folder: /home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/processed_data/
Splitted Data Folder: /home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset/splitted_data/

### Code - Markdown(CM)

Here the code is cleaned as usual. The basic cleaning is done for markdown texts and those are not summarized. The restrictions like only taking first sentence, no punctuation etc. that can reduce the length, we haven't applied to show th effect of raw markdown text.

Processed Data: <processed data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only
Splitted Data: <splitted data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2
CodeBERT Results: <CodeXglue folder>/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/without_summarization/code-with-usm-only-2

### Code - Summarized Markdown(CSM)

Here the code is cleaned as usual. All the cleaning steps are applied for markdown texts which are also summarized.

Processed Data: <processed data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only
Splitted Data: <splitted data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only
CodeBERT Results: <CodeXglue folder>output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/code-with-sm-only

### English Code tokens - Summarized Markdown(ECSM)

Similar to CSM where in place of complete code, only english like code tokens are used as input to the model.

Processed Data: Can use the data from "code-with-sm-only" folder, from which english code tokens are extracted using "split_notebook_dataset" script
Splitted Data: <splitted data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm
CodeBERT Results: <CodeXglue folder>/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/english-code-tokens-with-sm

### Split Code - Summarized Comment and Markdown(SCSCM)

Here summarized comment and markdowns are considered as seperate datapoints and so each code-markdown pair can produce more than one datapoints. For code-comment pairs, the immediate code after comment(till next comment) is considered.

Processed Data: <processed data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/todo-18
Splitted Data: <splitted data folder>/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18
CodeBERT Results: <CodeXglue folder>/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18

NOTE: Check in similar file paths for the results using GraphCodeBERT and UnixCoder. 

## Code Segmenation Dataset

## Training and Testing of Code Segmentation Model(CSM)

## Combining CDM with CSM

## Other Scripts

## Cell2Doc paper specific details
