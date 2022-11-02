# Cell2Doc - Machine Learning Pipeline for Generating Documentation in Computational Notebooks

## Data Collection

We have collected the notebooks having atleast "10 upvotes" or "1 madel" which we considered as well-documented notebooks from KGTorrent(https://arxiv.org/abs/2103.10558). These critetia resulted into 5430 notebooks. The SQL queries are present in the sql_scripts.txt file.

KGTorrent Repository: https://github.com/collab-uniba/KGTorrent
<br/>KGTorrent Documentation: https://collab-uniba.github.io/KGTorrent/docs_build/html/index.html
<br/>KGTorrent Corpus: https://zenodo.org/record/4468523#.Y2LYG3ZBy3A

## Notebooks Dataset

Raw notebooks, processed data with the pre-processing scripts are avaiable in this location: /home/cs19btech11056/cs21mtech12001-Tamal/Notebooks_Dataset

"notebooks_to_dataset"(v3 is the latest) converts raw notebooks to code-documentation pairs and stores in "processed_data" folder. "split_notebook_dataset" script divides the processed data in train/test/validation split.

Depending on the preprocessing logic, notebooks_to_dataset and split_notebook_dataset needs to be modified.

## Training and Testing of Code Documentation Model(CDM)

CDM can ideally be any model that can perform code documnetation. For our study, we have tested with CodeBERT(https://aclanthology.org/2020.findings-emnlp.139/), GraphCodeBERT(https://openreview.net/pdf?id=jLoC4ez43PZ) and UnixCoder(https://aclanthology.org/2022.acl-long.499.pdf).

All the models are implemented and tested in similar fashion. "fine_tuning_script.sh" is reponsible for fine-tuning, please change the dataset file(train and validation split) and output folder locations in the script before running. Like that, "testing_script.sh" is responsible for testing the fine-tuned models.

CodeBERT Implementation: /home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/repository/Code-Text/code-to-text/code

GraphCodeBERT Implementation: /raid/cs21mtech12001/Research/CodeBERT/Repository/GraphCodeBERT/code-summarization

UnixCoder Implementation: /raid/cs21mtech12001/Research/CodeBERT/Repository/UniXcoder/downstream-tasks/code-summarization

## Different Input Representations for CDM and Their Testing

## Code Segmenation Dataset

## Training and Testing of Code Segmentation Model(CSM)

## Combining CDM with CSM

## Other Scripts

## Cell2Doc paper specific details
