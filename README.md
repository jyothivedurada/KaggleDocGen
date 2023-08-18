# Cell2Doc - Machine Learning Pipeline for Generating Documentation in Computational Notebooks

#### NOTE 1: Please run all the scripts from the root folder.
#### NOTE 2: Raw notebooks, processed data, and model checkpoints are available in Zenodo: COMING SOON
#### NOTE 3: We have used Python 3.10 for all the experiments, please refer to requirements.txt for env specs.

## Data Collection

We have collected the notebooks having at least "10 upvotes" or "1 madel" which we considered as well-documented notebooks from KGTorrent(https://arxiv.org/abs/2103.10558). These criteria resulted in 5430 notebooks. The SQL queries are present in the sql_scripts.txt file.

KGTorrent Repository: https://github.com/collab-uniba/KGTorrent
<br/>KGTorrent Documentation: https://collab-uniba.github.io/KGTorrent/docs_build/html/index.html
<br/>KGTorrent Corpus: https://zenodo.org/record/4468523#.Y2LYG3ZBy3A

## Notebooks Dataset

"notebooks_to_dataset"(v3 is the latest) converts raw notebooks to code-documentation pairs and stores them in the "processed_data" folder. "split_notebook_dataset" script divides the processed data in the train/test/validation split.

Depending on the preprocessing logic, "notebooks_to_dataset" and "split_notebook_dataset" scripts need to be modified.

```
Scripts are available at: ./notebooks-dataset
```

## Code Documentation Model(CoDoc)

CoDoc can ideally be any model that can perform code documentation. For our study, we have tested with CodeBERT(https://aclanthology.org/2020.findings-emnlp.139/), GraphCodeBERT(https://openreview.net/pdf?id=jLoC4ez43PZ), UnixCoder(https://aclanthology.org/2022.acl-long.499.pdf), CodeT5 (https://aclanthology.org/2021.emnlp-main.685/), PLBART (https://arxiv.org/abs/2103.06333) and BLOOMZ (https://arxiv.org/abs/2211.01786).

All the models are implemented and tested in a similar fashion. "fine_tuning_script.sh" is responsible for fine-tuning, please change the dataset file(train and validation split) and output folder locations in the script before running. Like that, "testing_script.sh" tests the fine-tuned models. The finetuned checkpoints for all models and all four input representations (CM, CSM, ECSM, and SCSCM) are available in Zenodo.

```
CoDoc models and scripts are available at: ./codoc
```

## Code Segmentation Dataset
  
The code segmentation dataset is mined from the same notebook corpus that is used for documentation. We consider that control structures in AST and comments in the code define the boundary of logical contexts. So we create +ve and -ve examples on the basis of these two constraints. The script "prepare_dataset.py" is responsible for creating the dataset and "split_dataset.py" is responsible to get the train/test/validation splits.

```
The scripts are available at: ./coseg/dataset
```

## Code Segmentation Model(CoSeg)

Code Segmentation Model(CoSeg) is a binary classification model which is finetuned on CodeBERT using the code segmentation dataset. In terms of finetuning and testing the CoSeg, it is similar to CoDoc.

```
The scripts are available at: ./coseg/model
```

## Cell2Doc Pipeline (Combining CoSeg and CoDoc)
  
"inference.py" script is responsible for combining CoSeg and CoDoc to generate documentation for a single code snippet. It first uses CoSeg to get the individual code contexts and then generates documentation for each context using CoDoc (CodeBERT, UnixCoder, GraphCodeBERT, CodeT5, and PLBART).

```
The scripts are available at: ./coseg/inference
```

## Contact

Please feel free to contact Tamal Mondal (cs21mtech12001@iith.ac.in or tamalmondal495@gmail.com) if you have any further questions.

