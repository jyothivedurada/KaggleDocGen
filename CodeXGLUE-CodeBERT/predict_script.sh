batch_size=1
beam_size=10
source_length=256
target_length=128
test_model=/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/new/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python3 /home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/repository/Code-Text/code-to-text/code/predict.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size