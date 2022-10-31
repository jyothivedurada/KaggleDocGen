lang=python #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/notebooks_dataset/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/same_code_length
output_dir=/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/same_code_length
train_file=$data_dir/train_dataset.jsonl
dev_file=$data_dir/valid_dataset.jsonl
epochs=100
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
already_trained_model=$output_dir/pytorch_model.bin

python3 ./repository/Code-Text/code-to-text/code/run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --load_model_path $already_trained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs