batch_size=1
beam_size=10
source_length=256
target_length=128
data_dir=/raid/cs21mtech12001/Research/CodeSearchNet/dataset/python
output_dir=/raid/cs21mtech12001/Research/CodeXGLUE/output/temp_output
dev_file=$data_dir/valid.jsonl
test_file=$data_dir/test.jsonl
test_model=/raid/cs21mtech12001/Research/CodeXGLUE/output/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python3 ./repository/Code-Text/code-to-text/code/visualize.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size