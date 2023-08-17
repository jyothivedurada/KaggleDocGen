batch_size=1
beam_size=10
source_length=256
target_length=128
test_model=./codoc/CodeBERT/output/csn-checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python3 ./codoc/CodeBERT/code/predict.py \
        --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $test_model \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size