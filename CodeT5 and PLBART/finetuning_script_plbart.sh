DATA_DIR=/raid/cs21mtech12001/Research/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18
TASK=summarize
SUB_TASK=python-notebook
MODEL_TAG=plbart
GPU=0
DATA_NUM=-1
BS=8
LR=5
SRC_LEN=256
TRG_LEN=128
PATIENCE=2
EPOCH=15
WARMUP=1000
MODEL_DIR=/raid/cs21mtech12001/Research/CodeT5/Repository/output
SUMMARY_DIR=/raid/cs21mtech12001/Research/CodeT5/Repository/summary
RES_FN=python_notebooks_summarization_PLBART_scscm_2
DATA_TAG='scscm_2(todo-18)'
FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}_pat${PATIENCE}_e${EPOCH}
OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == bart_large ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-large
  MODEL_PATH=facebook/bart-large
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
elif [[ $MODEL_TAG == codet5_large ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-large
  MODEL_PATH=Salesforce/codet5-large
elif [[ $MODEL_TAG == plbart ]]; then
  MODEL_TYPE=plbart
  TOKENIZER=uclanlp/plbart-large
  MODEL_PATH=uclanlp/plbart-large
fi

RUN_FN=/raid/cs21mtech12001/Research/CodeT5/Repository/run_gen.py
SAVED_MODEL=/raid/cs21mtech12001/Research/CodeT5/Repository/output/summarize/python-csn/plbart_csn_lr5_bs16_src256_trg128_pat2_e15/checkpoint-best-bleu/pytorch_model.bin

CUDA_VISIBLE_DEVICES=${GPU} \
python3 ${RUN_FN}   \
  --do_train --do_eval --do_eval_bleu --do_test  \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${DATA_DIR}  \
  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  --load_model_path ${SAVED_MODEL} \
  2>&1 | tee ${LOG}
