DATA_DIR=../data
OUTPUT_DIR=../output
EPOCH=30
EVAL_PERIOD=2 # eval per k epoch
LR=5e-5
SHOT=100

CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10244 oscar/run_instance_pred_cls.py \
  --eval_model_dir ${OUTPUT_DIR}/pretrained_model/pretrained_base/checkpoint-2000000/ \
  --do_train \
  --train_dir ${DATA_DIR} \
  --test_dir ${DATA_DIR} \
  --do_lower_case \
  --learning_rate ${LR} \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 16 \
  --freeze_embedding \
  --shot ${SHOT}\
  --eval_period ${EVAL_PERIOD} \
  --num_train_epochs ${EPOCH} \
  --output_dir ${OUTPUT_DIR}/VRD_lr${LR}_bsz64_${SHOT}shot_${EPOCH}epoch
