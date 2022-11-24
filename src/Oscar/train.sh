DATA_DIR=../data
OUTPUT_DIR=../output
SAMPLE_PER_CARD=1
ACCUMULATE_STEP=8
WARMUP_STEP=2000
HEAD=att

seed=42
for lr in {4e-5,5e-5,6e-5}; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=10 --master_port 10225 oscar/run_bag.py \
    --eval_model_dir ${OUTPUT_DIR}/pretrained_model/pretrained_base/checkpoint-2000000/ \
    --pretrained_weight ${OUTPUT_DIR}/pretrained_avg_model/model.bin \
    --do_train \
    --train_dir ${DATA_DIR} \
    --test_dir ${DATA_DIR} \
    --do_lower_case \
    --learning_rate ${lr} \
    --warmup_steps ${WARMUP_STEP} \
    --gradient_accumulation_steps ${ACCUMULATE_STEP} \
    --per_gpu_train_batch_size ${SAMPLE_PER_CARD} \
    --per_gpu_eval_batch_size ${SAMPLE_PER_CARD} \
    --num_train_epochs 100 \
    --freeze_embedding \
    --output_dir ${OUTPUT_DIR}/${HEAD}_lr${lr}_bsz-10card-${SAMPLE_PER_CARD}-${ACCUMULATE_STEP}_seed${seed}_warm${WARMUP_STEP} \
    --sfmx_t 11 \
    --attention_w 0.0 \
    --head ${HEAD}\
    --real_bag_size 50 \
    --select_size 50 \
    --seed ${seed} \
    --mAUC_weight 3
done
