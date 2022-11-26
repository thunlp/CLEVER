DATA_DIR=../data
OUTPUT_DIR=../output
SAMPLE_PER_CARD=6
ACCUMULATE_STEP=8
WARMUP_STEP=2000
HEAD=max
SHOT=100
VRD_weight=${OUTPUT_DIR}/VRD_lr5e-5_bsz160_MLP_100_shot_15_epoch/best/pytorch_model.bin

seed=42
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 --master_port 10225 oscar/run_bag.py \
    --eval_model_dir ${OUTPUT_DIR}/pretrained_model/pretrained_base/checkpoint-2000000/ \
    --VRD_weight ${VRD_weight} \
    --do_train \
    --train_dir ${DATA_DIR} \
    --test_dir ${DATA_DIR} \
    --do_lower_case \
    --learning_rate 0 \
    --warmup_steps ${WARMUP_STEP} \
    --gradient_accumulation_steps ${ACCUMULATE_STEP} \
    --per_gpu_train_batch_size ${SAMPLE_PER_CARD} \
    --per_gpu_eval_batch_size ${SAMPLE_PER_CARD} \
    --freeze_embedding \
    --output_dir ${OUTPUT_DIR}/${HEAD}_from_${VRD_weight} \
    --head ${HEAD}\
    --real_bag_size 50 \
    --select_size 50 \
    --seed 0
