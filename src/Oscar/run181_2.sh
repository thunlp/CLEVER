seed=42
for lr in {6e-5,7e-5}; do
  CUDA_VISIBLE_DEVICES=5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=5 --master_port 10226 oscar/run_bag.py \
    --eval_model_dir pretrained_model/pretrained_base/checkpoint-2000000/ \
    --do_train \
    --train_dir ../scene_graph_benchmark/datasets/visualgenome/ \
    --test_dir ../scene_graph_benchmark/datasets/visualgenome/ \
    --do_lower_case \
    --learning_rate ${lr} \
    --warmup_steps 2000 \
    --gradient_accumulation_steps 8 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --num_train_epochs 100 \
    --freeze_embedding \
    --output_dir ../oscar_output/AAAI23_conatt_lr${lr}_batch_40_seed${seed}_warm_2k_mAUC-weight3 \
    --loss_w_t -1.0\
    --sfmx_t 11 \
    --attention_w 0.0 \
    --head att\
    --select_size 50 \
    --real_bag_size 50 \
    --seed ${seed} \
    --mAUC_weight 3
done