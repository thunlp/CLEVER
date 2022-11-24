seed=42
for lr in {5e-5,6e-5}; do
  for acc_step in {4,5,6}; do
    CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10437 oscar/run_bag.py \
      --eval_model_dir pretrained_model/pretrained_base/checkpoint-2000000/ \
      --do_train \
      --train_dir /data/private/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
      --test_dir /data/private/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
      --do_lower_case \
      --learning_rate ${lr} \
      --warmup_steps 1000 \
      --gradient_accumulation_steps ${acc_step} \
      --per_gpu_train_batch_size 6 \
      --per_gpu_eval_batch_size 1 \
      --num_train_epochs 100 \
      --freeze_embedding \
      --output_dir ../oscar_output/AAAI23_conatt_lr${lr}_batch_36_seed_${seed}_mAUC-weight3_acc_step${acc_step}\
      --loss_w_t -1.0\
      --sfmx_t 11 \
      --attention_w 0.0 \
      --head att \
      --select_size 50 \
      --real_bag_size 50 \
      --seed ${seed} \
      --mAUC_weight 3
    done
done