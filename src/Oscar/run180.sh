#CUDA_VISIBLE_DEVICES=1,2,3,4,9 python -m torch.distributed.launch --nproc_per_node=5 --master_port 10223 oscar/run_instance_pred_cls.py \
#    --eval_model_dir pretrained_model/pretrained_base/checkpoint-2000000/ \
#    --do_train \
#    --train_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/\
#    --test_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/\
#    --do_lower_case \
#    --learning_rate 4e-5 \
#    --per_gpu_train_batch_size 32 \
#    --per_gpu_eval_batch_size 16 \
#    --num_train_epochs 15 \
#    --freeze_embedding \
#    --output_dir ../oscar_output/tempse;

#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 --master_port 10223 oscar/run_bag.py \
#    --eval_model_dir output/lr4e-5_bsz160_MLP/best \
#    --do_train \
#    --train_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/\
#    --test_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/\
#    --do_lower_case \
#    --learning_rate 4e-5 \
#    --per_gpu_train_batch_size 1 \
#    --per_gpu_eval_batch_size 32 \
#    --num_train_epochs 15 \
#    --freeze_embedding \
#    --output_dir ../oscar_output/bag_att_lr4e-5_bsz5_grad1_use_sorted_images_sfmx5 \
#    --sfmx_t 5;

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=10 --master_port 10224 oscar/run_bag.py \
#  --eval_model_dir output/lr4e-5_bsz160_MLP/best \
#  --do_train \
#  --train_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
#  --test_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
#  --do_lower_case \
#  --learning_rate 5e-5 \
#  --per_gpu_train_batch_size 1 \
#  --per_gpu_eval_batch_size 1 \
#  --num_train_epochs 10 \
#  --freeze_embedding \
#  --output_dir ../oscar_output/use_IoU_selected_10_images_for_train_and_test_lr5e-5_ecpoh10_att_classifier_label_score0.7 \
#  --sfmx_t 18

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=10 --master_port 10225 oscar/run_bag.py \
#  --eval_model_dir ../oscar_output/lr5e-5_bsz160_MLP_low_resource_100_shot_30_epoch/best \
#  --do_train \
#  --train_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
#  --test_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
#  --do_lower_case \
#  --learning_rate 5e-5 \
#  --per_gpu_train_batch_size 1 \
#  --per_gpu_eval_batch_size 1 \
#  --num_train_epochs 10 \
#  --freeze_embedding \
#  --output_dir ../oscar_output/show_max \
#  --sfmx_t 18

# IoU_selected_10_images_for_train_and_test_lr5e-5_ecpoh10_64_dim_feature_use_diag

#pretrained_model/pretrained_base/checkpoint-2000000/

#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 --master_port 10225 oscar/run_bag.py \

CUDA_VISIBLE_DEVICES=5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=5 --master_port 10224 oscar/run_bag.py \
  --eval_model_dir pretrained_model/pretrained_base/checkpoint-2000000/ \
  --do_train \
  --train_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
  --test_dir /data_local/yutianyu/VinVL/scene_graph_benchmark/datasets/visualgenome/ \
  --do_lower_case \
  --learning_rate 8e-5 \
  --warmup_steps 1000 \
  --gradient_accumulation_steps 12 \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 1 \
  --num_train_epochs 100 \
  --freeze_embedding \
  --output_dir ../oscar_output/new_metric_att_8e-5_bag50_bsz_60_200perEval_t11_init_all_fix_acc_CLIP_hasRelationWith \
  --loss_w_t -1.0\
  --sfmx_t 11 \
  --attention_w 0.0 \
  --head att \
  --select_size 50 \
  --real_bag_size 50 \
  --num_workers 1
