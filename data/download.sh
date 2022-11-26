# Download meta file
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_vg_dict.json -O vg_dict.json

# Download bag split
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_val_bag_data.json -O val_bag_data.json
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_test_bag_data.json -O test_bag_data.json
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_train_bag_data.json -O train_bag_data.json

# Downlaod bag object pairs info
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_val_pairs_data.json -O val_pairs_data.json
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_test_pairs_data.json -O test_pairs_data.json
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_train_pairs_data.json -O train_pairs_data.json

# Download bag image sampling cache
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_val_selected_image_of_bags.pkl -O val_image_sort_by_IoU_and_distance.pkl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_test_selected_image_of_bags.pkl -O test_image_sort_by_IoU_and_distance.pkl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_train_selected_image_of_bags.pkl -O train_image_sort_by_IoU_and_distance.pkl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_val_image_idx_to_id.pkl -O val_image_idx_to_id.pkl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_test_image_idx_to_id.pkl -O test_image_idx_to_id.pkl
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_train_image_idx_to_id.pkl -O train_image_idx_to_id.pkl


# Download feature file
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_val_obj_feat.json -O obj_feat_val.tsv
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_test_obj_feat.json -O obj_feat_test.tsv
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_train_obj_feat.json -O obj_feat_train.tsv
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_val_feat_idx_to_label_line.tsv -O val_feat_idx_to_label_line.tsv
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_test_feat_idx_to_label_line.tsv -O test_feat_idx_to_label_line.tsv
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_train_feat_idx_to_label_line.tsv -O train_feat_idx_to_label_line.tsv


# Download label file
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_AAAI23_VG100-100_label.tsv -O VG_100_100_label.tsv

# Download VRD-baseline data files
mkdir VRD && cd VRD
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_VRD-data_100shot.tsv -O VRD_100shot.tsv
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/CLEVER_CKE_VRD-data_100shot_feat-file_line-ids.json -O VRD_100shot_feat-lines.json
