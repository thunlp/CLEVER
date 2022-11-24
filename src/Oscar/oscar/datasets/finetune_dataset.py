# Copyright (c) 2022 THUNLP Lab Tsinghua University. Licensed under the MIT license.
# This file defines datasets and evaluation metrics for CLEVER
# Author: Tianyu Yu
# Data: 2022-09

import json
import pickle
import random
import base64

import numpy
import numpy as np
import torch
import torch.distributed
from collections import defaultdict
import sklearn.metrics
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile

real_bag_size = 50


# base_dir = f'/data_local/yutianyu'
# base_dir = f'/data/private/yutianyu'

class BalancedPositiveNegativePairSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs_per_image):
        """
        Arguments:
            matched matched_idxs_per_image: tensor containing -1, 0 or positive values.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
        Returns:
            pos_idx (tensor)
            neg_idx (tensor)

        Returns two binary masks for the image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        positive = torch.nonzero(matched_idxs_per_image >= 1, as_tuple=False).squeeze(1)
        negative = torch.nonzero(matched_idxs_per_image == 0, as_tuple=False).squeeze(1)

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # TODO: DEBUG ONLY, DELETE ME
        # Results show that use this line of code significantly improve the performance on train (test-all-box-pairs)
        # num_neg = min(num_pos, num_neg)
        # print(num_pos, num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]

        # create binary mask from indices
        pos_idx_per_image_mask = torch.zeros_like(
            matched_idxs_per_image, dtype=torch.bool
        )
        neg_idx_per_image_mask = torch.zeros_like(
            matched_idxs_per_image, dtype=torch.bool
        )

        pos_idx_per_image_mask[pos_idx_per_image] = 1
        neg_idx_per_image_mask[neg_idx_per_image] = 1

        return pos_idx_per_image_mask, neg_idx_per_image_mask


def cal_metrics(sorted_results, facts):
    correct = 0
    precisions = []
    recalls = []

    num_facts_of_predicates = defaultdict(int)
    for f in facts:
        p = f[2]
        num_facts_of_predicates[p] += 1
    recalls_of_predicate = {p: [] for p in num_facts_of_predicates}
    precision_of_predicate = {p: [] for p in num_facts_of_predicates}
    f1_of_predicate = {p: [] for p in num_facts_of_predicates}
    correct_of_predicates = {p: 0 for p in num_facts_of_predicates}
    count_predication_of_p = {p: 0 for p in num_facts_of_predicates}
    print(len(num_facts_of_predicates))

    class_pair_result = {}
    pr_curve_labels = []  # binary array
    pr_curve_predictions = []  # scores

    for i, item in enumerate(sorted_results):
        p = item[2]
        p_score = item[3]
        pair_key = (item[0], item[1])
        if p in num_facts_of_predicates:
            count_predication_of_p[p] += 1
        if pair_key not in class_pair_result:
            class_pair_result[pair_key] = {
                'pred': np.zeros((101), dtype=int),
                'label': np.zeros((101), dtype=int),
                'score': np.zeros((101), dtype=float)
            }

        if item[:3] in facts:
            correct += 1
            correct_of_predicates[p] += 1
            class_pair_result[pair_key]['label'][p] = 1
        class_pair_result[pair_key]['score'][p] = item[3]
        pr_curve_labels.append(item[:3] in facts)
        pr_curve_predictions.append(item[3])

        precisions.append(correct / (i + 1))
        recalls.append(correct / len(facts))

        if p in num_facts_of_predicates:
            pr = correct_of_predicates[p] / count_predication_of_p[p]
            rc = correct_of_predicates[p] / max(1, num_facts_of_predicates[p])
            recalls_of_predicate[p].append(rc)
            precision_of_predicate[p].append(pr)
            f1_of_predicate[p].append((2 * pr * rc) / (pr + rc + 1e-20))

    label_vec = []
    pred_result_vec = []
    score_vec = []
    for cls_p in class_pair_result:
        label_vec.append(class_pair_result[cls_p]['label'])
        pred_result_vec.append(class_pair_result[cls_p]['pred'])
        score_vec.append(class_pair_result[cls_p]['score'])
    label_vec = np.stack(label_vec, 0)
    pred_result_vec = np.stack(pred_result_vec, 0)
    score_vec = np.stack(score_vec, 0)

    np_recall_of_predicates = {p: np.array(r) for p, r in recalls_of_predicate.items()}
    np_precision_of_predicate = {p: np.array(pr) for p, pr in precision_of_predicate.items()}
    max_f1_of_predicate = {p: np.array(f1).max() for p, f1 in f1_of_predicate.items()}
    auc_of_predicate = {p: sklearn.metrics.auc(x=np_recall_of_predicates[p], y=np_precision_of_predicate[p]) for p, f1
                        in f1_of_predicate.items()}

    auc = sklearn.metrics.auc(x=recalls, y=precisions)
    np_prec = np.array(precisions)
    np_rec = np.array(recalls)
    max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()

    best_threshold = sorted_results[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()][3].item()

    pred_result_vec = score_vec >= best_threshold
    valid_p = list(num_facts_of_predicates.keys())
    # max_macro_f1 = sklearn.metrics.f1_score(label_vec[:, valid_p],
    #                                         pred_result_vec[:, valid_p],
    #                                         average='macro')
    max_macro_f1 = sum(max_f1_of_predicate.values()) / len(max_f1_of_predicate)
    assert len(auc_of_predicate) == len(valid_p)
    macro_auc = sum(auc_of_predicate.values()) / len(auc_of_predicate)
    macro_p = sum(np_precision_of_predicate.values(), np.zeros(len(np_recall_of_predicates[40]))) / len(
        np_precision_of_predicate)
    return label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, max_macro_f1, numpy.array(
        pr_curve_labels), numpy.array(pr_curve_predictions)


def load_cached_image_ids(data_dir, split):
    idx_file_name = f'{data_dir}/{split}_image_sort_by_IoU_and_distance.pkl'
    idx_to_id_fname = f'{data_dir}/{split}_image_idx_to_id.pkl'

    idx_to_id = pickle.load(open(idx_to_id_fname, 'rb'))
    bag_key_to_idx = pickle.load(open(idx_file_name, 'rb'))
    mapping = json.load(open(f'{data_dir}/vg_dict.json'))

    _150_idx_to_label = {v: k for k, v in mapping['label_to_vg_150_idx'].items()}
    bag_key_to_ids = {
        '#'.join([str(mapping['label_to_idx'][_150_idx_to_label[x]]) for x in k]): [idx_to_id[x] for x in v] for k, v in
        bag_key_to_idx.items()}

    return bag_key_to_ids


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_vg_150_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        # info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_vg_150_idx']
    predicate_to_ind = info['predicate_to_idx']
    # attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    # ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])
    _100_cls_idx_to_vg150_idx = {int(k): info['label_to_vg_150_idx'][info['idx_to_label'][k]] for k in
                                 info['idx_to_label']}
    return ind_to_classes, ind_to_predicates, [], _100_cls_idx_to_vg150_idx


class BagDatasetPairAsUnit(Dataset):
    def __init__(self, data_dir, bag_data_file, split, args=None, tokenizer=None, txt_seq_len=70, img_seq_len=50,
                 shuffle=False, **kwargs):
        self.split = split

        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.txt_seq_len = txt_seq_len
        self.img_seq_len = img_seq_len

        self.bag_data = json.load(open(bag_data_file))
        self.idx_to_bag_key = {i: k for i, k in enumerate(self.bag_data.keys())}

        self.label_tsv = TSVFile(f'{data_dir}/VG_100_100_label.tsv')
        self.line_tsv = TSVFile(f'{data_dir}/{split}_feat_idx_to_label_line.tsv')
        self.prediction = TSVFile(f'{data_dir}/obj_feat_{split}.tsv')

        self.bag_pair_data = json.load(open(f'{data_dir}/{split}_pairs_data.json'))
        self.img_id_to_key = {k: v[0] for k, v in enumerate(self.label_tsv)}

        self.key_to_prediction_line = {v[0]: i for i, v in enumerate(self.prediction)}

        self.predicate_to_idx = json.load(open(f'{data_dir}/vg_dict.json'))['predicate_to_idx']

        self.shuffle = shuffle

        mapping = json.load(open(f'{data_dir}/vg_dict.json'))
        self.idx_to_cls_name = mapping['idx_to_label']

        self.facts = set()
        for key in self.bag_data:
            label = self.bag_data[key]['label']
            key = key.split('#')
            for l in label:
                if l:
                    self.facts.add((key[0], key[1], l))
        print(f'{len(self.facts)} facts for {bag_data_file}')

        self.bag_key_to_sorted_image_ids = load_cached_image_ids(data_dir, split)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return len(self.idx_to_bag_key)

    def get_image_item(self, item, pairs):
        # load data
        img_108076_id_str, object_classes, object_features, object_boxes = self.decode_features(item)
        assert len(object_features) == len(object_boxes)

        object_classes = object_classes[: self.img_seq_len]
        object_features = object_features[: self.img_seq_len]
        object_boxes = object_boxes[: self.img_seq_len]
        pairs = [p for p in pairs if p[0] < self.img_seq_len and p[1] < self.img_seq_len]

        pair_image_feats = []
        pair_input_ids = []
        pair_input_mask = []
        pair_segment_ids = []
        pair_object_boxes = []
        pair_object_classes = []
        pair_object_name_positions = []
        for pair in pairs:
            obj_idx_permutation = [pair[0], pair[1],
                                   *[idx for idx in range(len(object_boxes)) if idx not in pair]]
            obj_class_permutation = [object_classes[idx] for idx in obj_idx_permutation]

            object_tag_text = " ".join(obj_class_permutation)
            text_a = object_tag_text
            text_b = ''

            # generate features
            input_ids, input_mask, segment_ids, lm_label_ids, tokens_a = tokenize(self.tokenizer,
                                                                                  text_a=text_a, text_b=text_b,
                                                                                  img_feat=object_features,
                                                                                  max_img_seq_len=self.img_seq_len,
                                                                                  max_seq_a_len=70, max_seq_len=70,
                                                                                  cls_token_segment_id=0,
                                                                                  pad_token_segment_id=0,
                                                                                  sequence_a_segment_id=0,
                                                                                  sequence_b_segment_id=1)
            object_name_positions = []
            current_object_positions = []
            for token_idx, tok in enumerate(tokens_a, 1):  # omit [CLS]
                tok: str

                # find a new name, save word-piece positions of previous one
                if not tok.startswith('##'):
                    object_name_positions.append(current_object_positions)
                    current_object_positions = []
                current_object_positions.append(token_idx)
            del object_name_positions[0]
            object_name_positions.append(current_object_positions)

            object_num = object_features.size(0)
            img_feat = torch.cat([torch.stack([object_features[idx] for idx in obj_idx_permutation]),
                                  torch.zeros([self.img_seq_len - object_num, 2054])], 0)

            assert len(object_name_positions) == len(object_classes)
            pair_image_feats.append(img_feat)
            pair_input_ids.append(input_ids)
            pair_input_mask.append(input_mask)
            pair_segment_ids.append(segment_ids)
            pair_object_boxes.append([object_boxes[idx] for idx in obj_idx_permutation])
            pair_object_classes.append(obj_class_permutation)
            pair_object_name_positions.append(object_name_positions)

        return (img_108076_id_str, pair_image_feats, pair_input_ids, pair_input_mask, pair_segment_ids,
                pair_object_boxes, pair_object_classes, pair_object_name_positions)

    def __getitem__(self, item):
        bag_key = self.idx_to_bag_key[item]
        bag_data = self.bag_data[bag_key]
        bag_pair_data = self.bag_pair_data[bag_key]
        bag_label = torch.tensor(random.choice(bag_data['label']))

        bag_image_ids = self.bag_key_to_sorted_image_ids[bag_key][:real_bag_size]
        rel_bag_image_ids = bag_data['relation_image_ids']

        bag_pairs = bag_pair_data[:real_bag_size]
        bag_image_id_to_pairs = {}
        attention_label = [int(img_id in rel_bag_image_ids) for img_id, *_ in bag_pairs]
        for pair in bag_pairs:
            img_id, sub_idx, obj_idx, v = pair
            if img_id not in bag_image_id_to_pairs:
                bag_image_id_to_pairs[img_id] = []
            bag_image_id_to_pairs[img_id].append((sub_idx, obj_idx))

        bag_image_feats = []
        bag_input_ids = []
        bag_input_mask = []
        bag_segment_ids = []
        bag_object_boxes = []
        bag_object_classes = []
        bag_object_name_positions = []
        for img_id, pairs in bag_image_id_to_pairs.items():
            idx = self.key_to_prediction_line[self.img_id_to_key[img_id]]
            (img_108076_id_str, pair_image_feats, pair_input_ids, pair_input_mask, pair_segment_ids,
             pair_object_boxes, pair_object_classes, pair_object_name_positions) = self.get_image_item(idx, pairs)

            bag_image_feats += pair_image_feats
            bag_input_ids += pair_input_ids
            bag_input_mask += pair_input_mask
            bag_segment_ids += pair_segment_ids
            bag_object_boxes += pair_object_boxes
            bag_object_classes += pair_object_classes
            bag_object_name_positions += pair_object_name_positions

        bag_image_feats = torch.stack(bag_image_feats, 0)
        bag_input_ids = torch.stack(bag_input_ids, 0)
        bag_input_mask = torch.stack(bag_input_mask, 0)
        bag_segment_ids = torch.stack(bag_segment_ids, 0)

        return (bag_key, bag_image_feats, bag_input_ids, bag_input_mask, bag_segment_ids, bag_object_boxes,
                bag_object_classes, bag_object_name_positions, [], [], bag_label, attention_label, bag_image_ids, [])

    def eval(self, pred_result):
        seen = set()  # 多卡训练的时候可能最后一个 batch 里面有重复的 bag 需要去掉
        deduplicated_pred_result = []
        for item in pred_result:
            if (item['class_pair'][0], item['class_pair'][1], item['relation']) in seen:
                continue
            deduplicated_pred_result.append(item)
            seen.add((item['class_pair'][0], item['class_pair'][1], item['relation']))
        pred_result = deduplicated_pred_result

        print(f'Sorting evaluation results, size={len(pred_result)}')
        sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)

        sorted_pred_result = [(x['class_pair'][0], x['class_pair'][1], x['relation'], x['score']) for x in
                              sorted_pred_result]

        label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, \
        max_macro_f1, pr_curve_labels, pr_curve_predictions = cal_metrics(sorted_pred_result, self.facts)
        return {
            'results': sorted_pred_result,
            'auc': auc,
            'macro_auc': macro_auc,
            'max_micro_f1': max_micro_f1,
            'max_macro_f1': max_macro_f1,
            'p@2%': np_prec[int(0.02 * len(np_prec))],
            'mp@2%': macro_p[int(0.02 * len(macro_p))],
            'recalls': np_rec.tolist(),
            'pr_curve_labels': pr_curve_labels,
            'pr_curve_predictions': pr_curve_predictions
        }

    def decode_features(self, item_idx):
        img_108076_id_str, prediction_str = self.prediction.seek(item_idx)
        feat_info = json.loads(prediction_str)

        label_tsv_row_idx = int(self.line_tsv[item_idx][0])
        _, annotation_str = self.label_tsv[label_tsv_row_idx]
        assert _ == img_108076_id_str

        annotation_info = json.loads(annotation_str)
        objects_annotation, relationships_annotation = annotation_info['objects'], annotation_info['relations']

        # prediction_objects: [{'rect':, 'feature':}, ...]
        prediction_objects = feat_info["objects"]
        object_features = [np.frombuffer(base64.b64decode(o['feature']), np.float32) for o in prediction_objects]
        object_features = torch.Tensor(np.stack(object_features))

        # class names
        prediction_classes = [o['class'] for o in prediction_objects]
        label_classes = [o['class'] for o in objects_annotation]
        assert len(label_classes) == len(prediction_classes)

        # bboxes
        prediction_boxes = [o['rect'] for o in prediction_objects]
        label_boxes = [o['rect'] for o in objects_annotation]
        assert len(prediction_boxes) == len(label_boxes)

        return img_108076_id_str, label_classes, object_features, label_boxes


class NormalFinetuneDataset(Dataset):
    def __init__(self, data_file, args=None, tokenizer=None, txt_seq_len=70, img_seq_len=50, shuffle=False, **kwargs):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.txt_seq_len = txt_seq_len
        self.img_seq_len = img_seq_len

        base_dir = f'/data_local/yutianyu'
        data_dir = f'{base_dir}/datasets/scene_graph_benchmark_data/visualgenome/visualgenome'
        if 'train' in data_file:
            if args.shot == -1:
                self.label_tsv = TSVFile(f'{data_dir}/yty_100_100_label_low_resource_keep{args.keep_ratio}.tsv')
            else:
                self.label_tsv = TSVFile(f'{data_dir}/yty_100_100_label_low_resource_{args.shot}shot.tsv')
        else:
            self.label_tsv = TSVFile(f'{data_dir}/yty_100_100_label.tsv')
        # self.label_tsv = TSVFile(f'{data_dir}/vg150.tsv')

        self.prediction = TSVFile(data_file)
        self.line_tsv = TSVFile(data_file + '.line_list.tsv')
        self.item_idx_to_prediction_line = list(range(len(self.prediction)))
        self.data_file = data_file
        if 'train' in data_file:
            if args.shot == -1:
                self.item_idx_to_prediction_line = json.load(
                    open(
                        f'{data_dir}/yty_100_100_label_low_resource_keep{args.keep_ratio}_prediction_line_numbers.json'))
            else:
                self.item_idx_to_prediction_line = json.load(
                    open(f'{data_dir}/yty_100_100_label_low_resource_{args.shot}shot_prediction_line_numbers.json'))

        self.predicate_to_idx = json.load(open(
            f'{base_dir}/datasets/VisualGenome/100_100_fix/VG-dicts_100_cls_100_pred_with_100_cls_to_vg_150_mapping.json'))[
            'predicate_to_idx']

        self.shuffle = shuffle

        # copy from scene-graph-benchmark config
        self.sampler = BalancedPositiveNegativePairSampler(512, 0.25)

    def __len__(self):
        return len(self.item_idx_to_prediction_line)

    def __getitem__(self, item):
        item = self.item_idx_to_prediction_line[item]
        # load data
        img_108076_id_str, object_classes, object_features, object_boxes, relations, relations_map = \
            self.decode_features(item)
        assert len(object_features) == len(object_boxes)

        if self.shuffle:
            random_positions = [i for i in range(len(object_boxes))]
            random.shuffle(random_positions)
            object_boxes = [object_boxes[i] for i in random_positions]
            object_classes = [object_classes[i] for i in random_positions]
            object_features = object_features[random_positions]

        object_classes = object_classes[: self.img_seq_len]
        object_features = object_features[: self.img_seq_len]
        object_boxes = object_boxes[: self.img_seq_len]
        relations = [r for r in relations if r[0] < len(object_boxes) and r[1] < len(object_boxes)]

        # Generate NA pairs, apply balance-sampling
        labels = []
        pair_box_ids = []
        for s in range(len(object_boxes)):
            for o in range(len(object_boxes)):
                if o == s:
                    continue
                if (s, o) in relations_map:
                    p = relations_map[(s, o)]
                else:
                    p = 0  # no-relation
                labels.append(p)
                pair_box_ids.append((s, o))
        labels = torch.tensor(labels)
        pair_box_ids = torch.tensor(pair_box_ids)

        sampled_pos_inds, sampled_neg_inds = self.sampler(labels)

        labels = labels[sampled_pos_inds + sampled_neg_inds]
        pairs = pair_box_ids[sampled_pos_inds + sampled_neg_inds]

        object_tag_text = " ".join(object_classes)
        text_a = object_tag_text
        text_b = ''

        # generate features
        input_ids, input_mask, segment_ids, lm_label_ids, tokens_a = tokenize(self.tokenizer,
                                                                              text_a=text_a, text_b=text_b,
                                                                              img_feat=object_features,
                                                                              max_img_seq_len=self.img_seq_len,
                                                                              max_seq_a_len=70, max_seq_len=70,
                                                                              cls_token_segment_id=0,
                                                                              pad_token_segment_id=0,
                                                                              sequence_a_segment_id=0,
                                                                              sequence_b_segment_id=1)
        object_name_positions = []
        current_object_positions = []
        for token_idx, tok in enumerate(tokens_a, 1):  # omit [CLS]
            tok: str

            # find a new name, save word-piece positions of previous one
            if not tok.startswith('##'):
                object_name_positions.append(current_object_positions)
                current_object_positions = []
            current_object_positions.append(token_idx)
        del object_name_positions[0]
        object_name_positions.append(current_object_positions)

        object_num = object_features.size(0)
        img_feat = torch.cat([object_features,
                              torch.zeros([self.img_seq_len - object_num, 2054])], 0)

        assert len(object_name_positions) == len(object_classes)
        return (img_108076_id_str, img_feat, input_ids, input_mask, segment_ids, object_boxes, object_classes,
                object_name_positions, relations, labels, pairs)

    def decode_features(self, item_idx):
        img_108076_id_str, prediction_str = self.prediction.seek(item_idx)
        feat_info = json.loads(prediction_str)

        label_tsv_row_idx = int(self.line_tsv[item_idx][0])
        _, annotation_str = self.label_tsv[label_tsv_row_idx]
        assert _ == img_108076_id_str

        annotation_info = json.loads(annotation_str)
        objects_annotation, relationships_annotation = annotation_info['objects'], annotation_info['relations']

        # prediction_objects: [{'rect':, 'feature':}, ...]
        prediction_objects = feat_info["objects"]
        object_features = [np.frombuffer(base64.b64decode(o['feature']), np.float32) for o in prediction_objects]
        object_features = torch.Tensor(np.stack(object_features))

        # class names
        prediction_classes = [o['class'] for o in prediction_objects]
        label_classes = [o['class'] for o in objects_annotation]
        assert len(label_classes) == len(prediction_classes), f'{len(label_classes)}, {len(prediction_classes)}'

        # bboxes
        prediction_boxes = [o['rect'] for o in prediction_objects]
        label_boxes = [o['rect'] for o in objects_annotation]
        assert len(prediction_boxes) == len(label_boxes)

        # relations
        relations = [(r['subj_id'], r['obj_id'], self.predicate_to_idx[r['class']]) for r in relationships_annotation]
        random.shuffle(relations)
        relations_map = {(s, o): p for s, o, p in relations}
        relations_tuples = [(s, o, p) for (s, o), p in relations_map.items()]
        for r in relations_tuples:
            assert 0 <= r[0] < len(label_boxes)
            assert 0 <= r[1] < len(label_boxes)

        return img_108076_id_str, label_classes, object_features, label_boxes, relations_tuples, relations_map


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def tokenize(tokenizer, text_a, text_b, img_feat, max_img_seq_len=50,
             max_seq_a_len=40, max_seq_len=70, cls_token_segment_id=0,
             pad_token_segment_id=0, sequence_a_segment_id=0, sequence_b_segment_id=1):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
    else:
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:(max_seq_len - 2)]

    t1_label = len(tokens_a) * [-1]
    if tokens_b:
        t2_label = [-1] * len(tokens_b)

    # concatenate lm labels and account for CLS, SEP, SEP
    if tokens_b:
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
    else:
        lm_label_ids = ([-1] + t1_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    # image features
    if max_img_seq_len > 0:
        img_feat_len = img_feat.shape[0]
        if img_feat_len > max_img_seq_len:
            input_mask = input_mask + [1] * img_feat_len
        else:
            input_mask = input_mask + [1] * img_feat_len
            pad_img_feat_len = max_img_seq_len - img_feat_len
            input_mask = input_mask + ([0] * pad_img_feat_len)

    lm_label_ids = lm_label_ids + [-1] * max_img_seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
    return input_ids, input_mask, segment_ids, lm_label_ids, tokens_a
