# Copyright (c) 2022 THUNLP Lab Tsinghua University. Licensed under the MIT license.
# This file contains implementation of some existing multi-instance-learning baselines and CLEVER
# Author: Tianyu Yu
# Data: 2022-09

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math

import torch
from torch import nn
from oscar.utils.comm import get_rank
from oscar.modeling.modeling_bert import BertImgModel
from transformers.pytorch_transformers.modeling_bert import BertPreTrainedModel

logger = logging.getLogger(__name__)

NC = '\033[0;0m'
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
LIGHT_BLUE = '\033[0;34m'
LIGHT_PURPLE = '\033[0;35m'

sfmx_t = 5
attention_w = 0.1
head = 'att'
select_size = 10
loss_w_t = 0.5


def attention_analysis_file():
    return f'/data/private/yutianyu/dump_score/{get_rank()}.txt'


class VRDBaselineModel(BertPreTrainedModel):
    def __init__(self, config):
        super(VRDBaselineModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        self.dropout = nn.Dropout(0.1)

        feat_size = config.hidden_size * 4
        num_cls = 101

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, feat_size * 2),
            nn.ReLU(),
            nn.Linear(feat_size * 2, num_cls)
        )

        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, relation_list=None,
                position_ids=None, head_mask=None, img_feats=None, object_box_lists=None,
                object_name_positions_lists=None, rel_labels_list=None, pairs_list=None):
        """

        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param relation_list: List of [[s, o, p], ...], each [[s, o, p], ...] contains relations labels
                              of one image
        :param position_ids:
        :param head_mask:
        :param img_feats:
        :param object_box_lists:
        :param object_name_positions_lists:
        :param rel_labels_list:
        :param pairs_list:
        :return:
        """
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        # [ [CLS]-emb, tok-0-emb, tok-1-emb, ..., obj-0-emb, obj-1-emb, ...]
        sequence_output = outputs[0]

        num_object_of_images = [len(lst) for lst in object_box_lists]
        img_hidden = self._get_image_hidden(sequence_output, num_object_of_images)

        assert len(object_name_positions_lists) == len(num_object_of_images)
        object_name_hidden = self._get_obj_name_hidden(sequence_output, object_name_positions_lists)

        labels = []
        logits = []

        for object_img_features, object_name_features, relations, rel_labels, pair_idxs in zip(
                img_hidden, object_name_hidden, relation_list, rel_labels_list, pairs_list):
            relation_pairs = {(s, o): p for s, o, p in relations}

            object_features = torch.cat([object_img_features, object_name_features], dim=-1)

            if self.training:
                # print(f'Training {pair_idxs.shape}')
                # [N, 768 * 4], where N is num of pairs
                img_pair_features = object_features[pair_idxs].view(pair_idxs.size(0), -1)

                # [N, 101]
                img_rel_logits = self.classifier(img_pair_features)

                logits.append(img_rel_logits)
                labels.append(rel_labels)
            else:
                for s in range(len(object_img_features)):
                    for o in range(len(object_img_features)):
                        if o == s:
                            continue
                        if (s, o) in relation_pairs:
                            p = relation_pairs[(s, o)]
                        else:
                            p = 0  # no-relation
                        s_feat = object_features[s]
                        o_feat = object_features[o]
                        pair_feat = torch.cat([s_feat, o_feat], dim=-1)
                        logits.append(self.classifier(pair_feat))
                        labels.append(p)
        if self.training:
            logits = torch.cat(logits, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            logits = torch.stack(logits)
            labels = torch.tensor(labels).to(logits.device)

        loss = None
        if self.training:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

        scores = logits.softmax(-1)
        return loss, scores, labels

    def _get_image_hidden(self, sequence_output, num_obj_of_images):
        outputs = []
        for seq, num_obj in zip(sequence_output, num_obj_of_images):
            outputs.append(seq[70:70 + num_obj])
        return outputs

    def _get_obj_name_hidden(self, sequence_output, object_name_positions_lists):
        outputs = []
        for seq, object_name_positions in zip(sequence_output, object_name_positions_lists):
            name_feats = []
            for object_name_pos in object_name_positions:
                object_name_feat = seq[object_name_pos].sum(dim=0) / len(object_name_pos)
                name_feats.append(object_name_feat)
            outputs.append(torch.stack(name_feats))
        return outputs


class BagAttention(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagAttention, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101

        self.fc = classifier
        self.softmax = nn.Softmax(-1)
        self.diag = nn.Parameter(torch.ones(self.pooling_dim))
        self.drop = nn.Dropout(p=0.2)

        self.discriminator = nn.Sequential(
            nn.Linear(self.pooling_dim, self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, 1)
        )

    def forward(self, features, picked_label, bag_attention_target, write_attention=False):
        """
        name pooling_dim with `H`

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # (N, H)
        # features = torch.stack(features)

        if self.training and False:
            # (1, H)
            # att_mat = self.fc.weight[picked_label].unsqueeze(0)
            # assert att_mat.shape == (1, self.pooling_dim + 400), f'{att_mat.shape}, {(1, self.pooling_dim)}'
            #
            # # (1, H) * (1, H) -> (1, H)
            # att_mat = att_mat * self.diag.unsqueeze(0)
            # assert att_mat.shape == (1, self.pooling_dim + 400), f'{att_mat.shape}, {(1, self.pooling_dim)}'

            # (N, H) * (1, H) -> (N, H) -> (N)
            # att_score = (features * att_mat).sum(-1)
            att_score = self.fc(features)[:, picked_label]
            assert att_score.shape == (len(features),), f'{att_score.shape}, {(len(features),)}'

            # (N) -> (N)
            softmax_att_score = self.softmax(att_score / sfmx_t)
            # if random.random() < 0.01:
            #     msg = f'{RED}Train: {softmax_att_score} {GREEN}{bag_attention_target}{NC}'
            #     print(msg)
            #     logging.info(msg)

            # (N, 1) * (N, H) -> (N, H) -> (H)
            bag_feature = (softmax_att_score.unsqueeze(-1) * features).sum(0)
            # assert torch.equal(bag_feature, features[0]) # Test bag-size 1

            # (H)
            bag_feature = self.drop(bag_feature)

            # (R)
            bag_logits = self.fc(bag_feature)
            # if random.random() < 0.001:
            #     v = bag_logits.softmax(-1)
            #     msg = f'{RED}Train logits: {NC}[' + ', '.join(
            #         map(lambda x: f'{int(x * 1000)}',
            #             v.tolist())) + f'] {GREEN}{int(v[picked_label].item() * 1000)}{NC}'
            #     print(msg)
        else:
            # (H, R)
            # att_mat = self.fc.weight.transpose(0, 1)

            # (H, R) * (H, 1) -> (H, R)
            # att_mat = att_mat * self.diag.unsqueeze(1)

            # (N, H) x (H, R) -> (N, R)
            # att_score = torch.matmul(features, att_mat)
            att_score = self.fc(features)

            # (N, R) -> (R, N)
            # print(f'temp is {sfmx_t}')
            softmax_att_score = self.softmax(att_score.transpose(0, 1) / sfmx_t)

            # (R, N) x (N, H) -> (R, H)
            feature_for_each_rel = torch.matmul(softmax_att_score, features)

            # (R, H) -> (R, R) -> (R)
            bag_logits = self.fc(feature_for_each_rel).diagonal().contiguous()

            if not self.training:
                bag_logits = self.softmax(bag_logits)

        attention_loss = torch.nn.BCEWithLogitsLoss()(self.discriminator(features).squeeze(-1),
                                                      bag_attention_target.to(att_score.device).float())

        return bag_logits, attention_loss


class BagOne(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagOne, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101

        self.fc = classifier
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(p=0.2)

        self.discriminator = nn.Sequential(
            nn.Linear(self.pooling_dim, self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, 1)
        )

    def forward(self, features, picked_label, bag_attention_target):
        """

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """

        if self.training:
            # (N, R)
            instance_scores = self.fc(features).softmax(dim=-1)

            # (N, R) -> (N, ) -> 1
            max_index = instance_scores[:, picked_label].argmax()

            # (N, H) -> (H, )
            bag_rep = features[max_index]

            # (H, ) -> (R, )
            bag_logits = self.fc(self.drop(bag_rep))
        else:
            # (N, R)
            instance_scores = self.fc(features).softmax(dim=-1)

            # (N, R) -> (R, )
            score_for_each_rel = instance_scores.max(dim=0)[0]

            bag_logits = score_for_each_rel

        assert bag_logits.shape == (self.num_rel_cls,)
        attention_loss = 0

        return bag_logits, attention_loss


class BagOriginAttention(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagOriginAttention, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101

        self.fc = classifier
        self.softmax = nn.Softmax(-1)
        self.diag = nn.Parameter(torch.ones(self.pooling_dim))
        self.drop = nn.Dropout(p=0.2)

        self.discriminator = nn.Sequential(
            nn.Linear(self.pooling_dim, self.pooling_dim),
            nn.ReLU(),
            nn.Linear(self.pooling_dim, 1)
        )

    def forward(self, features, picked_label, bag_attention_target):
        """
        name pooling_dim with `H`

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # (N, H)
        # features = torch.stack(features)

        if self.training:
            att_score = self.fc(features)[:, picked_label]
            assert att_score.shape == (len(features),), f'{att_score.shape}, {(len(features),)}'

            # (N) -> (N)
            softmax_att_score = self.softmax(att_score)

            # (N, 1) * (N, H) -> (N, H) -> (H)
            bag_feature = (softmax_att_score.unsqueeze(-1) * features).sum(0)
            # assert torch.equal(bag_feature, features[0]) # Test bag-size 1

            # (H)
            bag_feature = self.drop(bag_feature)

            # (R)
            bag_logits = self.fc(bag_feature)
        else:
            att_score = self.fc(features)

            # (N, R) -> (R, N)
            softmax_att_score = self.softmax(att_score.transpose(0, 1))

            # (R, N) x (N, H) -> (R, H)
            feature_for_each_rel = torch.matmul(softmax_att_score, features)

            # (R, H) -> (R, R) -> (R)
            bag_logits = self.fc(feature_for_each_rel).diagonal().contiguous()

            if not self.training:
                bag_logits = self.softmax(bag_logits)

        attention_loss = torch.nn.BCEWithLogitsLoss()(self.discriminator(features).squeeze(-1),
                                                      bag_attention_target.to(att_score.device).float())

        return bag_logits, attention_loss


class BagAverage(nn.Module):
    def __init__(self, pooling_dim, classifier):
        super(BagAverage, self).__init__()
        self.pooling_dim = pooling_dim
        self.num_rel_cls = 101
        self.drop = nn.Dropout(p=0.2)
        self.classifier = classifier

    def forward(self, features, picked_label, bag_attention_target):
        """

        :param features: list of (pooling_dim, ) tensor
        :return: (num_rel_cls, ) multi-class classification logits
        """
        # features = self.encoder(features)

        # (N, H) -> (H, )
        mean = torch.mean(features, dim=0)
        mean = self.drop(mean)
        bag_logits = self.classifier(mean)
        if not self.training:
            return bag_logits.softmax(-1), 0
        return bag_logits, 0


loss_weight_mapping = {0: 440.59513211235145,
                       54: 110.32487581783062,
                       51: 66.93374551939341,
                       33: 50.139223301518015,
                       8: 41.85005530474936,
                       39: 36.878871732195094,
                       98: 31.628015101468367,
                       53: 23.793742588542496,
                       88: 22.620054402246215,
                       1: 20.56463634969338,
                       40: 16.45791874150557,
                       37: 10.61060933667979,
                       96: 8.681973910344412,
                       58: 7.621084731017706,
                       74: 6.282771784689187,
                       7: 5.644552344228087,
                       26: 4.939925713997064,
                       31: 4.373901853494484,
                       81: 3.8737330214188908,
                       11: 3.396503631015793,
                       18: 2.91547547143941,
                       9: 2.7550418409759665,
                       46: 2.615388003787501,
                       10: 2.4822557041660707,
                       6: 2.3779620821870386,
                       89: 2.346876979590424,
                       97: 2.305120336532797,
                       87: 2.158412668472037,
                       57: 1.9383963944310325,
                       44: 1.8565349095608585,
                       55: 1.7611777163331161,
                       13: 1.7528220549648263,
                       27: 1.7366186258024359,
                       52: 1.732422195430825,
                       43: 1.6472619390089882,
                       59: 1.5812956471871604,
                       80: 1.4960636383763144,
                       68: 1.4001355296554245,
                       22: 1.347257591741266,
                       67: 1.310690617150464,
                       99: 1.2992113301600332,
                       62: 1.2849433840644424,
                       95: 1.26396136449272,
                       77: 1.2033634019429706,
                       35: 1.1924238571220072,
                       76: 1.174902305422834,
                       4: 1.1452327507252373,
                       94: 1.0996338682453777,
                       5: 1.0992885288817815,
                       86: 1.0335516559933502,
                       47: 1.0,
                       49: 0.9912563736603252,
                       2: 0.9837690347145164,
                       17: 0.9653855313800715,
                       75: 0.9138665382930393,
                       79: 0.8956014943519729,
                       50: 0.8928091882629436,
                       82: 0.8781612117156786,
                       14: 0.869434136198863,
                       16: 0.8060053627694533,
                       78: 0.7825855541803189,
                       30: 0.7643828338234829,
                       83: 0.7615158350873189,
                       65: 0.7277257421307538,
                       34: 0.699948558090723,
                       72: 0.6655062437997392,
                       28: 0.6361935917946145,
                       38: 0.5933966154459162,
                       69: 0.5921318005292479,
                       41: 0.5744157109733371,
                       66: 0.5554251031811426,
                       93: 0.5139429626467886,
                       19: 0.4893878096652238,
                       42: 0.4775245408043885,
                       20: 0.4494381445212448,
                       61: 0.4417820256533278,
                       70: 0.4408367505576713,
                       91: 0.43301642340779656,
                       23: 0.43252134846367,
                       45: 0.42941216792840625,
                       90: 0.4111187136753738,
                       71: 0.4073080158355863,
                       92: 0.404266710648464,
                       56: 0.4026510151556321,
                       32: 0.3486241745258764,
                       36: 0.33308271808843853,
                       64: 0.3293766614665665,
                       100: 0.30493770777046764,
                       15: 0.2997460533286469,
                       3: 0.26597983827252,
                       48: 0.2479611722429104,
                       60: 0.2381913660976317,
                       63: 0.20104329046098846,
                       73: 0.18810984088035806,
                       12: 0.17373078876272166,
                       84: 0.147893870965909,
                       21: 0.1446076858743876,
                       29: 0.1315428715671936,
                       25: 0.09002235624011852,
                       24: 0.038581009817193646,
                       85: 0.0185808207815247}


def BagLoss(rel_logits, rel_label, loss_weight):
    # criterion = nn.CrossEntropyLoss(loss_weight.to(rel_logits.device))
    criterion = nn.CrossEntropyLoss()
    # print(f'In loss {rel_logits.unsqueeze(0).shape} {rel_label.shape}')
    loss = criterion(rel_logits.unsqueeze(0), rel_label)
    if loss_w_t != -1.0:
        loss = loss / (loss_weight_mapping[rel_label[0].item()]) ** loss_w_t
    return loss


def instanceMaxHead(pair_feats, classifier):
    shard_size = 1000
    num_shard = math.ceil(pair_feats.shape[0] / shard_size)
    scores = []
    for i in range(num_shard):
        logits = classifier(pair_feats[shard_size * i: shard_size * (i + 1)])  # (shard_size, 101)
        scores.append(logits.softmax(-1))
    scores = torch.cat(scores, dim=0).max(0)[0]
    return scores


def forward_bag_pair_as_unit(model, label, input_ids, token_type_ids, attention_mask, img_feats,
                             object_box_lists, object_name_positions_lists, training, attention_label):
    # get output feat for images in this bag
    shard_size = 50 if training else 50
    num_shard = math.ceil(input_ids.shape[0] / shard_size)
    outputs = []

    for i in range(num_shard):
        # [ [CLS]-emb, tok-0-emb, tok-1-emb, ..., obj-0-emb, obj-1-emb, ...]
        # (BagSize, SeqLength, 768)
        outputs.append(model.bert(input_ids[shard_size * i: shard_size * (i + 1)],
                                  token_type_ids=token_type_ids[shard_size * i: shard_size * (i + 1)],
                                  attention_mask=attention_mask[shard_size * i: shard_size * (i + 1)],
                                  img_feats=img_feats[shard_size * i: shard_size * (i + 1)])[0])
    sequence_output = torch.cat(outputs, dim=0)

    num_object_of_images = [len(lst) for lst in object_box_lists]
    img_hidden = model._get_image_hidden(sequence_output, num_object_of_images)

    assert len(object_name_positions_lists) == len(num_object_of_images)
    object_name_hidden = model._get_obj_name_hidden(sequence_output, object_name_positions_lists)

    pair_feat = []
    pair_attention_label = []
    for image_idx in range(len(object_box_lists)):
        sub_img_feat = img_hidden[image_idx][0]
        sub_name_feat = object_name_hidden[image_idx][0]
        sub_feat = torch.cat([sub_img_feat, sub_name_feat], dim=-1)

        obj_img_feat = img_hidden[image_idx][1]
        obj_name_feat = object_name_hidden[image_idx][1]
        obj_feat = torch.cat([obj_img_feat, obj_name_feat], dim=-1)
        pair_feat.append(torch.cat([sub_feat, obj_feat], dim=-1))
        pair_attention_label.append(attention_label[image_idx])

    pair_feat = torch.stack(pair_feat)
    pair_attention_label = torch.tensor(pair_attention_label)

    bag_logits, attention_loss = model.head(pair_feat, label, pair_attention_label)

    return bag_logits, attention_loss


sorted_test_image_ids = {}


def process_image_ids(image_ids_raw, pair_image_idx, sorted_pair_belonging_image_idxs):
    sorted_pair_belonging_image_ids = [image_ids_raw[pair_image_idx[x]] for x in sorted_pair_belonging_image_idxs]
    sorted_image_ids = []
    collected_ids = set()
    for i in sorted_pair_belonging_image_ids:
        if i in collected_ids:
            continue
        collected_ids.add(i)
        sorted_image_ids.append(i)
    return sorted_image_ids


def select_images(model, bag_labels, bag_input_ids, bag_token_type_ids, bag_attention_mask, bag_img_feats,
                  bag_object_box_lists, bag_object_name_positions_lists, bag_head_obj_idxs_list, bag_tail_obj_idxs_list,
                  training, bag_image_ids_list, bag_key_list):
    all_selected_images = []
    all_pair_feat = []
    all_pair_img_idx = []
    all_pair_scores = []
    with torch.no_grad():
        # process one bag at each iteration
        for bag_idx in range(len(bag_labels)):
            pair_feat = []
            label = bag_labels[bag_idx]
            key = bag_key_list[bag_idx]
            input_ids = bag_input_ids[bag_idx]
            token_type_ids = bag_token_type_ids[bag_idx]
            attention_mask = bag_attention_mask[bag_idx]
            img_feats = bag_img_feats[bag_idx]
            object_box_lists = bag_object_box_lists[bag_idx]
            object_name_positions_lists = bag_object_name_positions_lists[bag_idx]
            head_obj_idxs_list = bag_head_obj_idxs_list[bag_idx]
            tail_obj_idxs_list = bag_tail_obj_idxs_list[bag_idx]
            image_ids_raw = bag_image_ids_list[bag_idx]

            # print(f'input_ids.shape={input_ids.shape}, img_feat.shape={img_feats.shape}')

            # get output feat for images in this bag
            shard_size = 400
            num_shard = math.ceil(input_ids.shape[0] / shard_size)
            outputs = []
            for i in range(num_shard):
                # [ [CLS]-emb, tok-0-emb, tok-1-emb, ..., obj-0-emb, obj-1-emb, ...]
                # (BagSize, SeqLength, 768)
                outputs.append(model.bert(input_ids[shard_size * i: shard_size * (i + 1)],
                                          token_type_ids=token_type_ids[shard_size * i: shard_size * (i + 1)],
                                          attention_mask=attention_mask[shard_size * i: shard_size * (i + 1)],
                                          img_feats=img_feats[shard_size * i: shard_size * (i + 1)])[0])
            sequence_output = torch.cat(outputs, dim=0)

            num_object_of_images = [len(lst) for lst in object_box_lists]
            img_hidden = model._get_image_hidden(sequence_output, num_object_of_images)

            assert len(object_name_positions_lists) == len(num_object_of_images)
            object_name_hidden = model._get_obj_name_hidden(sequence_output, object_name_positions_lists)

            pair_image_idx = []
            for image_idx in range(len(object_box_lists)):
                sub_img_feat = torch.stack([img_hidden[image_idx][x] for x in head_obj_idxs_list[image_idx]])
                sub_name_feat = torch.stack([object_name_hidden[image_idx][x] for x in head_obj_idxs_list[image_idx]])
                sub_feat = torch.cat([sub_img_feat, sub_name_feat], dim=-1)

                obj_img_feat = torch.stack([img_hidden[image_idx][x] for x in tail_obj_idxs_list[image_idx]])
                obj_name_feat = torch.stack([object_name_hidden[image_idx][x] for x in tail_obj_idxs_list[image_idx]])
                obj_feat = torch.cat([obj_img_feat, obj_name_feat], dim=-1)

                for s in range(len(sub_feat)):
                    for o in range(len(obj_feat)):
                        if head_obj_idxs_list[image_idx][0] == tail_obj_idxs_list[image_idx][0] and s == o:
                            continue
                        pair_feat.append(torch.cat([sub_feat[s], obj_feat[o]], dim=-1))
                        pair_image_idx.append(image_idx)

            pair_feat = torch.stack(pair_feat)  # N, H
            pair_scores = model.classifier(pair_feat).softmax(-1)  # N, 101
            all_pair_scores.append(pair_scores)

            if training:
                sorted_indices = pair_scores[:, label].sort(descending=True)[1].tolist()
                assert len(sorted_indices) == len(pair_feat)

                selected_image_idxs = set()
                for i in sorted_indices:
                    selected_image_idxs.add(pair_image_idx[i])
                    if len(selected_image_idxs) == select_size:
                        break
                all_selected_images.append(selected_image_idxs)
            else:
                all_label_selected_image_idxs = []
                sorted_test_image_ids[key] = {}
                for label in range(101):
                    sorted_indices = pair_scores[:, label].sort(descending=True)[1].tolist()
                    sorted_test_image_ids[key][label] = process_image_ids(image_ids_raw, pair_image_idx, sorted_indices)
                    assert len(sorted_indices) == len(pair_feat)

                    selected_image_idxs = set()
                    for i in sorted_indices:
                        selected_image_idxs.add(pair_image_idx[i])
                        if len(selected_image_idxs) == select_size:
                            break
                    all_label_selected_image_idxs.append(selected_image_idxs)
                all_selected_images.append(all_label_selected_image_idxs)
                all_pair_feat.append(pair_feat)
                all_pair_img_idx.append(pair_image_idx)

                open(attention_analysis_file(), 'a').write(f'{key}\n'
                                                           f'{pair_image_idx}\n')

    return all_selected_images, all_pair_feat, all_pair_img_idx, all_pair_scores


class BagModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BagModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)

        self.dropout = nn.Dropout(0.1)

        feat_size = config.hidden_size * 4
        num_cls = 101

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, feat_size * 2),
            nn.ReLU(),
            nn.Linear(feat_size * 2, num_cls)
        )

        if head == 'att':
            self.head = BagAttention(feat_size, self.classifier)
        elif head == 'avg':
            self.head = BagAverage(feat_size, self.classifier)
            print('Use Average bag head')
        elif head == 'origin_att':
            self.head = BagOriginAttention(feat_size, self.classifier)
        elif head == 'one':
            self.head = BagOne(feat_size, self.classifier)
        else:
            raise NotImplemented

    def forward(self, bag_input_ids, bag_token_type_ids=None, bag_attention_mask=None,
                position_ids=None, head_mask=None, bag_img_feats=None, bag_object_box_lists=None,
                bag_object_name_positions_lists=None, bag_head_obj_idxs_list=None, bag_tail_obj_idxs_list=None,
                bag_labels=None, attention_label_list=None, bag_image_ids_list=None, bag_key_list=None,
                preload_ids_list=None):
        logits_list = []
        loss_list = []

        # process one bag at each iteration
        for bag_idx in range(len(bag_labels)):
            label = bag_labels[bag_idx]
            input_ids = bag_input_ids[bag_idx]
            token_type_ids = bag_token_type_ids[bag_idx]
            attention_mask = bag_attention_mask[bag_idx]
            img_feats = bag_img_feats[bag_idx]
            object_box_lists = bag_object_box_lists[bag_idx]
            object_name_positions_lists = bag_object_name_positions_lists[bag_idx]
            attention_label = attention_label_list[bag_idx]

            if self.training:
                bag_logits, attention_loss = forward_bag_pair_as_unit(
                    model=self, label=label, input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=attention_mask, img_feats=img_feats, object_box_lists=object_box_lists,
                    object_name_positions_lists=object_name_positions_lists, attention_label=attention_label,
                    training=True)
                bag_loss = BagLoss(bag_logits, label.unsqueeze(0), None)

                w = attention_w if head == 'att' else 0
                sum_loss = w * attention_loss + (1 - w) * bag_loss
                loss_list.append(sum_loss)
            else:
                bag_logits, attention_loss = forward_bag_pair_as_unit(
                    model=self, label=label, input_ids=input_ids, token_type_ids=token_type_ids,
                    attention_mask=attention_mask, img_feats=img_feats, object_box_lists=object_box_lists,
                    object_name_positions_lists=object_name_positions_lists, attention_label=attention_label,
                    training=True)
                logits_list.append(bag_logits)

        return sum(loss_list) / len(loss_list) if self.training else torch.stack(logits_list)

    def _get_image_hidden(self, sequence_output, num_obj_of_images):
        outputs = []
        for seq, num_obj in zip(sequence_output, num_obj_of_images):
            outputs.append(seq[70:70 + num_obj])
        return outputs

    def _get_obj_name_hidden(self, sequence_output, object_name_positions_lists):
        outputs = []
        for seq, object_name_positions in zip(sequence_output, object_name_positions_lists):
            name_feats = []
            for object_name_pos in object_name_positions:
                object_name_feat = seq[object_name_pos].sum(dim=0) / len(object_name_pos)
                name_feats.append(object_name_feat)
            outputs.append(torch.stack(name_feats))
        return outputs
