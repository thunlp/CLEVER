# Copyright (c) 2022 THUNLP Lab Tsinghua University. Licensed under the MIT license.
# This file defines training and evaluation of VRD baselines for CKE.
# Author: Tianyu Yu
# Date: 2022-09

import argparse
import json
import os
import time
import pathlib
import os.path as op
import numpy as np

import torch
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

from oscar.datasets.finetune_dataset import NormalFinetuneDataset
from oscar.modeling.modeling_clever import VRDBaselineModel
from oscar.utils.logger import setup_logger
from oscar.utils.misc import (mkdir, set_seed)
from oscar.utils.optim_sched import get_lr_sched
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.utils.comm import all_gather, gather_on_master, reduce_dict, get_world_size, get_rank, is_main_process
from oscar.utils.save_model import save_model
from torch.utils.tensorboard import SummaryWriter
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule


## Start Instance-Level Eval

def cal_pred_cls_ng_recall_img(scores, num_box, relations):
    relations = [tuple(r) for r in relations]
    time_st = time.time()
    ks = [20, 50, 100]
    assert len(scores) == num_box * (num_box - 1), f'{len(scores)}, {num_box}'

    time_collect_predictions_st = time.time()
    idx = 0
    predictions = []
    scores = scores.cpu().numpy()
    for s in range(num_box):
        for o in range(num_box):
            if o == s:
                continue
            for p in range(1, 101):
                predictions.append((s, o, p, scores[idx][p]))
            idx += 1

    time_collect_predictions = time.time() - time_collect_predictions_st
    time_sort_st = time.time()

    sorted_predictions = sorted(predictions, key=lambda x: x[-1], reverse=True)
    # print(f'Collected {len(sorted_predictions)} predictions for {num_box} boxes')

    time_sort = time.time() - time_sort_st
    time_cal_st = time.time()

    correct = 0
    recalls = []

    num_facts_of_predicates = defaultdict(int)
    for f in relations:
        p = f[2]
        num_facts_of_predicates[p] += 1
    recalls_of_predicate = {p: [] for p in num_facts_of_predicates}
    correct_of_predicates = {p: 0 for p in num_facts_of_predicates}
    num_seen_of_predicates = {p: 0 for p in num_facts_of_predicates}

    # print(f'check pred:{sorted_predictions[:3]} and gt:{relations}')
    for i in range(len(sorted_predictions)):
        idx = i
        s, o, p = sorted_predictions[idx][:3]
        if (s, o, p) in relations:
            correct += 1
            correct_of_predicates[p] += 1
        recalls.append(correct / len(relations))

        for p in recalls_of_predicate:
            recalls_of_predicate[p].append(correct_of_predicates[p] / max(1, num_facts_of_predicates[p]))
    assert recalls[-1] == 1

    time_cal = time.time() - time_cal_st
    t = time.time() - time_st
    # print(f'Calculate recall for image with {num_box} costs {int(t + 0.5):4d}s, '
    #       f'collect:{int(time_collect_predictions / t * 100 + 0.5):3d} '
    #       f'sort:{int(time_sort / t * 100 + 0.5):3d} '
    #       f'cal:{int(time_cal / t * 100 + 0.5):3d} ')
    # print(len(recalls), recalls[0], recalls[len(recalls) // 2], recalls[len(recalls) - 1])
    recalls = {k: recalls[k - 1] for k in ks}
    predicate_recalls = {k: {p: recalls_of_predicate[p][k - 1]
                             for p in recalls_of_predicate if num_facts_of_predicates[p]}
                         for k in ks}
    # print(predicate_recalls)
    # exit()
    # print(recalls)
    return recalls, predicate_recalls

def cal_pred_cls_recall_img(scores, num_box, relations):
    relations = [tuple(r) for r in relations]
    ks = [20, 50, 100]
    assert len(scores) == num_box * (num_box - 1), f'{len(scores)}, {num_box}'

    idx = 0
    predictions = []
    scores = scores.cpu().numpy()
    for s in range(num_box):
        for o in range(num_box):
            if o == s:
                continue
            p = 1 + np.argmax(scores[idx][1:])
            predictions.append((s, o, p, scores[idx][p]))
            idx += 1

    sorted_predictions = sorted(predictions, key=lambda x: x[-1], reverse=True)

    correct = 0
    recalls = []

    num_facts_of_predicates = defaultdict(int)
    for f in relations:
        p = f[2]
        num_facts_of_predicates[p] += 1
    recalls_of_predicate = {p: [] for p in num_facts_of_predicates}
    correct_of_predicates = {p: 0 for p in num_facts_of_predicates}

    for i in range(len(sorted_predictions)):
        idx = i
        s, o, p = sorted_predictions[idx][:3]
        if (s, o, p) in relations:
            correct += 1
            correct_of_predicates[p] += 1
        recalls.append(correct / len(relations))

        for p in recalls_of_predicate:
            recalls_of_predicate[p].append(correct_of_predicates[p] / max(1, num_facts_of_predicates[p]))

    recalls = {k: recalls[min(k - 1, len(recalls) - 1)] for k in ks}
    predicate_recalls = {k: {p: recalls_of_predicate[p][min(k - 1, len(recalls_of_predicate[p]) - 1)]
                             for p in recalls_of_predicate if num_facts_of_predicates[p]}
                         for k in ks}
    return recalls, predicate_recalls


## End Instance-Level Eval


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
                       'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank


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


def collate(batch):
    new_batch = list(zip(*batch))
    (img_keys, img_feat_list, input_ids_list, input_mask_list, segment_ids_list,
     object_box_list, object_cls_list, object_name_positions_list, relations_list, labels_list, pairs_list) = new_batch

    img_feat_list = torch.stack(img_feat_list, 0)
    input_ids_list = torch.stack(input_ids_list, 0)
    input_mask_list = torch.stack(input_mask_list, 0)
    segment_ids_list = torch.stack(segment_ids_list, 0)
    return ((img_keys, object_box_list, object_cls_list, object_name_positions_list,
             relations_list, labels_list, pairs_list),
            (img_feat_list, input_ids_list, input_mask_list, segment_ids_list))


def build_test_dataset(data_file, tokenizer, args):
    if not op.isfile(data_file):
        data_file = op.join(args.test_dir, data_file)
        assert op.isfile(data_file), f'{data_file}'
    return NormalFinetuneDataset(data_file, tokenizer=tokenizer, args=args, shuffle=False)


def build_train_dataset(data_file, tokenizer, args):
    if not op.isfile(data_file):
        data_file = op.join(args.train_dir, data_file)
        assert op.isfile(data_file), f'{data_file}'
    return NormalFinetuneDataset(data_file, tokenizer=tokenizer, args=args, shuffle=False)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, rank=get_rank(),
                                                               num_replicas=get_world_size())
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, data_file, tokenizer, is_distributed=True, is_train=False):
    if is_train:
        dataset = build_train_dataset(data_file, tokenizer, args)
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        collate_fn = collate
    else:
        dataset = build_test_dataset(data_file, tokenizer, args)
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        collate_fn = collate

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader


def eval(args, model, data_loader, tokenizer, desc='Eval'):
    world_all_scores = {}
    model.eval()
    ks = [20, 50, 100]
    process_recall_at_ks = {k: {} for k in ks}
    process_predicate_recall_at_ks = {k: {} for k in ks}

    loader_time = 0
    forward_time = 0
    recall_time = 0

    loader_start_time = time.time()
    rank = get_rank()
    start_time = time.time()
    for step, (label_list, batch) in tqdm(enumerate(data_loader), desc=desc, disable=(rank not in [0, -1])):
        (img_keys, object_box_lists, object_cls_lists, object_name_positions_lists, relations_lists,
         rel_labels_list, pairs_list) = label_list
        batch = tuple(t.to(args.device) for t in batch)
        img_feat, input_ids, input_mask, segment_ids = batch

        loader_end_time = time.time()
        loader_time += loader_end_time - loader_start_time

        # eval
        forward_start_time = time.time()
        with torch.no_grad():
            loss, scores, labels = model(input_ids, segment_ids, input_mask, img_feats=img_feat,
                                         relation_list=relations_lists, object_box_lists=object_box_lists,
                                         object_name_positions_lists=object_name_positions_lists,
                                         rel_labels_list=rel_labels_list, pairs_list=pairs_list)
        split_labels = labels.split([len(lst) * (len(lst) - 1) for lst in object_cls_lists])
        split_scores = scores.split([len(lst) * (len(lst) - 1) for lst in object_cls_lists])

        forward_end_time = time.time()
        forward_time += forward_end_time - forward_start_time

        recall_start_time = time.time()
        for img_key, img_scores, img_obj_boxes, img_relations in zip(img_keys, split_scores, object_box_lists,
                                                                     relations_lists):
            # image_recalls, image_predicate_recalls = cal_pred_cls_ng_recall_img(img_scores, len(img_obj_boxes),
            #                                                                     img_relations)
            image_recalls, image_predicate_recalls = cal_pred_cls_recall_img(img_scores, len(img_obj_boxes),
                                                                                img_relations)
            world_all_scores[img_key] = {
                'scores': img_scores.cpu(),
                'relations': img_relations
            }
            for k in ks:
                process_recall_at_ks[k][img_key] = image_recalls[k]
                process_predicate_recall_at_ks[k][img_key] = image_predicate_recalls[k]

        recall_end_time = time.time()
        recall_time += recall_end_time - recall_start_time

        loader_start_time = time.time()
        t = time.time() - start_time
        # print(f'loader: {int(loader_time / t * 100):4d}, forward: {int(forward_time / t * 100):4d}'
        #       f'recall: {int(recall_time / t * 100):4d}' )

    # print(f'Rank-{get_rank()} eval size is {len(process_recall_at_ks[20])}, dataset size is {len(data_loader.dataset)}')
    all_recalls_at_k = all_gather(process_recall_at_ks)
    all_predicate_recalls_at_k = all_gather(process_predicate_recall_at_ks)
    all_world_scores = all_gather(world_all_scores)
    # print(f'Collect {len(all_recalls_at_k)} lists from all processes, sizes are {[len(process_data[20]) for process_data in all_recalls_at_k]}')

    all_scores = {}
    if get_rank() in [0, -1]:
        for world_scores in all_world_scores:
            all_scores.update(world_scores)

    total_recalls_at_k = {k: {} for k in ks}
    total_predicate_recalls_at_k = {k: {} for k in ks}
    for k in ks:
        for process_data in all_recalls_at_k:
            total_recalls_at_k[k].update(process_data[k])
        for process_predicate_data in all_predicate_recalls_at_k:
            total_predicate_recalls_at_k[k].update(process_predicate_data[k])
    total_recalls_at_k = {k: list(d.values()) for k, d in total_recalls_at_k.items()}

    m_recalls = {}
    m_recalls_detail = {}
    for k in total_predicate_recalls_at_k.keys():
        m_recalls_detail[k] = {}
        for p in range(1, 101):
            values = [total_predicate_recalls_at_k[k][img_key][p]
                      for img_key in total_predicate_recalls_at_k[k]
                      if p in total_predicate_recalls_at_k[k][img_key]]
            tmp_recall = np.mean(values) if len(values) else 0
            m_recalls_detail[k][p] = tmp_recall
        m_recalls[k] = np.mean(list(m_recalls_detail[k].values()))
    recalls = {k: np.array(v).sum() / len(v) for k, v in total_recalls_at_k.items()}
    assert len(total_recalls_at_k[100]) == len(
        data_loader.dataset), f'{len(total_recalls_at_k[100])}, {len(data_loader.dataset)}'
    assert len(total_predicate_recalls_at_k[100]) == len(
        data_loader.dataset), f'{len(total_predicate_recalls_at_k[100])}, {len(data_loader.dataset)}'
    if is_main_process():
        logger.info(f'Recalls: {recalls}')
        logger.info(f'mRecalls: {m_recalls}\n')
    return recalls, m_recalls, all_scores


def write_scalar(writer, name, v, step):
    if writer is None:
        return
    writer.add_scalar(name, v, step)


def train_batch(args, train_loader, test_loader, model, scheduler, optimizer, tokenizer, config, global_step,
                desc='Train',
                writer=None):
    rank = get_rank()
    for step, (label_list, batch) in tqdm(enumerate(train_loader), desc=desc, disable=(rank not in [0, -1])):
        model.train()
        (img_keys, object_box_lists, object_cls_lists, object_name_positions_lists,
         relations_lists, rel_labels_list, pairs_list) = label_list
        batch = tuple(t.to(args.device) for t in batch)
        img_feat, input_ids, input_mask, segment_ids = batch

        # schedule lr
        lr_this_step = get_lr_sched(global_step, config)
        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0 or i == 1:
                param_group['lr'] = lr_this_step * config.lr_mul
            elif i == 2 or i == 3:
                param_group['lr'] = lr_this_step
            else:
                raise ValueError()
        optimizer.zero_grad()

        loss, scores, labels = model(input_ids, segment_ids, input_mask, img_feats=img_feat,
                                     relation_list=relations_lists, object_box_lists=object_box_lists,
                                     object_name_positions_lists=object_name_positions_lists,
                                     rel_labels_list=rel_labels_list, pairs_list=pairs_list)
        split_scores = scores.split([len(pairs) for pairs in pairs_list])
        write_scalar(writer, 'train_loss', loss, global_step)
        # print('Before', model.module.bert.embeddings.position_embeddings.weight.grad)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        # print('After', model.module.bert.embeddings.position_embeddings.weight.grad)
        # exit()
    return global_step


def build_optimizer(model, opts):
    print(f'@Learning rate is {opts.learning_rate}')
    """ Re linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'classifier' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'classifier' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--test_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int,
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true',
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true',
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=8, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=5,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--sc_beam_size', type=int, default=1,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")

    parser.add_argument('--keep_ratio', type=float, default=1.0,
                        help='Low resource scenario train label keep ratio')
    parser.add_argument('--shot', type=int, default=-1,
                        help='Low resource scenario train label shot')
    parser.add_argument('--eval_period', type=int, default=1,
                        help='number of training epochs between evaluation')
    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, VRDBaselineModel, BertTokenizer
    checkpoint = args.eval_model_dir
    assert op.isdir(checkpoint)
    config = config_class.from_pretrained(checkpoint)
    config.output_hidden_states = args.output_hidden_states
    # config.hidden_dropout_prob = 0.6
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    config.loss_type = "cls"

    model = model_class.from_pretrained(checkpoint, config=config)
    model.to(args.device)

    # distributed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    logger.info(args)
    writer = None
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(
            log_dir=os.path.join(output_dir, 'summary'),
            flush_secs=10
        )

    train_loader = make_data_loader(args, "predictions_train.tsv", tokenizer, is_distributed=args.distributed,
                                    is_train=True)
    val_loader = make_data_loader(args, "predictions_val.tsv", tokenizer, is_distributed=args.distributed,
                                  is_train=False)
    test_loader = make_data_loader(args, "predictions_test.tsv", tokenizer, is_distributed=args.distributed,
                                   is_train=False)

    print(f'Train-size: {len(train_loader.dataset)} samples, {len(train_loader)} batches, '
          f'Val-size: {len(val_loader.dataset)} samples, {len(val_loader)} batches'
          f'Test-size: {len(test_loader.dataset)} samples, {len(test_loader)} batches')

    config.learning_rate = args.learning_rate
    config.weight_decay = 0.01
    config.betas = [0.9, 0.98]
    config.num_train_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    # print(f'Training steps is {config.num_train_steps} as {len(train_loader)} // {args.gradient_accumulation_steps} * {args.num_train_epochs}')
    config.lr_mul = 1.0
    config.warmup_steps = int(0.1 * config.num_train_steps)
    # print(f'Warmup steps is {config.warmup_steps}')
    optimizer = build_optimizer(model, config)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        print(f'linear schedule')
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=config.num_train_steps)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    best_mRecallAT20 = 0
    global_step = 0

    # pre-eval
    # recallAT20, mRecallAT20, prediction_result = eval(args, model, test_loader, tokenizer,
    #                                                   desc=f'Eval Epoch-{0}/{args.num_train_epochs}')

    for epoch in range(args.num_train_epochs):
        # print("epoch: {}/{}".format(epoch + 1, args.num_train_epochs))
        global_step = train_batch(args, train_loader, test_loader, model, scheduler, optimizer, tokenizer, config,
                                  global_step,
                                  desc=f'Training Epoch-{epoch + 1}/{args.num_train_epochs}', writer=writer)
        # print(f'Epoch-{epoch} finished, at step-{global_step}')
        if (epoch + 1) % args.eval_period == 0:
            recall, mRecall, prediction_result = eval(args, model, val_loader, tokenizer,
                                                      desc=f'Eval Epoch-{epoch + 1}/{args.num_train_epochs}')
            test_recall, test_mRecall, test_prediction_result = eval(args, model, test_loader, tokenizer,
                                                                     desc=f'Test Epoch-{epoch + 1}/{args.num_train_epochs}')
            recallAT20 = recall[20]
            mRecallAT20 = mRecall[20]
            test_recallAT20 = test_recall[20]
            test_mRecallAT20 = test_mRecall[20]
            write_scalar(writer, 'val_recall@20', recallAT20, (epoch + 1))
            write_scalar(writer, 'val_mrecall@20', mRecallAT20, (epoch + 1))
            write_scalar(writer, 'test_recall@20', test_recallAT20, (epoch + 1))
            write_scalar(writer, 'test_mrecall@20', test_mRecallAT20, (epoch + 1))
            write_scalar(writer, 'val_recall@50', recall[50], (epoch + 1))
            write_scalar(writer, 'val_mrecall@50', mRecall[50], (epoch + 1))
            write_scalar(writer, 'test_recall@50', test_recall[50], (epoch + 1))
            write_scalar(writer, 'test_mrecall@50', test_mRecall[50], (epoch + 1))
            if mRecallAT20 >= best_mRecallAT20:
                if is_main_process():
                    logger.info(
                        "save best model at {} step, the recallAT20 is {}, mRecallAT20 is {}".format(global_step,
                                                                                                     recallAT20,
                                                                                                     mRecallAT20))
                    logger.info(
                        "Test-recallAT20 is {}, Test-mRecallAT20 is {}".format(test_recallAT20, test_mRecallAT20))
                save_model(args, model, tokenizer, logger, save_mode="best")
                best_mRecallAT20 = mRecallAT20
            if get_rank() in [0, -1]:
                output_dir = pathlib.Path(os.path.join(args.output_dir, 'prediction_results'))
                if not output_dir.exists():
                    output_dir.mkdir()
                torch.save(test_prediction_result,
                           os.path.join(output_dir,
                                        f'Epoch-{epoch + 1}-recallAT20-{test_recallAT20:.4f}-mRecallAT20-{test_mRecallAT20}.bin'))
    save_model(args, model, tokenizer, logger, save_mode="final")


if __name__ == "__main__":
    main()
