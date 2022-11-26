# Copyright (c) 2022 THUNLP Lab Tsinghua University. Licensed under the MIT license.
# This file defines training and evaluation code of CLEVER
# Author: Tianyu Yu
# Date: 2022-09

import argparse
import pickle
import os
import time
import os.path as op
import numpy as np
import random

import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.optim as optim
from tqdm import tqdm

from oscar.datasets.finetune_dataset import BagDatasetPairAsUnit
from oscar.modeling.modeling_clever import BagModel
from oscar.utils.logger import setup_logger
from oscar.utils.misc import (mkdir, set_seed)
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.utils.comm import all_gather, gather_on_master, reduce_dict, get_world_size, get_rank, is_main_process
from oscar.utils.save_model import save_model
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer

import oscar.modeling.modeling_clever as modeling_gre


# base_dir = f'/data_local/yutianyu'


# base_dir = f'/data/private/yutianyu'


class WarmupReduceLROnPlateau(object):
    def __init__(
            self,
            optimizer,
            gamma=0.5,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
            patience=2,
            threshold=1e-4,
            cooldown=1,
            logger=None,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.stage_count = 0
        self.best = -1e12
        self.num_bad_epochs = 0
        self.under_cooldown = self.cooldown
        self.logger = logger

        # The following code is copied from Pytorch=1.2.0
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        warmup_factor = 1
        # during warming up
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #
        return [
            base_lr
            * warmup_factor
            * self.gamma ** self.stage_count
            for base_lr in self.base_lrs
        ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # The following part is modified from ReduceLROnPlateau
        if metrics is None:
            # not conduct validation yet
            pass
        else:
            # s = '=' * 40
            # print(f'{s} Try Decay {s}')
            if float(metrics) > (self.best + self.threshold):
                self.best = float(metrics)
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.under_cooldown > 0:
                self.under_cooldown -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs >= self.patience:
                # print(f'{s} Do Decay {s}')
                if self.logger is not None:
                    self.logger.info("Trigger Schedule Decay, RL has been reduced by factor {}".format(self.gamma))
                self.stage_count += 1  # this will automatically decay the learning rate
                self.under_cooldown = self.cooldown
                self.num_bad_epochs = 0

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


logger = ...


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
    (bag_key_list, bag_image_feats_list, bag_input_ids_list, bag_input_mask_list, bag_segment_ids_list,
     bag_object_boxes_list,
     bag_object_classes_list, bag_object_name_positions_list,
     bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_label_list,
     attention_label_list, bag_image_ids_list, preload_ids_list) = new_batch

    bag_labels = torch.stack(bag_label_list, 0)
    return ((bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list, bag_image_ids_list,
             preload_ids_list),
            (bag_image_feats_list, bag_input_ids_list, bag_input_mask_list, bag_segment_ids_list))


def build_test_dataset(bag_data_file, split, tokenizer, args):
    return BagDatasetPairAsUnit(args.train_dir, bag_data_file, split, tokenizer=tokenizer, args=args, shuffle=False)


def build_train_dataset(bag_data_file, split, tokenizer, args):
    return BagDatasetPairAsUnit(args.train_dir, bag_data_file, split, tokenizer=tokenizer, args=args, shuffle=False)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, rank=get_rank(),
                                                               num_replicas=get_world_size())
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(args, bag_data_file, split, tokenizer, is_distributed=True, is_train=False):
    if is_train:
        dataset = build_train_dataset(bag_data_file, split, tokenizer, args)
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
        dataset = build_test_dataset(bag_data_file, split, tokenizer, args)
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        collate_fn = collate

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return data_loader


def eval(args, model, data_loader, tokenizer, desc='Eval'):
    world_all_scores = {}
    model.eval()
    pred_results = {}

    loader_time = 0
    forward_time = 0
    recall_time = 0

    loader_start_time = time.time()
    rank = get_rank()
    start_time = time.time()
    all_results = []
    for step, (label_list, batch) in tqdm(enumerate(data_loader), desc=desc, disable=(rank not in [0, -1])):
        (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
         bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
         bag_image_ids_list, preload_ids_list) = label_list
        batch = tuple([x.to(args.device) for x in t] for t in batch)
        img_feat, input_ids, input_mask, segment_ids = batch

        loader_end_time = time.time()
        loader_time += loader_end_time - loader_start_time

        # eval
        forward_start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(input_ids, segment_ids, input_mask, bag_img_feats=img_feat,
                               bag_object_box_lists=bag_object_boxes_list,
                               bag_object_name_positions_lists=bag_object_name_positions_list,
                               bag_head_obj_idxs_list=bag_head_obj_idxs_list,
                               bag_tail_obj_idxs_list=bag_tail_obj_idxs_list,
                               bag_labels=bag_labels, attention_label_list=attention_label_list,
                               bag_image_ids_list=bag_image_ids_list, bag_key_list=bag_key_list,
                               preload_ids_list=preload_ids_list)

        forward_end_time = time.time()
        forward_time += forward_end_time - forward_start_time

        recall_start_time = time.time()

        all_logits = torch.cat(all_gather(logits.cpu()), dim=0)
        all_pair = sum(all_gather(bag_key_list), [])

        for bag_key, logits in zip(all_pair, all_logits):
            logits = logits.cpu().numpy()
            assert len(logits) == 101, f'{len(logits)}'  # NOTE: change it if necessary
            for rel_id in range(len(logits)):
                if rel_id == 0:
                    continue
                all_results.append({
                    'class_pair': bag_key.split('#'),
                    'relation': rel_id,
                    'score': logits[rel_id]
                })

        recall_end_time = time.time()
        recall_time += recall_end_time - recall_start_time

        loader_start_time = time.time()
        t = time.time() - start_time
    result = data_loader.dataset.eval(all_results)
    return result


def write_scalar(writer, name, v, step):
    if writer is None:
        return
    writer.add_scalar(name, v, step)


def write_pr_curve(writer, name, labels, predictions, step):
    if writer is None:
        return
    writer.add_pr_curve(name, labels, predictions, step)


def train(args, train_loader, val_loader, test_loader, model, scheduler, optimizer, tokenizer, config,
          writer=None):
    rank = get_rank()
    scaler = torch.cuda.amp.GradScaler()

    best_score = 0
    global_step = 0
    iteration = 0
    log_loss = 0

    pre_eval = True
    if args.head == 'max':
        val_loader = test_loader
    if pre_eval:
        val_rtn = eval(args, model, val_loader, tokenizer, desc=f'Eval Epoch-{0}/{config.num_train_steps}')

        best_score = val_rtn['auc'] + args.mAUC_weight * val_rtn['macro_auc']
        write_scalar(writer, 'val_AUC+mAUC(weighted)', best_score, iteration)
        write_scalar(writer, 'val_auc', val_rtn['auc'], iteration)
        write_scalar(writer, 'val_macro_auc', val_rtn['macro_auc'], iteration)
        write_scalar(writer, 'val_max_micro_f1', val_rtn['max_micro_f1'], iteration)
        write_scalar(writer, 'val_max_macro_f1', val_rtn['max_macro_f1'], iteration)
        write_scalar(writer, 'val_p@2%', val_rtn['p@2%'], iteration)
        write_scalar(writer, 'val_mp@2%', val_rtn['mp@2%'], iteration)
        write_pr_curve(writer, 'val_pr_curve', val_rtn['pr_curve_labels'], val_rtn['pr_curve_predictions'],
                       iteration)

        logger.info(f'Step-0 auc: {val_rtn["auc"]:.4f}, m_auc: {val_rtn["macro_auc"]:.4f}, '
                    f'micro-f1:{val_rtn["max_micro_f1"]:.4f}, '
                    f'macro-f1:{val_rtn["max_macro_f1"]:.4f}, p@2%:{val_rtn["p@2%"]:.4f}, '
                    f'mp@2%:{val_rtn["mp@2%"]:.4f}')
    if args.head == 'max':
        print(f'End eval for VRD-{args.head} evaluation')
        exit()

    for epoch in range(args.num_train_epochs):
        desc = f'Training Epoch-{epoch + 1}/{args.num_train_epochs}'
        for step, (label_list, batch) in tqdm(enumerate(train_loader), desc=desc, disable=(rank not in [0, -1])):
            global_step += 1
            model.train()
            (bag_key_list, bag_object_boxes_list, bag_object_classes_list, bag_object_name_positions_list,
             bag_head_obj_idxs_list, bag_tail_obj_idxs_list, bag_labels, attention_label_list,
             bag_image_ids_list, preload_ids_list) = label_list
            batch = tuple([x.to(args.device) for x in t] for t in batch)
            img_feat, input_ids, input_mask, segment_ids = batch

            with torch.cuda.amp.autocast():
                loss = model(input_ids, segment_ids, input_mask, bag_img_feats=img_feat,
                             bag_object_box_lists=bag_object_boxes_list,
                             bag_object_name_positions_lists=bag_object_name_positions_list,
                             bag_head_obj_idxs_list=bag_head_obj_idxs_list,
                             bag_tail_obj_idxs_list=bag_tail_obj_idxs_list,
                             bag_labels=bag_labels, attention_label_list=attention_label_list,
                             bag_image_ids_list=bag_image_ids_list, bag_key_list=bag_key_list,
                             preload_ids_list=preload_ids_list)

            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            # scaler.scale(loss).backward()
            log_loss += (loss / args.gradient_accumulation_steps).item()

            val_result = None

            if global_step % args.gradient_accumulation_steps == 0:
                iteration += 1

                write_scalar(writer, 'train_loss', log_loss, iteration)
                write_scalar(writer, 'learning rate', optimizer.param_groups[-1]["lr"], iteration)
                log_loss = 0

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if iteration and (iteration % 200 == 0):
                    result = eval(args, model, val_loader, tokenizer, desc=f'Eval step {iteration}')

                    epoch_score = result['auc'] + args.mAUC_weight * result['macro_auc']
                    val_result = epoch_score
                    write_scalar(writer, 'val_AUC+mAUC(weighted)', epoch_score, iteration)
                    write_scalar(writer, 'val_auc', result['auc'], iteration)
                    write_scalar(writer, 'val_macro_auc', result['macro_auc'], iteration)
                    write_scalar(writer, 'val_max_micro_f1', result['max_micro_f1'], iteration)
                    write_scalar(writer, 'val_max_macro_f1', result['max_macro_f1'], iteration)
                    write_scalar(writer, 'val_p@2%', result['p@2%'], iteration)
                    write_scalar(writer, 'val_mp@2%', result['mp@2%'], iteration)
                    write_pr_curve(writer, 'val_pr_curve', result['pr_curve_labels'], result['pr_curve_predictions'],
                                   iteration)

                    logger.info(f'Step-{iteration} auc: {result["auc"]:.4f}, m_auc: {result["macro_auc"]:.4f}, '
                                f'micro-f1:{result["max_micro_f1"]:.4f}, '
                                f'macro-f1:{result["max_macro_f1"]:.4f}, p@2%:{result["p@2%"]:.4f}, '
                                f'mp@2%:{result["mp@2%"]:.4f}')
                    if epoch_score >= best_score:
                        best_score = epoch_score
                        save_model(args, model, tokenizer, logger, save_mode="best")

                        # Run Test Split
                        result = eval(args, model, test_loader, tokenizer, desc=f'Test step-{iteration}')

                        write_scalar(writer, 'test_auc', result['auc'], iteration)
                        write_scalar(writer, 'test_macro_auc', result['macro_auc'], iteration)
                        write_scalar(writer, 'test_max_micro_f1', result['max_micro_f1'], iteration)
                        write_scalar(writer, 'test_max_macro_f1', result['max_macro_f1'], iteration)
                        write_scalar(writer, 'test_p@2%', result['p@2%'], iteration)
                        write_scalar(writer, 'test_mp@2%', result['mp@2%'], iteration)
                        write_pr_curve(writer, 'test_pr_curve', result['pr_curve_labels'],
                                       result['pr_curve_predictions'],
                                       iteration)
                        pickle.dump(result['results'], open(f'{args.output_dir}/best_results.pkl', 'wb'))

            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= 3:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                return model
    return model


def build_optimizer(model, opts):
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
    parser.add_argument("--num_workers", default=3, type=int, help="Workers in dataloader.")
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
    parser.add_argument('--sfmx_t', type=float, default=18)
    parser.add_argument('--loss_w_t', type=float, default=-1.0)
    parser.add_argument('--attention_w', type=float, default=0.3)
    parser.add_argument('--head', type=str, default='att')
    parser.add_argument('--select_size', type=int, default=10)
    parser.add_argument('--real_bag_size', type=int, default=10)
    parser.add_argument('--mAUC_weight', type=int, default=1)
    parser.add_argument('--pretrained_weight', type=str, default='')
    parser.add_argument('--VRD_weight', type=str, default='')

    args = parser.parse_args()

    import oscar.datasets.finetune_dataset as finetune_dataset
    finetune_dataset.real_bag_size = args.real_bag_size
    modeling_gre.sfmx_t = args.sfmx_t
    modeling_gre.attention_w = args.attention_w
    modeling_gre.head = args.head
    modeling_gre.select_size = args.select_size
    modeling_gre.loss_w_t = args.loss_w_t

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
    config_class, model_class, tokenizer_class = BertConfig, BagModel, BertTokenizer
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

    print(f'Load from {checkpoint}')

    if modeling_gre.head == 'att' and args.pretrained_weight:
        print(f'Load pretrained weight from {args.pretrained_weight}')
        model.load_state_dict(torch.load(args.pretrained_weight, map_location='cpu'), strict=False)
    elif modeling_gre.head in ['avg', 'max'] and args.VRD_weight:
        print(f'Load pretrained weight from {args.VRD_weight}')
        model.load_state_dict(torch.load(args.VRD_weight, map_location='cpu'), strict=False)

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

    if local_rank != 0:
        torch.distributed.barrier()

    train_loader = make_data_loader(
        args, f'{args.train_dir}/train_bag_data.json', 'train', tokenizer,
        is_distributed=args.distributed,
        is_train=True)
    test_loader = make_data_loader(
        args, f'{args.train_dir}/test_bag_data.json', 'test', tokenizer,
        is_distributed=args.distributed,
        is_train=False)
    val_loader = make_data_loader(
        args, f'{args.train_dir}/val_bag_data.json', 'val', tokenizer, is_distributed=args.distributed,
        is_train=False)

    if local_rank == 0:
        torch.distributed.barrier()

    print(f'Train-size: {len(train_loader.dataset)} bags, {len(train_loader)} batches, '
          f'Test-size: {len(test_loader.dataset)} bags, {len(test_loader)} batches')

    config.learning_rate = args.learning_rate
    config.weight_decay = 0.01
    config.betas = [0.9, 0.98]
    config.num_train_steps = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    config.lr_mul = 1.0
    config.warmup_steps = int(0.1 * config.num_train_steps)
    optimizer = build_optimizer(model, config)

    print(f'Warm up step {args.warmup_steps}')
    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        0.1,
        warmup_factor=0.1,
        warmup_iters=args.warmup_steps,
        warmup_method='linear',
        patience=3,
        threshold=0.001,
        cooldown=0,
        logger=logger,
    )

    model = train(args, train_loader, val_loader, test_loader, model, scheduler, optimizer, tokenizer, config,
                  writer=writer)
    save_model(args, model, tokenizer, logger, save_mode="final")


if __name__ == "__main__":
    main()
