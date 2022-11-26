import os
import json
import random
import warnings

warnings.filterwarnings('once')  # "error", "ignore", "always", "default", "module" or "once"

import tqdm
import torch
import pickle
import argparse
import numpy as np
import sklearn.metrics
import torch.utils.data
import numpy.random
from transformers.pytorch_transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from collections import defaultdict, Counter
from torch.utils.tensorboard import SummaryWriter


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


def read_split(data, ind_to_classes, ind_to_predicates, cls_m):
    items = []
    for bag_key in data:
        s_o = [cls_m[int(x)] for x in bag_key.split('#')]
        s = ind_to_classes[s_o[0]]  # vg-150 idx for sub, obj
        o = ind_to_classes[s_o[1]]
        labels = data[bag_key]['label']
        item = {
            'subject': s,
            'object': o,
            'sub_id': s_o[0],
            'obj_id': s_o[1],
            'label': labels,
            'y': torch.zeros(len(ind_to_predicates)).scatter(0, torch.tensor(labels, dtype=torch.int64), 1)
        }
        items.append(item)
    return items


class TextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, data, split, tkz=None):
        self.data = data

        self.facts = set()
        for item in self.data:
            for label in item['label']:
                if label:
                    self.facts.add((item['sub_id'], item['obj_id'], label))
        print(f'{split} facts are {len(self.facts)}')
        self.tkz = tkz

    def __getitem__(self, item):
        item = self.data[item]
        input_ids = self.tkz.encode(item['subject'] + ' ' + item['object'])
        input_ids = input_ids[:20] + [self.tkz._convert_token_to_id(self.tkz.pad_token)] * max((20 - len(input_ids)), 0)
        attention_mask = torch.tensor([(1 if _ else 0) for _ in input_ids])
        inputs = {
            'input_ids': torch.tensor(input_ids).cuda(),
            'attention_mask': attention_mask.cuda(),
        }
        labels = item['y'].cuda()
        return inputs, labels

    def __len__(self):
        return len(self.data)


def forward(model, x, y, loss_func=None):
    outputs = model(**x, labels=y)
    loss = outputs[0]
    pred_y = outputs[1]

    return pred_y, loss


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
    # print(len(num_facts_of_predicates))

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


def run_test(model, loss_func, val_dataset, val_dataloader):
    prediction_logits = []
    labels = []
    pred_result = []
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            x, y = batch
            pred_y, loss = forward(model, x, y, loss_func)
            prediction_logits.append(pred_y.cpu())

            labels.append(y)
            scores = pred_y.cpu().softmax(-1)
            for r in range(1, 101):
                pred_result.append((
                    val_dataset.data[idx]['sub_id'],
                    val_dataset.data[idx]['obj_id'],
                    r,
                    scores[0][r]
                ))
    prediction_logits = torch.cat(prediction_logits, dim=0)
    labels = torch.cat(labels, dim=0)
    # print(prediction_logits.shape, labels.shape)

    val_loss = loss_func(prediction_logits, labels.cpu())
    # print(f'val_loss: {val_loss.item():.5f}')

    # print(len(pred_result), 'results')
    sorted_pred_result = sorted(pred_result, key=lambda x: x[3], reverse=True)
    return sorted_pred_result


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_scalar(writer, name, v, step):
    if writer is None:
        return
    writer.add_scalar(name, v, step)


def write_pr_curve(writer, name, labels, predictions, step):
    if writer is None:
        return
    writer.add_pr_curve(name, labels, predictions, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default='1e-5')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--bsz', type=int, default=50)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--mAUC_weight', type=int, default=1)

    args = parser.parse_args()

    model_name = f'vanilla_ft_{args.epoch}epoch-lr={args.lr}-bsz={args.bsz}'
    writer = SummaryWriter(
        log_dir=os.path.join('./text_only_baseline_summary', model_name),
        flush_secs=10
    )
    seed_all(0)

    ind_to_classes, ind_to_predicates, _, _100_cls_idx_to_vg150_idx = load_info(f'{args.data_dir}/vg_dict.json')
    val_bag_data = json.load(open(f'{args.data_dir}/val_bag_data.json'))
    test_bag_data = json.load(open(f'{args.data_dir}/test_bag_data.json'))
    train_bag_data = json.load(open(f'{args.data_dir}/train_bag_data.json'))
    val = read_split(val_bag_data, ind_to_classes, ind_to_predicates, _100_cls_idx_to_vg150_idx)
    test = read_split(test_bag_data, ind_to_classes, ind_to_predicates, _100_cls_idx_to_vg150_idx)
    train = read_split(train_bag_data, ind_to_classes, ind_to_predicates, _100_cls_idx_to_vg150_idx)

    tkz = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    val_dataset = TextOnlyDataset(val, 'val', tkz)
    test_dataset = TextOnlyDataset(test, 'test', tkz)
    train_dataset = TextOnlyDataset(train, 'train', tkz)

    bsz = args.bsz
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    model = model.cuda()
    model.num_labels = 101
    model.config.num_labels = 101
    model.config.problem_type = 'multi_label_classification'
    model.classifier = torch.nn.Linear(model.config.hidden_size, model.num_labels).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epoch = args.epoch
    loss = 0
    last_100_loss = []
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_score = -1
    for e in range(epoch):
        tqdm_bar = tqdm.tqdm(enumerate(train_dataloader))
        for idx, batch in tqdm_bar:
            # print(e, idx)
            x, y = batch
            pred_y, loss = forward(model, x, y, loss_func)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            last_100_loss.append(loss.item())
            last_100_loss = last_100_loss[-100:]
            tqdm_bar.set_description(f'loss: {sum(last_100_loss) / len(last_100_loss):.5f}')

        sorted_pred_result = run_test(model, loss_func, val_dataset, val_dataloader)
        label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, \
        max_macro_f1, pr_curve_labels, pr_curve_predictions = cal_metrics(sorted_pred_result, val_dataset.facts)
        val_rtn = {
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
        epoch_score = val_rtn['auc'] + args.mAUC_weight * val_rtn['macro_auc']
        write_scalar(writer, 'val_AUC+mAUC(weighted)', epoch_score, e)
        write_scalar(writer, 'val_auc', val_rtn['auc'], e)
        write_scalar(writer, 'val_macro_auc', val_rtn['macro_auc'], e)
        write_scalar(writer, 'val_max_micro_f1', val_rtn['max_micro_f1'], e)
        write_scalar(writer, 'val_max_macro_f1', val_rtn['max_macro_f1'], e)
        write_scalar(writer, 'val_p@2%', val_rtn['p@2%'], e)
        write_scalar(writer, 'val_mp@2%', val_rtn['mp@2%'], e)
        write_pr_curve(writer, 'val_pr_curve', val_rtn['pr_curve_labels'], val_rtn['pr_curve_predictions'],
                       e)
        if epoch_score > best_score:
            best_score = epoch_score

            sorted_pred_result = run_test(model, loss_func, test_dataset, test_dataloader)
            label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, \
            max_macro_f1, pr_curve_labels, pr_curve_predictions = cal_metrics(sorted_pred_result, test_dataset.facts)
            test_rtn = {
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
            write_scalar(writer, 'test_auc', test_rtn['auc'], e)
            write_scalar(writer, 'test_macro_auc', test_rtn['macro_auc'], e)
            write_scalar(writer, 'test_max_micro_f1', test_rtn['max_micro_f1'], e)
            write_scalar(writer, 'test_max_macro_f1', test_rtn['max_macro_f1'], e)
            write_scalar(writer, 'test_p@2%', test_rtn['p@2%'], e)
            write_scalar(writer, 'test_mp@2%', test_rtn['mp@2%'], e)
            write_pr_curve(writer, 'test_pr_curve', test_rtn['pr_curve_labels'], test_rtn['pr_curve_predictions'],
                           e)
            print(f'Epoch-{epoch} Test auc: {test_rtn["auc"]:.4f}, m_auc: {test_rtn["macro_auc"]:.4f}, '
                  f'micro-f1:{test_rtn["max_micro_f1"]:.4f}, '
                  f'macro-f1:{test_rtn["max_macro_f1"]:.4f}, p@2%:{test_rtn["p@2%"]:.4f}, '
                  f'mp@2%:{test_rtn["mp@2%"]:.4f}')

            pickle.dump(sorted_pred_result, open(f'{model_name}.pkl', 'wb'))
