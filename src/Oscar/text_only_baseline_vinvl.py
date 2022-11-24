# Copyright (c) 2022 THUNLP Lab Tsinghua University. Licensed under the MIT license.
# This file defines the implementation of PLM-finetune baselines for CKE
# Author: Tianyu Yu
# Date: 2022-09

import json

import matplotlib.pyplot as plt
import tqdm
import torch
import numpy as np
import torch.utils.data
import sklearn.metrics
from collections import defaultdict

use_vinvl = True
prefix = 'VinVL'


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


def read_split(data, ind_to_classes, ind_to_predicates, vectors, cls_m):
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
            # 'x': torch.cat([vectors[s_o[0] - 1], vectors[s_o[1] - 1]], dim=-1),
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
        # print(f'yield item {item}')
        inputs = self.tkz(item['subject'] + ' ' + item['object'], return_tensors="pt", pad_to_max_length=True,
                          max_length=20)
        # print(item['subject'] + ' ' + item['object'], inputs)
        inputs = {k: v[0].cuda() for k, v in inputs.items()}
        # print([x.shape for x in inputs])
        # exit()
        labels = item['y'].cuda()
        # print(labels.shape)
        return inputs, labels

    def __len__(self):
        return len(self.data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 101),
        )

    def forward(self, x):
        return self.classifier(x)


def forward(model, x, y, loss_func=None):
    outputs = model(**x, labels=y)
    pred_y = outputs.logits
    loss = outputs.loss

    return pred_y, loss


from collections import Counter


def cal_metrics(sorted_results, facts, ouput_fname='', n_samples=2000, show_detail=False):
    top_n_predicates = [ind_to_predicates[x[2]] for x in sorted_pred_result[:n_samples]]
    top_n_predicate_counter = Counter(top_n_predicates)
    if show_detail:
        print(f'Top-{n_samples} results, predicate distribution:\n{top_n_predicate_counter}')

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

    class_pair_result = {}

    for i, item in enumerate(sorted_results):
        p = item[2]
        pair_key = (item[0], item[1])
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
        precisions.append(correct / (i + 1))
        recalls.append(correct / len(facts))


        for p in recalls_of_predicate:
            pr = correct_of_predicates[p] / (i + 1)
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
    np_f1_of_predicate = {p: np.array(f1) for p, f1 in f1_of_predicate.items()}
    macro_f1_list = sum(np_f1_of_predicate.values(), np.zeros(len(sorted_results))) / len(
        np_f1_of_predicate)


    macro_recall = sum(np_recall_of_predicates.values(), np.zeros(len(sorted_results))) / len(
        np_recall_of_predicates)
    macro_recall_auc = macro_recall.sum() / len(macro_recall)
    # json.dump(macro_recall.tolist(), open(ouput_fname + '-macro_recalls.json', 'w'))

    auc = sklearn.metrics.auc(x=recalls, y=precisions)
    np_prec = np.array(precisions)
    np_rec = np.array(recalls)
    max_micro_f1 = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).max()
    best_threshold = sorted_results[(2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()][3].item()

    pred_result_vec = score_vec >= best_threshold
    max_macro_f1 = sklearn.metrics.f1_score(label_vec, pred_result_vec,
                                            labels=list(range(1, 101)), average='macro')

    print(f'max_f1={max_micro_f1:.5f}, auc={auc:.5f}, max_macro_f1={max_macro_f1}')

    # json.dump(recalls, open(ouput_fname + '-micro_recalls.json', 'w'))

    fig, ax = plt.subplots(figsize=(24, 8))
    x = range(len(np_rec))
    ax.plot(x, np_rec, label='micro-recall')
    ax.plot(x, macro_recall, label='macro-recall')

    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    plt.show()

    return np_rec, np_prec, recalls_of_predicate, macro_recall, macro_recall_auc, auc, max_micro_f1, max_macro_f1

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
    print(f'val_loss: {val_loss.item():.5f}')

    # print(len(pred_result), 'results')
    sorted_pred_result = sorted(pred_result, key=lambda x: x[3], reverse=True)
    return sorted_pred_result


if __name__ == '__main__':
    ind_to_classes, ind_to_predicates, _, _100_cls_idx_to_vg150_idx = \
        load_info(
            '/data_local/yutianyu/datasets/VisualGenome/100_100_fix/VG-dicts_100_cls_100_pred_with_100_cls_to_vg_150_mapping.json')
    # vectors = obj_edge_vectors(ind_to_classes[1:], wv_dir='/data_local/yutianyu/GloVe', wv_dim=200)
    vectors = ...
    val = read_split(json.load(open('/data_local/yutianyu/datasets/VisualGenome/100_100_fix/val_data.json')),
                     ind_to_classes, ind_to_predicates, vectors, _100_cls_idx_to_vg150_idx)
    test = read_split(json.load(open('/data_local/yutianyu/datasets/VisualGenome/100_100_fix/test_data.json')),
                      ind_to_classes, ind_to_predicates, vectors, _100_cls_idx_to_vg150_idx)
    train = read_split(json.load(open('/data_local/yutianyu/datasets/VisualGenome/100_100_fix/train_data.json')),
                       ind_to_classes, ind_to_predicates, vectors, _100_cls_idx_to_vg150_idx)

    tkz = None
    # if use_bert:
    #     tkz = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    val_dataset = TextOnlyDataset(val, 'val', tkz)
    test_dataset = TextOnlyDataset(test, 'test', tkz)
    train_dataset = TextOnlyDataset(train, 'train', tkz)

    bsz = 50
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True)

    if use_vinvl:
        from transformers.pytorch_transformers import BertTokenizer, BertConfig
        from oscar.modeling.modeling_clever import VinVLTextBaseline
        # Load pretrained model and tokenizer
        config_class, model_class, tokenizer_class = BertConfig, VinVLTextBaseline, BertTokenizer
        checkpoint = 'pretrained_model/pretrained_base/checkpoint-2000000/'
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = False
        # config.hidden_dropout_prob = 0.6
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        print(f'Load from {checkpoint}')
        config.loss_type = "cls"

        model = model_class.from_pretrained(checkpoint, config=config)
        model.to('cuda:1')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    else:
        model = Model().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch = 100
    loss = 0
    last_100_loss = []
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_f1, best_auc, best_recall = 0, 0, ...
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
        np_rec, np_prec, recalls_of_predicate, macro_recall, macro_recall_auc, auc, max_micro_f1, max_macro_f1 = cal_metrics(
            sorted_pred_result,
            val_dataset.facts)

        if auc > best_auc:
            best_f1 = max_micro_f1
            best_auc = auc
            best_recall = np_rec.tolist()
            best_macro_recall = macro_recall.tolist()
            print(
                f'Epoch-{e} best val max_f1={max_micro_f1:.5f}, auc={auc:.5f}, recall@{1000}={np_rec[1000]:.3f}, macro-recall@{1000}={best_macro_recall[1000]:.5f}')

            sorted_pred_result = run_test(model, loss_func, test_dataset, test_dataloader)
            np_rec, np_prec, recalls_of_predicate, macro_recall, macro_recall_auc, auc, max_micro_f1, max_macro_f1 = cal_metrics(
                sorted_pred_result, test_dataset.facts)
            print(f'Test auc={auc}, f1={max_micro_f1}, macro_f1={max_macro_f1}')
            sorted_predicates = [x[2] for x in sorted_pred_result]
            json.dump(np_rec.tolist(), open(f'{prefix}_recalls.json', 'w'))
            json.dump(macro_recall.tolist(), open(f'{prefix}_macro_recalls.json', 'w'))
            assert len(np_rec) == len(macro_recall)
            # print(len(np_rec), '@@@@@@')
            json.dump(sorted_predicates, open(f'{prefix}_predicates.json', 'w'))
