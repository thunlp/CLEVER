import json
import torch
import random
import argparse
import numpy as np
import sklearn.metrics
from collections import defaultdict
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import numpy
import pickle


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

    best_threshold_idx = (2 * np_prec * np_rec / (np_prec + np_rec + 1e-20)).argmax()
    best_threshold = sorted_results[best_threshold_idx][3].item()
    # print(f'p={precisions[best_threshold_idx]}, r={recalls[best_threshold_idx]}')

    pred_result_vec = score_vec >= best_threshold
    valid_p = list(num_facts_of_predicates.keys())
    # print(len(valid_p))
    # print(label_vec[:, valid_p].shape)
    max_macro_f1 = sklearn.metrics.f1_score(label_vec[:, valid_p],
                                            pred_result_vec[:, valid_p],
                                            average='macro')
    # print(len(auc_of_predicate))
    macro_auc = sum(auc_of_predicate.values()) / len(auc_of_predicate)

    max_macro_f1 = sum(max_f1_of_predicate.values()) / len(max_f1_of_predicate)

    macro_p = sum(np_precision_of_predicate.values(), np.zeros(len(np_recall_of_predicates[40]))) / len(
        np_precision_of_predicate)
    return label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, max_macro_f1, numpy.array(
        pr_curve_labels), numpy.array(pr_curve_predictions)


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
        }
        items.append(item)
    return items


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default='1e-5')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--mAUC_weight', type=int, default=1)
    args = parser.parse_args()

    ind_to_classes, ind_to_predicates, _, _100_cls_idx_to_vg150_idx = load_info(f'{args.data_dir}/vg_dict.json')
    ind_to_predicates[0] = 'background'

    classes = list(map(str, range(101)))

    val_bag_data = json.load(open(f'{args.data_dir}/val_bag_data.json'))
    test_bag_data = json.load(open(f'{args.data_dir}/test_bag_data.json'))
    train_bag_data = json.load(open(f'{args.data_dir}/train_bag_data.json'))
    val = read_split(val_bag_data, ind_to_classes, ind_to_predicates, _100_cls_idx_to_vg150_idx)
    test = read_split(test_bag_data, ind_to_classes, ind_to_predicates, _100_cls_idx_to_vg150_idx)
    train = read_split(train_bag_data, ind_to_classes, ind_to_predicates, _100_cls_idx_to_vg150_idx)

    val_dataset = [
        InputExample(guid=i, text_a=item['subject'], text_b=item['object']) for i, item in enumerate(val)
    ]
    test_dataset = [
        InputExample(guid=i, text_a=item['subject'], text_b=item['object']) for i, item in enumerate(test)
    ]
    train_dataset = [
        InputExample(guid=i, text_a=item['subject'], text_b=item['object']) for i, item in enumerate(train)
    ]

    plm_bert, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-uncased")

    predicate_token_ids = [tokenizer.encode(pred, add_special_tokens=False) for pred in ind_to_predicates]
    predicate_tokens = [[tokenizer.ids_to_tokens[idx] for idx in ids] for ids in predicate_token_ids]

    promptTemplate = ManualTemplate(
        text='{"placeholder": "text_a"} {"mask"} {"placeholder": "text_b"}',
        tokenizer=tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            c: predicate_tokens[int(c)] for c in classes
        },
        tokenizer=tokenizer,
        multi_token_handler='mean'
    )

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm_bert,
        verbalizer=promptVerbalizer,
    ).cuda()

    val_loader = PromptDataLoader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.bsz
    )

    test_loader = PromptDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.bsz
    )

    train_loader = PromptDataLoader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.bsz,
        shuffle=True
    )

    # Now the training is standard
    from transformers import AdamW, get_linear_schedule_with_warmup

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    test_facts = set()
    for item in test:
        for label in item['label']:
            if label:
                test_facts.add((item['sub_id'], item['obj_id'], label))
    print(f'test facts are {len(test_facts)}')

    val_facts = set()
    for item in val:
        for label in item['label']:
            if label:
                val_facts.add((item['sub_id'], item['obj_id'], label))
    print(f'val facts are {len(val_facts)}')

    train_facts = set()
    for item in train:
        for label in item['label']:
            if label:
                train_facts.add((item['sub_id'], item['obj_id'], label))
    print(f'train facts are {len(train_facts)}')


    def eval(model, split='test'):
        loader = test_loader if split == 'test' else val_loader
        data = test if split == 'test' else val
        facts = test_facts if split == 'test' else val_facts

        model.eval()
        with torch.no_grad():
            pred_result = []
            for batch in loader:
                logits = model(batch.cuda())
                scores = logits.softmax(-1).cpu()
                indices = batch['guid']
                assert len(indices) == len(scores)
                assert scores.shape[-1] == 101
                for idx, score in zip(indices, scores):
                    for r in range(1, 101):
                        pred_result.append((
                            data[idx]['sub_id'],
                            data[idx]['obj_id'],
                            r,
                            score[r]
                        ))
        sorted_pred_result = sorted(pred_result, key=lambda x: x[3], reverse=True)
        label_vec, pred_result_vec, np_rec, np_prec, macro_p, np_recall_of_predicates, macro_auc, auc, max_micro_f1, max_macro_f1, \
        pr_curve_labels, pr_curve_predictions = cal_metrics(sorted_pred_result, facts)
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


    print(f'Epoch-0, LAMA Result:')
    result = eval(promptModel, 'test')
    print(f'Epoch-0 auc: {result["auc"]:.4f}, m_auc: {result["macro_auc"]:.4f}, '
          f'micro-f1:{result["max_micro_f1"]:.4f}, '
          f'macro-f1:{result["max_macro_f1"]:.4f}, p@2%:{result["p@2%"]:.4f}, '
          f'mp@2%:{result["mp@2%"]:.4f}')
    pickle.dump(result['results'], open(f'{args.output_dir}/BERT_LAMA.json', 'wb'))

    best_test_auc = 0
    best_score = 0

    for epoch in range(args.epoch):
        tot_loss = 0
        for step, inputs in enumerate(train_loader):
            logits = promptModel(inputs.cuda())
            indices = inputs['guid']
            labels = torch.tensor([random.choice(labels) for labels in [train[_]['label'] for _ in indices]]).cuda()
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step and step % 400 == 0:
                print("Epoch {}-{}/{}, average loss: {}".format(epoch, step, len(train_loader), tot_loss / (step + 1)),
                      flush=True)
        val_result = eval(promptModel, 'val')

        score = val_result['auc'] + val_result['macro_auc']
        print(f'Epoch-{epoch}, Val-score {score}, auc={val_result["auc"]}, mauc={val_result["macro_auc"]}')
        if score > best_score:
            best_score = score
            print(f'Epoch-{epoch}, Test')
            result = eval(promptModel, 'test')
            print(f'Epoch-{epoch} auc: {result["auc"]:.4f}, m_auc: {result["macro_auc"]:.4f}, '
                  f'micro-f1:{result["max_micro_f1"]:.4f}, '
                  f'macro-f1:{result["max_macro_f1"]:.4f}, p@2%:{result["p@2%"]:.4f}, '
                  f'mp@2%:{result["mp@2%"]:.4f}')
            json.dump(result['results'], open(f'{args.output_dir}/BERT_prompt_results.json'))
        promptModel.train()
