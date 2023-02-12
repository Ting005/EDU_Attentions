import pickle

from tqdm import tqdm
import pdb
import json
import pickle as pkl
import torch
from torch import optim
from src.dataloaders.multiHAN import DatasetLoader
from src.dataloaders.ASP_DICT import REST14_ASP_DICT as ASP_DICT
from models.han_reg import HANREG
import argparse
import numpy as np
import random
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

polarity_labels = ['negative', 'neutral', 'positive']


def load_model(_data_name, _model_name, _model):
    model_path = os.path.join('../model_files/{}_{}.pth'.format(_data_name, _model_name))
    _model.load_state_dict(torch.load(model_path))
    return _model


def evaluation(_compute_device, _model, _eval_data):
    with torch.no_grad():
        lst_all_inst = []
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            batch_seg, segs, num_word_seg, num_seg, sent_order, batch_asplabels, num_asp, batch_pol_labels = inputs
            batch_seg, sent_order, num_asp = [v.to(_compute_device) for v in [batch_seg, sent_order, num_asp]]
            pol_logits, asp_prob, var, [edu_att_score, word_att_score] = _model(batch_seg, num_word_seg, num_seg, sent_order, num_asp)
            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1)
            y_sem_true = batch_pol_labels.numpy()
            y_asp_ture = batch_asplabels.numpy()
            print(y_asp_ture.shape)
            batch_size, num_asp = y_asp_ture.shape
            edu_att_score = edu_att_score.cpu().numpy()
            word_att_score = word_att_score.cpu().numpy()

            for i in range(batch_size):
                text = segs[i]
                word_score = word_att_score[i, :, :, :]  # num_asp, num_edu, num_word
                edu_score = edu_att_score[i, :, :] # num_asp, num_edu
                total_seg = num_seg[i].cpu().numpy()
                asp_labels = [[*ASP_DICT][idx] for idx, v in enumerate(y_asp_ture[i].tolist()) if v == 1]
                sem_labels = [polarity_labels[v] for idx, v in enumerate(y_sem_true[i].tolist()) if v != -1]
                labels = ['{}:{}'.format(a, b) for a, b in zip(asp_labels, sem_labels)]

                inst = {'text': text, 'word_score': word_score, 'edu_score': edu_score, 'total_seg': total_seg, 'labels': labels,
                        'asp_true': y_asp_ture[i].tolist(),
                        'sem_true': y_sem_true[i].tolist()}
                lst_all_inst.append(inst)

            for s_y_t, s_y_p, a_y_t in zip(y_sem_true, y_sem_pred, y_asp_ture):
                for y_t, y_p, y_t_asp in zip(s_y_t, s_y_p, a_y_t):
                    if y_t_asp > 0:
                        lst_y_true.append(y_t)
                        lst_y_pred.append(y_p)

        avg_f1 = f1_score(y_true=lst_y_true, y_pred=lst_y_pred, average='macro')
        f1s = f1_score(y_true=lst_y_true, y_pred=lst_y_pred, average=None, labels=[1, 0, 2])
        acc = accuracy_score(y_true=lst_y_true, y_pred=lst_y_pred)
        cfm = confusion_matrix(y_true=lst_y_true, y_pred=lst_y_pred, labels=[1, 0, 2])
        print(avg_f1, acc)
        # pdb.set_trace()
        return avg_f1, f1s, acc, cfm, lst_all_inst


def store_attention_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument('--const_random_sate', type=int, default=42)
    FLAGS = parser.parse_args()
    print(FLAGS)
    torch.manual_seed(FLAGS.const_random_sate)
    random.seed(FLAGS.const_random_sate)
    np.random.seed(seed=FLAGS.const_random_sate)
    torch.manual_seed(FLAGS.const_random_sate)
    torch.cuda.manual_seed(FLAGS.const_random_sate)
    torch.cuda.manual_seed_all(FLAGS.const_random_sate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    compute_device = torch.device("cuda:0")

    dataset = json.load(open('../exp_data/rest14_mul.json', 'r'))
    vocab_embed = pkl.load(open('../exp_data/rest14_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    glove_asp = [vocab_embed['glove'][vocab_embed['vocab'][k]] for k, v in ASP_DICT.items()]
    glove_asp = np.array(glove_asp)
    aspect_indexes = [vocab_embed['vocab'][k] for k, v in ASP_DICT.items()]
    print(len(dataset['test']), len(dataset['hard']))
    test_dataloader = DatasetLoader(batch_size=512, asp_dict=ASP_DICT, input_data=dataset['test'])
    hard_dataloader = DatasetLoader(batch_size=512, asp_dict=ASP_DICT, input_data=dataset['hard'])
    del dataset

    model = HANREG(dim_word=300, dim_hidden=256, n_layer=1, glove=glove, asp_glove=glove_asp, embed_dropout=0.5,
                   ASP_DICT=ASP_DICT, aspect_indexes=aspect_indexes)
    model.to(compute_device)
    model = load_model(_model_name='han_reg_v2', _data_name='rest', _model=model)
    model.eval()

    test_avg_f1, test_f1s, test_acc, test_cfm, test_lst_all_inst = evaluation(compute_device, model, test_dataloader.Data)
    hard_avg_f1, hard_f1s, hard_acc, hard_cfm, hard_lst_all_inst = evaluation(compute_device, model, hard_dataloader.Data)
    pdb.set_trace()
    # pickle.dump({'hard': hard_lst_all_inst, 'test': test_lst_all_inst}, open('../model_files/rest_han_reg_v2.raw', 'wb'))
    print('score finished.')


def viz_attention():
    rest_han = pickle.load(open('../model_files/rest_han_reg.raw', 'rb'))
    hard = rest_han['hard']
    # test = rest_han['test']
    for idx, inst in enumerate(hard):
        segs = inst['text']
        word_score = inst['word_score']
        edu_score = inst['edu_score']
        total_seg = inst['total_seg']
        labels = inst['labels']
        asp_true = inst['asp_true']
        sem_true = inst['sem_true']

        num_asp, num_seg, num_word = word_score.shape
        _, num_seg = edu_score.shape

        asp_true_idx = {a_i: [*ASP_DICT][a_i] for a_i, a_v in enumerate(asp_true) if a_v == 1.0}
        print('=' * 100)
        print('|'.join(segs))
        print(labels)
        print('-' * 10)
        for seg_idx, seg in enumerate(segs):
            print(seg)
            seg_terms = seg.split()
            edu_word_scores = []
            for a_idx, asp in asp_true_idx.items():
                asp_edu_score = '{}({:.2f})'.format(asp, edu_score[a_idx, seg_idx] * 100)

                asp_seg_word_scores = word_score[a_idx, seg_idx].tolist()
                asp_seg_score_text = []
                for t_idx, term in enumerate(seg_terms):
                    asp_seg_score_text.append('{}({:.2f})'.format(term, asp_seg_word_scores[t_idx]* 100))

                asp_seg_text = '{} --> {}'.format(asp_edu_score, ' '.join(asp_seg_score_text))
                print(asp_seg_text)
            print('-' * 20)

        # pdb.set_trace()


def attention_accuracy():
    def __search_annotation(_data, _search_string):
        found = False
        max_len_seg_idx = np.argmax([len(s) for s in _search_string])
        for _inst in _data:
            if _search_string[max_len_seg_idx] in _inst['text']:
                found = True
                label_inst = _inst
                break
        assert found
        segs = label_inst['segs']
        segs_labels = label_inst['segs_labels']
        return {'segs': segs, 'segs_labels': segs_labels}

    rest_data = json.load(open('../exp_data/rest14_mul.json', 'r'))
    rest_data_test, rest_data_hard = rest_data['test'],  rest_data['hard']
    rest_han = pickle.load(open('../model_files/rest_han_reg_v2.raw', 'rb'))
    hard = rest_han['hard']
    test = rest_han['test']
    cnt = 0
    for idx, inst in enumerate(test):
        segs = inst['text']
        word_score = inst['word_score']
        edu_score = inst['edu_score']
        total_seg = inst['total_seg']
        labels = inst['labels']
        asp_true = inst['asp_true']
        sem_true = inst['sem_true']
        if len(segs) == 1 or sum(asp_true) == 1: continue
        cnt += 1
        '''
        searching annotation
        '''
        anno = __search_annotation(rest_data_test, segs)

        num_asp, num_seg, num_word = word_score.shape
        _, num_seg = edu_score.shape

        asp_true_idx = {a_i: [*ASP_DICT][a_i] for a_i, a_v in enumerate(asp_true) if a_v == 1.0}
        print('=' * 50, cnt, '='*50)
        print(anno)
        print('|'.join(segs))
        print(labels)
        print('-' * 10)
        for seg_idx, seg in enumerate(segs):
            print(seg)
            seg_terms = seg.split()
            edu_word_scores = []
            for a_idx, asp in asp_true_idx.items():
                asp_edu_score = '{}({:.2f})'.format(asp, edu_score[a_idx, seg_idx] * 100)

                asp_seg_word_scores = word_score[a_idx, seg_idx].tolist()
                asp_seg_score_text = []
                for t_idx, term in enumerate(seg_terms):
                    asp_seg_score_text.append('{}({:.2f})'.format(term, asp_seg_word_scores[t_idx] * 100))

                asp_seg_text = '{} --> {}'.format(asp_edu_score, ' '.join(asp_seg_score_text))
                print(asp_seg_text)
            print('-' * 20)
    print(cnt)
    pass


def calculate_attention_scores():
    scores = open('accuracy_attention', 'r').readlines()
    # dict_result = {'SR': [], 'FD': [], 'AM': [], 'PR': [], 'MI': []}
    lst_true, lst_pred = [], []
    num_edu = 0
    for line in scores:
        if line.startswith('***'):
            label, pred = line.replace('***', '').strip().split('-')
            lst_true.append(label)
            lst_pred.append(pred)
            print(label, pred)
            num_edu += 1
    print(classification_report(y_true=lst_true, y_pred=lst_pred, digits=4))
    print(accuracy_score(y_true=lst_true, y_pred=lst_pred))
    print(f1_score(y_true=lst_true, y_pred=lst_pred, average='micro'))
    print(num_edu)
    pdb.set_trace()

if __name__ == "__main__":
    store_attention_scores()
    # calculate_attention_scores()
    # attention_accuracy()
    # viz_attention()