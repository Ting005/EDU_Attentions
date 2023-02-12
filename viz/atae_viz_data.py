import pickle

from tqdm import tqdm
import pdb
import json
import pickle as pkl
import torch
from torch import optim
from src.dataloaders.singleFlat import DatasetLoader
from src.dataloaders.ASP_DICT import REST14_ASP_DICT as ASP_DICT
from models.atae_lstm import ATAE_LSTM
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


def evaluation(_compute_device, _model, _eval_data, _aspect_dict):
    with torch.no_grad():
        lst_all_inst = []
        lst_y_pred, lst_y_true = [], []
        for step, inputs in enumerate(_eval_data):
            batch_sent, num_sent_words, batch_asp_idxes, batch_pol_labels, batch_text = inputs
            pol_logits, word_att_score = _model(batch_sent.to(_compute_device), num_sent_words, batch_asp_idxes.to(_compute_device))

            y_sem_pred = np.argmax(pol_logits.cpu().numpy(), axis=-1).tolist()
            y_sem_true = batch_pol_labels.numpy().tolist()

            lst_y_true.extend(y_sem_true)
            lst_y_pred.extend(y_sem_pred)

            batch_size, max_num_words = batch_sent.size()

            for i in range(batch_size):
                text = batch_text[i]
                word_score = word_att_score[i, :].cpu().numpy().tolist()
                aspect = _aspect_dict[batch_asp_idxes[i].item()]
                scores = ['{:.4f}'.format(v) for v in word_score]
                inst = {'text': text, 'aspect': aspect, 'scores': scores, 'sem_pred': y_sem_pred[i], 'sem_true': y_sem_true[i]}
                lst_all_inst.append(inst)

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

    dataset = json.load(open('../exp_data/rest14_sgl.json', 'r'))
    vocab_embed = pkl.load(open('../exp_data/rest14_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    # glove_asp = [vocab_embed['glove'][vocab_embed['vocab'][k]] for k, v in ASP_DICT.items()]
    aspect_dict = {vocab_embed['vocab'][k]: k for k, v in ASP_DICT.items()}

    print(len(dataset['test']), len(dataset['hard']))
    test_dataloader = DatasetLoader(batch_size=512, asp_dict=ASP_DICT, input_data=dataset['test'])
    hard_dataloader = DatasetLoader(batch_size=512, asp_dict=ASP_DICT, input_data=dataset['hard'])
    del dataset

    model = ATAE_LSTM(dim_word=300, dim_hidden=256, n_layer=1, n_label=3, glove=glove, embed_dropout=0.5, bidirectional=True)

    model.to(compute_device)
    model = load_model(_model_name='atae_lstm', _data_name='rest', _model=model)
    model.eval()

    test_avg_f1, test_f1s, test_acc, test_cfm, test_lst_all_inst = evaluation(compute_device, model, test_dataloader.Data, aspect_dict)
    hard_avg_f1, hard_f1s, hard_acc, hard_cfm, hard_lst_all_inst = evaluation(compute_device, model, hard_dataloader.Data, aspect_dict)

    pickle.dump({'hard': hard_lst_all_inst, 'test': test_lst_all_inst}, open('../model_files/rest_atae.raw', 'wb'))
    print('score finished.')


def viz_attention():
    rest_han = pickle.load(open('../model_files/rest_atae.raw', 'rb'))
    hard = rest_han['hard']
    # test = rest_han['test']
    # inst = {'text': text, 'aspect': aspect, 'scores': scores, 'sem_pred': y_sem_pred[i], 'sem_true': y_sem_true[i]}
    for idx, inst in enumerate(hard):
        text = inst['text']
        aspect = inst['aspect']
        scores = inst['scores']
        sem_pred = inst['sem_pred']
        sem_true = inst['sem_true']

        inst_out = []
        for w_idx, word in enumerate(text):
            inst_out.append('{}({})'.format(word, scores[w_idx]))
        print('-' * 100)
        print(aspect, sem_true, sem_pred)
        print(' '.join(inst_out))


if __name__ == "__main__":
    # store_attention_scores()
    viz_attention()
