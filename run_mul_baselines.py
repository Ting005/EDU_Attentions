import logging
import os.path
from tqdm import tqdm
import pdb
import json
import pickle as pkl
import torch
from torch import optim
import time

# from models.atae_lstm import ATAE_LSTM, AE_LSTM
# from models.bilstm import BiLSTM, AT_LSTM
# from models.memnet import MemNet
from src.dataloaders.multiHAN import DatasetLoader
# from src.dataloaders.multiFlat import DatasetLoader
# from src.dataloaders.multiBertHAN import DatasetLoader

from src.dataloaders.ASP_DICT import MAMS_AC, REST14_ASP_DICT, LAPTOP_ASP_DICT
from models.han_reg import HANREG
# from models.can import CAN
# from models.atae_lstm import ATAE_LSTM_Mul
# from models.as_capsule import AspectCapsule
# from models.han_bert_hard_reg import BertHANREG
# from models.as_capsule import AspectCapsule
from src.trainer import train_multilabel_han, train_multilabel_flat, train_multilabel_han_bert, train_multilabel_flat_can
from src.evaluation import eval_multilabel_han, eval_multilabel_flat, eval_multilabel_han_bert, eval_multilabel_flat_can

# from src.evaluation import train_multilabel_han, evaluate_single_label_wo_asp
import argparse
import numpy as np
import random
import os
from utiil import save_model


def run(FLAGS, compute_device, dataset, vocab_embed, ASP_DICT):
    FLAGS.n_aspect = len(ASP_DICT)
    glove = vocab_embed['glove']
    glove_asp = [vocab_embed['glove'][vocab_embed['vocab'][k]] for k, v in ASP_DICT.items()]
    glove_asp = np.array(glove_asp)
    aspect_indexes = [vocab_embed['vocab'][k] for k, v in ASP_DICT.items()]
    # aspect_indexes = None

    dataloaders = {}
    for data_name, subset in dataset.items():
        dataloaders[data_name] = DatasetLoader(batch_size=FLAGS.batch_size, asp_dict=ASP_DICT, input_data=subset)

    # del vocab_embed
    del dataset
    # pdb.set_trace()
    # model = CAN(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, glove, FLAGS.embed_dropout, FLAGS.bidirectional, aspect_indexes, len(ASP_DICT))
    model = HANREG(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, glove, glove_asp, FLAGS.embed_dropout, ASP_DICT, aspect_indexes)
    # model = ATAE_LSTM_Mul(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, glove, FLAGS.embed_dropout, FLAGS.bidirectional, aspect_indexes, len(ASP_DICT))
    #dim_word, dim_hidden, n_layer, n_label, glove, embed_dropout, bidirectional, all_asp_idxes, n_aspect
    # model = AspectCapsule(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, FLAGS.n_label, FLAGS.n_aspect,
    #                       glove.shape[0], glove, FLAGS.embed_dropout, FLAGS.cell_dropout, FLAGS.final_dropout,
    #                       FLAGS.bidirectional, FLAGS.rnn_type, compute_device)
    # asp_inputs = DatasetLo ader.get_asp_ids(ASP_DICT)
    # model = BertHANREG()
    # model = AspectCapsule(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, FLAGS.n_label, FLAGS.n_aspect,
    #                       glove.shape[0], glove, FLAGS.embed_dropout, FLAGS.cell_dropout, FLAGS.final_dropout,
    #                       FLAGS.bidirectional, FLAGS.rnn_type, compute_device)

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print('model size', pytorch_total_params)
    model.to(compute_device)
    # print(model)
    optimizer = getattr(optim, FLAGS.optim_type)([{'params': model.base_params, 'weight_decay': FLAGS.weight_decay},
                                                  {'params': model.embed.parameters(), 'lr': FLAGS.lr_word_vector, 'weight_decay': 0},
                                                  {'params': model.asp_embed.parameters(), 'lr': FLAGS.lr_word_vector, 'weight_decay': 0}
                                                  ], lr=FLAGS.learning_rate)

    # optimizer = getattr(optim, FLAGS.optim_type)(model.parameters(), lr=FLAGS.learning_rate)


    # config for bert
    # optimizer = model.configure_optimizers(learning_rate_b=2e-5, lr=1e-3, weight_decay=.0, is_freeze_bert=False)
    curr_acc, curr_f1, best_acc, best_f1 = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}, \
                                           {'train': 0, 'test': 0,'dev': 0, 'hard': 0}, \
                                           {'train': 0, 'test': 0, 'dev': 0, 'hard': 0},  \
                                           {'train': 0, 'test': 0,'dev': 0, 'hard': 0}

    curr_acc = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}
    curr_f1s = {'train': '', 'test': '', 'dev': '', 'hard': ''}
    curr_ma_f1 = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}
    best_acc = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}
    best_ma_f1 = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}
    output_msg = {}

    rpt_test_acc = 0
    rpt_test_ma_f1 = 0
    rpt_hard_acc = 0
    rpt_hard_ma_f1 = 0
    rpt_dev_ma_f1 = 0
    counter = 0
    patient = 0
    check_point_count = 0

    with tqdm(total=100000, desc='') as pbar:
        for epoch in range(0, 100000):
            if patient == 10:
                print()
                pdb.set_trace()
                pdb.set_trace()
            if epoch > 0 and counter == 0:
                patient += 1
            else:
                patient = 0

            counter = 0
            for step, inputs in enumerate(dataloaders['train'].Data, start=1):
                # train_multilabel_han_bert(compute_device, model, optimizer, inputs, asp_inputs, len(ASP_DICT))
                train_multilabel_han(compute_device, model, optimizer, inputs, len(ASP_DICT))
                # train_multilabel_flat_can(compute_device, model, optimizer, inputs)
                # train_multilabel_flat(compute_device, model, optimizer, inputs)
                if step % FLAGS.per_checkpoint == 0:
                    # if check_point_count > 10:
                    for data_name, eval_data in dataloaders.items():
                        if eval_data is None: continue

                        # avg_f1, f1s, acc, cfm = evaluate_single_label_wo_asp(compute_device, model, eval_data.Data)
                        # avg_f1, f1s, acc, cfm = eval_multilabel_han_bert(compute_device, model, eval_data.Data, asp_inputs)
                        result = eval_multilabel_han(compute_device, model, eval_data.Data)
                        # result = eval_multilabel_flat_can(compute_device, model, eval_data.Data)
                        # result = eval_multilabel_flat(compute_device, model, eval_data.Data)
                        curr_acc[data_name] = result['acc']
                        curr_f1s[data_name] = result['f1s']
                        curr_ma_f1[data_name] = result['macro_f1']

                    # update reporting metric
                    if curr_ma_f1['dev'] > best_ma_f1['dev']:
                        counter += 1
                        rpt_test_acc = curr_acc['test']
                        rpt_test_f1s = curr_f1s['test']
                        rpt_test_ma_f1 = curr_ma_f1['test']
                        rpt_dev_ma_f1 = curr_ma_f1['dev']
                        save_model(_model_name=FLAGS.model, _data_name=FLAGS.data, _model=model)
                        if dataloaders.__contains__('hard'):
                            rpt_hard_acc = curr_acc['hard']
                            rpt_hard_ma_f1 = curr_ma_f1['hard']
                            rpt_hard_f1s = curr_f1s['hard']

                            output_msg['Rpt'] = "H:{:.4f}/{:.4f}({}),T:{:.4f}/{:.4f}({}), D:{:.4f}".format(rpt_hard_acc, rpt_hard_ma_f1, rpt_hard_f1s,
                                                                                                           rpt_test_acc, rpt_test_ma_f1, rpt_test_f1s,
                                                                                                           rpt_dev_ma_f1)
                        else:
                            output_msg['Rpt'] = "T:{:.4f}/{:.4f}({}), D:{:.4f}".format(rpt_test_acc, rpt_test_ma_f1, rpt_test_f1s, rpt_dev_ma_f1)

                    # update best figure
                    for data_name in dataloaders.keys():
                        best_acc[data_name] = max(curr_acc[data_name], best_acc[data_name])
                        best_ma_f1[data_name] = max(curr_ma_f1[data_name], best_ma_f1[data_name])

                        output_msg[data_name[0:2]] = "{:.4f}/{:.4f}, {:.4f}/{:.4f}".format(curr_acc[data_name], best_acc[data_name],
                                                                                           curr_ma_f1[data_name], best_ma_f1[data_name])

                    # output_msg = {
                    #     "Rpt":"H:{:.4f}/{:.4f},T:{:.4f}/{:.4f}, D:{:.4f}".format(rpt_hard_acc, rpt_hard_f1, rpt_test_acc, rpt_test_f1, rpt_dev_f1),
                    #     "Tr":"{:.4f}/{:.4f}, {:.4f}/{:.4f}".format(curr_acc['train'], best_acc['train'], curr_f1['train'], best_f1['train']),
                    #     "dev": "{:.4f}/{:.4f},{:.4f}/{:.4f}".format(curr_acc['dev'], best_acc['dev'], curr_f1['dev'], best_f1['dev']),
                    #     "test": "{:.4f}/{:.4f}, {:.4f}/{:.4f}".format(curr_acc['test'], best_acc['test'], curr_f1['test'], best_f1['test']),
                    #     "hard": "{:.4f}/{:.4f}, {:.4f}/{:.4f}".format(curr_acc['hard'], best_acc['hard'], curr_f1['hard'], best_f1['hard'])
                    # }
                    logging.info(output_msg)
                    pbar.set_postfix(output_msg)
                    pbar.update(1)
            check_point_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_size', type=int, default=32768)
    parser.add_argument('--dim_word', type=int, default=300, choices=[300])
    parser.add_argument('--dim_hidden', type=int, default=256, choices=[256])
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_label', type=int, default=3, choices=[3])
    parser.add_argument('--n_aspect', type=int, default=22, choices=[5])
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_word_vector', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--embed_dropout', type=float, default=0.5)
    parser.add_argument('--cell_dropout', type=float, default=0.5)
    parser.add_argument('--final_dropout', type=float, default=0.5)
    parser.add_argument('--lambda1', type=float, default=0.5)

    parser.add_argument('--iter_num', type=int, default=8 * 320)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--per_checkpoint', type=int, default=8*2)
    # parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
    parser.add_argument('--data', type=str, default='rest', choices=['rest', 'laptop', 'mams'])
    parser.add_argument('--const_random_sate', type=int, default=50)
    # parser.add_argument('--path_wordvec', type=str, default='vectors.glove.840B.txt')
    parser.add_argument('--model', type=str, default='han_reg')
    parser.add_argument('--gpu', type=int, default=0)
    FLAGS = parser.parse_args()
    logging.basicConfig(filename='./log/{}_{}.log'.format(FLAGS.data, FLAGS.model), level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    print('seed everything')

    torch.manual_seed(FLAGS.const_random_sate)

    random.seed(FLAGS.const_random_sate)
    np.random.seed(seed=FLAGS.const_random_sate)

    torch.manual_seed(FLAGS.const_random_sate)
    torch.cuda.manual_seed(FLAGS.const_random_sate)
    torch.cuda.manual_seed_all(FLAGS.const_random_sate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    compute_device = torch.device("cuda:{}".format(FLAGS.gpu) if torch.cuda.is_available() else "cpu")
    # use_gpu = torch.cuda.is_available()
    # pdb.set_trace()
    print('compute device:', compute_device)
    print(FLAGS)
    ASP_DICT, data, vocab_embed = None, None, None
    if FLAGS.data == 'mams':
        data = json.load(open('./exp_data/mams_mul.json', 'r'))
        vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
        ASP_DICT = MAMS_AC
    elif FLAGS.data == 'rest':
        data = json.load(open('./exp_data/rest14_mul.json', 'r'))
        vocab_embed = pkl.load(open('./exp_data/rest14_vocab_glove.pkl', 'rb'))
        ASP_DICT = REST14_ASP_DICT
    elif FLAGS.data == 'laptop':
        data = json.load(open('./exp_data/laptop15_mul.json', 'r'))
        vocab_embed = pkl.load(open('./exp_data/laptop15_vocab_glove.pkl', 'rb'))
        ASP_DICT = LAPTOP_ASP_DICT

    run(FLAGS, compute_device, data, vocab_embed, ASP_DICT)

