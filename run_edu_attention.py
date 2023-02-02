import json
import pdb
import pickle as pkl
import torch
from torch import optim

from src.EDUDataloader import DatasetLoader
from src.ASP_DICT import MAMS_AC, REST14_ASP_DICT, LAPTOP_ASP_DICT
from models.EDU_Attention_softmax import EDU_Attention
from src.trainer import train_edu_attention
from src.evaluation import eval_edu_attention
import argparse
import numpy as np
import random


def run(FLAGS, compute_device, dataset, vocab_embed, ASP_DICT):
    FLAGS.n_aspect = len(ASP_DICT)
    glove = vocab_embed['glove']
    glove_asp = [vocab_embed['glove'][vocab_embed['vocab'].get(k, vocab_embed['vocab']['<unk>'])] for k, v in ASP_DICT.items()]

    glove_asp = np.array(glove_asp)
    # pdb.set_trace() #vocab_embed['vocab']['na'] #3893
    aspect_indexes = [vocab_embed['vocab'].get(k, vocab_embed['vocab']['<unk>']) for k, v in ASP_DICT.items()]
    dataloaders = {}
    for data_name, subset in dataset.items():
        dataloaders[data_name] = DatasetLoader(batch_size=FLAGS.batch_size, asp_dict=ASP_DICT, input_data=subset)

    del dataset
    model = EDU_Attention(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, glove, glove_asp, ASP_DICT, aspect_indexes)
    model.to(compute_device)
    # print(model)
    optimizer = getattr(optim, FLAGS.optim_type)([{'params': model.base_params, 'weight_decay': FLAGS.weight_decay},
                                                  {'params': model.embed.parameters(), 'lr': FLAGS.lr_word_vector, 'weight_decay': 0},
                                                  {'params': model.asp_embed.parameters(), 'lr': FLAGS.lr_word_vector, 'weight_decay': 0}
                                                  ], lr=FLAGS.learning_rate)

    curr_ma_f1 = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}
    best_ma_f1 = {'train': 0, 'test': 0, 'dev': 0, 'hard': 0}
    curr_result = {}
    counter = 0
    patient = 0
    WARNUP_EPOCHS = 5
    for epoch in range(0, FLAGS.max_run):
        if epoch > WARNUP_EPOCHS and counter == 0:
            patient += 1
        else:
            patient = 0
        if patient >= 10:
            print('epoch-->:', epoch)
            break
        print('epoch:', epoch)
        counter = 0
        for step, inputs in enumerate(dataloaders['train'].Data, start=1):
            train_edu_attention(compute_device, model, optimizer, inputs, len(ASP_DICT))
            if step % FLAGS.per_checkpoint == 0 and epoch > WARNUP_EPOCHS:
                for data_name, eval_data in dataloaders.items():
                    if eval_data is None: continue
                    result = eval_edu_attention(compute_device, model, eval_data.Data)
                    curr_ma_f1[data_name] = result['macro_f1']
                    curr_result[data_name] = result
                    # print('{} --> macro-F1:{:.4f}, accuracy: {:.4f}, f1s: {}'.format(data_name, result['macro_f1'], result['acc'], result['f1s']))
                    # print(result['cfm'])
                if curr_ma_f1['dev'] >= best_ma_f1['dev']:
                    counter += 1
                    # print('-' * 50)
                    print('-' * 50)
                    for data_name in dataloaders.keys():
                        if data_name == 'dev': continue
                        print('{} --> macro-F1:{:.4f}, accuracy: {:.4f}, f1s: {}'.format(data_name, curr_result[data_name]['macro_f1'], curr_result[data_name]['acc'], curr_result[data_name]['f1s']))
                        print(curr_result[data_name]['cfm'])
                # update best figure
                for data_name in dataloaders.keys():
                    best_ma_f1[data_name] = max(curr_ma_f1[data_name], best_ma_f1[data_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_word', type=int, default=300, choices=[300])
    parser.add_argument('--dim_hidden', type=int, default=256, choices=[256])
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_label', type=int, default=3, choices=[3])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_word_vector', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--embed_dropout', type=float, default=0.5)
    parser.add_argument('--cell_dropout', type=float, default=0.5)
    parser.add_argument('--final_dropout', type=float, default=0.5)
    parser.add_argument('--iter_num', type=int, default=8 * 320)
    parser.add_argument('--max_run', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--per_checkpoint', type=int, default=8 * 2)  # for rest, laptop set to 8; for mamas set to 8 * 2
    parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
    parser.add_argument('--data', type=str, default='mams', choices=['rest', 'laptop', 'mams'])
    parser.add_argument('--const_random_sate', type=int, default=50)
    parser.add_argument('--model', type=str, default='han_reg')
    parser.add_argument('--gpu', type=int, default=0)
    FLAGS = parser.parse_args()

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
        data = json.load(open('./exp_data/laptop15_mul_w_hard.json', 'r'))
        vocab_embed = pkl.load(open('./exp_data/laptop15_vocab_glove.pkl', 'rb'))
        ASP_DICT = LAPTOP_ASP_DICT

    run(FLAGS, compute_device, data, vocab_embed, ASP_DICT)
