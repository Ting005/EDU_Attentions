import logging
import os.path
from tqdm import tqdm
import pdb
import json
import pickle as pkl
import torch
from models.bert import Bert
from torch import optim

from src.dataloaders.ASP_DICT import MAMS_AC, REST14_ASP_DICT, LAPTOP_ASP_DICT
# from models.han_reg import HANREG
# from models.atae_lstm import ATAE_LSTM_Mul
# from models.as_capsule import AspectCapsule
from models.han_bert_hard_reg import BertHANREG
# from models.as_capsule import AspectCapsule
from models.bert import Bert
import time

from src.trainer import train_multilabel_han, train_multilabel_flat, train_multilabel_han_bert
from src.evaluation import eval_multilabel_han, eval_multilabel_flat, eval_multilabel_han_bert, eval_single_label_bert

# from src.evaluation import train_multilabel_han, evaluate_single_label_wo_asp
import argparse
import numpy as np
import random


def eval_bert():
    from src.dataloaders.singleBertFlat import DatasetLoader
    from src.evaluation import eval_single_label_bert
    const_random_sate = 42
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(const_random_sate)
    random.seed(const_random_sate)
    np.random.seed(seed=const_random_sate)
    torch.cuda.manual_seed(const_random_sate)
    torch.cuda.manual_seed_all(const_random_sate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = json.load(open('./exp_data/mams_sgl.json', 'r'))
    ASP_DICT = MAMS_AC
    model = Bert()
    model.to(compute_device)
    optimizer = model.configure_optimizers()

    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_bert.pth")
    model.to(compute_device)
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    tr_avg_f1, tr_f1s, tr_acc, tr_cfm = eval_single_label_bert(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time) * 1000
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    print('finished')


def eval_ours(dataset):
    from src.dataloaders.multiHAN import DatasetLoader
    from src.evaluation import eval_multilabel_han
    from models.han_reg import HANREG

    ASP_DICT = MAMS_AC
    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    glove_asp = [vocab_embed['glove'][vocab_embed['vocab'][k]] for k, v in ASP_DICT.items()]
    glove_asp = np.array(glove_asp)
    aspect_indexes = [vocab_embed['vocab'][k] for k, v in ASP_DICT.items()]

    # const_random_sate = 42
    # compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(const_random_sate)
    # random.seed(const_random_sate)
    # np.random.seed(seed=const_random_sate)
    # torch.cuda.manual_seed(const_random_sate)
    # torch.cuda.manual_seed_all(const_random_sate)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    model = HANREG(300, 256, 1, glove, glove_asp, .5, ASP_DICT, aspect_indexes)
    model.to(compute_device)

    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_han_reg.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_multilabel_han(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    print('finished')
    return total_training_time


def eval_atae(dataset):
    from src.dataloaders.singleFlat import DatasetLoader
    from models.atae_lstm import ATAE_LSTM
    from src.evaluation import eval_single_label


    ASP_DICT = MAMS_AC
    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    glove_asp = [vocab_embed['glove'][vocab_embed['vocab'][k]] for k, v in ASP_DICT.items()]
    glove_asp = np.array(glove_asp)
    aspect_indexes = [vocab_embed['vocab'][k] for k, v in ASP_DICT.items()]

    const_random_sate = 42
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(const_random_sate)
    random.seed(const_random_sate)
    np.random.seed(seed=const_random_sate)
    torch.cuda.manual_seed(const_random_sate)
    torch.cuda.manual_seed_all(const_random_sate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ASP_DICT = MAMS_AC
    model = ATAE_LSTM(dim_word=300, dim_hidden=256, n_layer=1, n_label=3, glove=glove, embed_dropout=.5,
                      bidirectional=True)
    # model = HANREG(300, 256, 1, glove, glove_asp, .5, ASP_DICT, aspect_indexes)
    model.to(compute_device)

    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_atae_lstm.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_single_label(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    print('finished')
    return total_training_time


def eval_lstm(dataset):
    from src.dataloaders.singleFlat import DatasetLoader
    from models.bilstm import BiLSTM
    from src.evaluation import eval_single_label_wo_asp

    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']

    model = BiLSTM(dim_word=300, dim_hidden=256, n_layer=1, n_label=3, glove=glove, embed_dropout=.5,
                   bidirectional=True)

    # model = HANREG(300, 256, 1, glove, glove_asp, .5, ASP_DICT, aspect_indexes)
    model.to(compute_device)

    test_dataloader = DatasetLoader(batch_size=32, asp_dict=MAMS_AC, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_lstm.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_single_label_wo_asp(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def eval_memoryNet(dataset):
    from src.dataloaders.singleFlat import DatasetLoader
    from models.memnet import MemNet
    from src.evaluation import eval_single_label
    ASP_DICT = MAMS_AC
    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    ASP_DICT = MAMS_AC
    model = MemNet(dim_word=300,  n_label=3, glove=glove)
    model.to(compute_device)
    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_memoryNet.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_single_label(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def eval_HAN(dataset):
    from src.dataloaders.singleHAN import DatasetLoader
    from models.han import HAN
    from src.evaluation import eval_han_single_label
    ASP_DICT = MAMS_AC
    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    ASP_DICT = MAMS_AC
    model = HAN(dim_word=300, dim_hidden=256, n_layer=1, n_label=3, glove=glove, embed_dropout=.5)
    model.to(compute_device)
    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_HAN.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_han_single_label(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def eval_AS_Capsule(dataset):
    from src.dataloaders.multiFlat import DatasetLoader
    from models.as_capsule import AspectCapsule
    from src.evaluation import eval_multilabel_flat
    ASP_DICT = MAMS_AC
    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    ASP_DICT = MAMS_AC
    model = AspectCapsule(300, 256, 1, 3, 8,
                          glove.shape[0], glove, .5, 0, .5,
                          True, 'LSTM', compute_device)
    model.to(compute_device)
    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_As_Capsule.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_multilabel_flat(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def eval_gac(dataset):
    from src.dataloaders.singleFlatGac import DatasetLoader
    from models.gcae import CNN_Gate_Aspect_Text
    from src.evaluation import eval_single_label_gac
    ASP_DICT = MAMS_AC

    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    glove_asp = [vocab_embed['glove'][vocab_embed['vocab'][k]] for k, v in ASP_DICT.items()]
    glove_asp = np.array(glove_asp)
    model = CNN_Gate_Aspect_Text(embed=glove, asp_embed=glove_asp, dim_word=300, dim_asp=300, n_label=3,
                                 kernel_num=100, kernel_sizes='3,4,5')
    model.to(compute_device)
    test_dataloader = DatasetLoader(batch_size=32, asp_dict=ASP_DICT, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_gca.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_single_label_gac(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def eval_CAN(dataset):
    from src.dataloaders.multiFlat import DatasetLoader
    from models.can import CAN
    from src.evaluation import eval_multilabel_flat_can
    ASP_DICT = MAMS_AC
    vocab_embed = pkl.load(open('./exp_data/MAMS_ACSA_vocab_glove.pkl', 'rb'))
    glove = vocab_embed['glove']
    aspect_indexes = [vocab_embed['vocab'][k] for k, v in ASP_DICT.items()]
    model = CAN(300, 256, 1, glove, .5, True, aspect_indexes, len(ASP_DICT))

    # model = HANREG(300, 256, 1, glove, glove_asp, .5, ASP_DICT, aspect_indexes)
    model.to(compute_device)

    test_dataloader = DatasetLoader(batch_size=32, asp_dict=MAMS_AC, input_data=dataset['test'])

    model_path = os.path.join("./model_files/mams_CAN.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_multilabel_flat_can(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def eval_bert(dataset):
    from src.dataloaders.multiFlat import DatasetLoader
    from src.dataloaders.singleBertFlat import DatasetLoader
    from src.evaluation import eval_single_label_bert
    model = Bert()
    model.to(compute_device)
    test_dataloader = DatasetLoader(batch_size=32, asp_dict=MAMS_AC, input_data=dataset['test'])
    model_path = os.path.join("./model_files/mams_bert.pth")
    model.load_state_dict(torch.load(model_path))
    start_time = time.time()
    eval_single_label_bert(compute_device, model, test_dataloader.Data)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time


def ours_bert():
    from src.dataloaders.multiBertHAN_Inference import DatasetLoader
    from models.han_bert_hard_reg import BertHANREG
    from src.evaluation import eval_multilabel_han_bert

    model = BertHANREG()
    model.to(compute_device)
    model_path = os.path.join("./model_files/mams_bert_han_reg.pth")
    model.load_state_dict(torch.load(model_path))

    test_data = pkl.load(open('./exp_data/mams_mul_infer_test.pkl', 'rb'))
    test_dataloader = DatasetLoader(MAMS_AC, test_data)
    asp_ids_and_mask = test_dataloader.get_asp_ids_maks()

    start_time = time.time()
    eval_multilabel_han_bert(compute_device, model, test_dataloader.Data, asp_ids_and_mask)
    total_training_time = (time.time() - start_time)
    print('inference timing:', total_training_time)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters:', pytorch_total_params)
    return total_training_time

if __name__ == '__main__':
    const_random_sate = 42
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(const_random_sate)
    random.seed(const_random_sate)
    np.random.seed(seed=const_random_sate)
    torch.cuda.manual_seed(const_random_sate)
    torch.cuda.manual_seed_all(const_random_sate)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = json.load(open('./exp_data/mams_sgl.json', 'r'))

    timings = []
    for i in range(100):
        total_training_time = ours_bert()
        timings.append(total_training_time)
    print(np.mean(timings))



